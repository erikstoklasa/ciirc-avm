"""Semi-automatic vein segmentation: Frangi vesselness -> EM -> random walker.

Turns a DSA source image into a ``main.py``-ready binary ``.nii`` without any
trained model. Per image:

    image  --grayscale + right-panel crop-->  panel
           --Frangi vesselness + Canny edges (Stage 1)
           --FG/BG scribbles (weak user input)
           --Gaussian-Mixture EM per class (Stage 2)  -->  posterior P(vessel)
           --random-walker regularization (scribbles + posterior soft seeds)
           --morphological cleanup  -->  binary mask
           --paste back + spacing  -->  data/{case}.nii  (W x H x 1, uint8)

The output ``.nii`` drops straight into ``main.py`` (you still click start/end
and the existing FMM/metrics code runs unchanged), exactly like ``nnunet/``.

Usage:
    uv run python edge_em/segment.py pre-data/10m.jpg
    uv run python edge_em/segment.py pre-data/10m.png --spacing 0.0568
    uv run python edge_em/segment.py pre-data/ -o /tmp/out      # whole folder
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from scipy.special import logsumexp  # noqa: E402
from skimage.measure import label as cc_label  # noqa: E402
from skimage.morphology import (  # noqa: E402
    binary_closing,
    disk,
    remove_small_holes,
    remove_small_objects,
)
from skimage.segmentation import random_walker  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402

import config  # noqa: E402  (local module: edge_em/config.py)
import features  # noqa: E402
import scribble  # noqa: E402


# ---------------------------------------------------------------------------
# Output helpers (re-implemented from nnunet/predict.py so edge_em is isolated)
# ---------------------------------------------------------------------------
def resolve_spacing(case: str, out_dir: Path, override: float | None) -> float:
    """Pick mm/pixel for the output nii (isotropic single value)."""
    if override is not None:
        return override
    existing = out_dir / f"{case}.nii"
    if existing.exists():
        zooms = nib.load(str(existing)).header.get_zooms()
        print(f"  {case}: reusing spacing {zooms[0]:.4f} mm/px from existing nii")
        return float(zooms[0])
    print(
        f"  WARNING {case}: no spacing given and no existing nii — using 1.0 mm/px. "
        "Length/diameter/volume metrics will be uncalibrated. Pass --spacing."
    )
    return 1.0


def write_mask_nii(mask_hw: np.ndarray, nii_path: Path, spacing: float) -> int:
    """Write a full-size (H, W) binary mask as a (W, H, 1) uint8 nii (main.py-ready)."""
    mask_hw = (mask_hw > 0).astype(np.uint8)
    vol = mask_hw.T[..., np.newaxis]  # (W, H, 1), matches main.py's transpose
    affine = np.diag([spacing, spacing, 1.0, 1.0]).astype(np.float64)
    img = nib.Nifti1Image(vol, affine)
    img.header.set_zooms((spacing, spacing, 1.0))
    nib.save(img, str(nii_path))
    return int(mask_hw.sum())


# ---------------------------------------------------------------------------
# Stage 2: Gaussian-Mixture EM
# ---------------------------------------------------------------------------
def em_posterior(feats: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """Fit a Gaussian mixture per class (via EM) from scribble pixels.

    Returns a (H, W) per-pixel posterior P(vessel) in [0, 1]. ``feats`` is
    (H*W, F); ``seeds`` is (H, W) with 1=vessel, 2=background, 0=unlabeled.
    """
    flat = seeds.ravel()
    fg = feats[flat == scribble.FG_LABEL]
    bg = feats[flat == scribble.BG_LABEL]
    if len(fg) == 0 or len(bg) == 0:
        raise SystemExit(
            "Need both foreground and background scribbles (press 'f' and 'b')."
        )

    def fit(samples: np.ndarray) -> GaussianMixture:
        k = max(1, min(config.GMM_COMPONENTS, len(samples)))
        gmm = GaussianMixture(
            n_components=k,
            max_iter=config.GMM_MAX_ITER,
            random_state=config.GMM_RANDOM_STATE,
        )
        gmm.fit(samples)
        return gmm

    gmm_fg, gmm_bg = fit(fg), fit(bg)

    # Class priors from the relative amount of scribbled evidence.
    log_prior_fg = np.log(len(fg) / (len(fg) + len(bg)))
    log_prior_bg = np.log(len(bg) / (len(fg) + len(bg)))

    num_fg = gmm_fg.score_samples(feats) + log_prior_fg
    num_bg = gmm_bg.score_samples(feats) + log_prior_bg
    denom = logsumexp(np.stack([num_fg, num_bg], axis=0), axis=0)
    post_fg = np.exp(num_fg - denom)
    return post_fg.reshape(seeds.shape)


# ---------------------------------------------------------------------------
# Stage 3: spatial regularization + cleanup
# ---------------------------------------------------------------------------
def regularize(
    panel: np.ndarray, seeds: np.ndarray, posterior: np.ndarray
) -> np.ndarray:
    """Random-walker segmentation, pinned by scribbles and the EM posterior.

    The user scribbles are hard seeds; pixels where the EM posterior is very
    confident become extra soft seeds. The walker then diffuses these labels
    respecting image contrast (beta), giving a clean connected mask.
    """
    labels = seeds.copy().astype(np.int32)
    confident_fg = (posterior >= config.RW_PRIOR_FG_THRESH) & (labels == 0)
    confident_bg = (posterior <= config.RW_PRIOR_BG_THRESH) & (labels == 0)
    labels[confident_fg] = scribble.FG_LABEL
    labels[confident_bg] = scribble.BG_LABEL

    data = panel.astype(np.float64)
    data = (data - data.min()) / (np.ptp(data) + 1e-8)
    result = _solve_random_walker(data, labels)
    mask = result == scribble.FG_LABEL

    if config.CLOSING_RADIUS > 0:
        mask = binary_closing(mask, disk(config.CLOSING_RADIUS))
    if config.MIN_OBJECT_SIZE > 0:
        mask = remove_small_objects(mask, min_size=config.MIN_OBJECT_SIZE)

    # Keep only the vessel the user actually scribbled: drop any component that
    # no foreground scribble touches. This removes the scattered false-positive
    # blobs that EM soft-seeding otherwise leaves in the background.
    if config.KEEP_SCRIBBLE_COMPONENTS:
        mask = _keep_seed_components(mask, seeds == scribble.FG_LABEL)

    # Fill only *small* enclosed holes (pinholes inside the vessel). A blanket
    # fill would flood the interior of any loop the vein makes — common in these
    # tortuous veins — so cap the hole area.
    if config.MAX_HOLE_SIZE > 0 and mask.any():
        mask = remove_small_holes(mask, area_threshold=config.MAX_HOLE_SIZE)
    return mask.astype(np.uint8)


def _solve_random_walker(data: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Run ``random_walker`` robustly, easing tol then beta if it won't converge.

    When the conjugate-gradient solution drifts past ``prob_tol`` (more likely
    with a high ``beta`` on a large frame), skimage signals "The probability range
    is outside [0, 1] ..." — as a ``UserWarning`` in some versions, a ``ValueError``
    in others. We escalate that warning to an error so we can retry with a tighter
    tolerance, then a gentler beta. Each attempt keeps the same scribble + posterior
    seeding, so a successful retry only smooths the diffusion slightly. The final
    attempt does not escalate, so we always return a best-effort result.
    """
    attempts = (
        (config.RW_BETA, config.RW_TOL),
        (config.RW_BETA, config.RW_TOL / 10),
        (config.RW_BETA / 2, config.RW_TOL / 10),
        (config.RW_BETA / 4, config.RW_TOL / 100),
    )
    last = len(attempts) - 1
    for i, (beta, tol) in enumerate(attempts):
        try:
            with warnings.catch_warnings():
                if i < last:  # turn the convergence warning into a retry trigger
                    warnings.filterwarnings(
                        "error", message=".*probability range is outside.*"
                    )
                return random_walker(data, labels, beta=beta, mode="cg_j", tol=tol)
        except (ValueError, Warning) as e:
            print(f"  random_walker: retrying (beta={beta:g}, tol={tol:g}) after: {e}")
    raise RuntimeError("unreachable: final random_walker attempt does not escalate")


def _keep_seed_components(mask: np.ndarray, fg_seeds: np.ndarray) -> np.ndarray:
    """Keep only connected components of ``mask`` that overlap a FG scribble."""
    if not fg_seeds.any():
        return mask
    labelled = cc_label(mask, connectivity=2)
    keep = np.unique(labelled[fg_seeds & (labelled > 0)])
    return np.isin(labelled, keep)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def save_overlay(full_gray: np.ndarray, full_mask: np.ndarray, out_png: Path) -> None:
    """Save a grayscale image with the segmentation mask overlaid in red."""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(full_gray, cmap="gray")
    overlay = np.zeros((*full_mask.shape, 4), dtype=np.float32)
    overlay[full_mask > 0] = (1.0, 0.0, 0.0, 0.45)
    ax.imshow(overlay)
    ax.set_title("Segmentation overlay")
    ax.axis("off")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  overlay -> {out_png}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def collect_inputs(paths: list[str]) -> list[Path]:
    """Expand files/directories into a flat list of supported image paths."""
    out: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            for ext in config.IMAGE_EXTENSIONS:
                out.extend(sorted(path.glob(f"*{ext}")))
        elif path.suffix.lower() in config.IMAGE_EXTENSIONS:
            out.append(path)
        else:
            print(f"  skip {path}: not a supported image or directory")
    return out


def segment_one(
    image_path: Path,
    out_dir: Path,
    spacing_override: float | None,
    brush_radius: int | None,
) -> None:
    case = image_path.stem
    print(f"\n=== {case} ({image_path.name}) ===")

    panel, x0, (H, W) = features.load_panel_gray(image_path)
    vness = features.vesselness(panel)
    feats = features.feature_stack(panel, vness)

    strokes = scribble.load_or_collect_scribbles(case, panel, brush_radius)
    if not strokes:
        print(f"  {case}: no scribbles — skipping")
        return
    seeds = scribble.rasterize(strokes, panel.shape)

    posterior = em_posterior(feats, seeds)
    mask_crop = regularize(panel, seeds, posterior)

    # Paste the cropped mask back into the full image frame.
    full_mask = np.zeros((H, W), dtype=np.uint8)
    full_mask[:, x0 : x0 + mask_crop.shape[1]] = mask_crop

    spacing = resolve_spacing(case, out_dir, spacing_override)
    nii_path = out_dir / f"{case}.nii"
    n_px = write_mask_nii(full_mask, nii_path, spacing)
    print(f"  {case}: {n_px} vessel px -> {nii_path}")

    full_gray = np.zeros((H, W), dtype=np.uint8)
    full_gray[:, x0 : x0 + panel.shape[1]] = panel
    save_overlay(full_gray, full_mask, config.overlay_path(case))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="image file(s) or folder(s).")
    parser.add_argument(
        "-o",
        "--out-dir",
        default=str(config.NII_DIR),
        help="Where to write .nii masks. Default: data/ (so main.py finds them).",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=None,
        help="mm per pixel (isotropic). From your scale-bar calibration.",
    )
    parser.add_argument(
        "--brush",
        type=int,
        default=None,
        help=f"Scribble brush radius in px (default {config.DEFAULT_BRUSH_RADIUS}).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = collect_inputs(args.inputs)
    if not images:
        raise SystemExit("No image inputs found.")
    print(f"Segmenting {len(images)} image(s): {', '.join(i.stem for i in images)}")

    for image_path in images:
        segment_one(image_path, out_dir, args.spacing, args.brush)

    print("\nDone. Run the analysis with:  uv run main.py")


if __name__ == "__main__":
    main()
