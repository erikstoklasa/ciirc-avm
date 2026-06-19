"""Benchmark nnU-Net vs edge_em (EM) vein segmentation against a ground-truth .nii.

Both methods segment the *same* case and are scored (Dice + IoU) against the
existing ``data/{case}.nii`` label:

* **nnunet** -- the trained nnU-Net 2D model, run via ``nnunet/predict.py`` (the
  same right-panel crop + ensemble it uses in production). No user input.
* **em** -- the classical Frangi-vesselness + Gaussian-Mixture EM + random-walker
  pipeline from ``edge_em/``. Still prompts for weak FG/BG scribbles (cached in
  the benchmark dir so they are reused on re-run).

Nothing is written under ``data/``: every output (scribbles cache, masks,
overlays, reference, scores) goes to ``benchmark/{case}/``. The existing
``data/{case}.nii`` is only *read* (for ground truth and spacing).

Usage:
    uv run python benchmark.py 10m
    uv run python benchmark.py 10m --brush 4 --spacing 0.0568
    uv run python benchmark.py 10m --device mps --folds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "edge_em"))

import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402  (edge_em/config.py)
import features  # noqa: E402
import scribble  # noqa: E402
import segment  # noqa: E402


def find_image(case: str) -> Path:
    """Locate pre-data/{case}.{jpg,jpeg,png}."""
    for ext in config.IMAGE_EXTENSIONS:
        p = config.JPG_DIR / f"{case}{ext}"
        if p.exists():
            return p
    raise SystemExit(f"No source image for case '{case}' in {config.JPG_DIR}")


def scores(mask: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Return (Dice, IoU) of a binary mask against ground truth."""
    m, g = mask > 0, gt > 0
    inter = int(np.logical_and(m, g).sum())
    union = int(np.logical_or(m, g).sum())
    denom = int(m.sum()) + int(g.sum())
    dice = 2 * inter / denom if denom else 1.0
    iou = inter / union if union else 1.0
    return dice, iou


def make_reference(panel: np.ndarray, gt_panel: np.ndarray) -> np.ndarray:
    """RGB float image: grayscale panel with the true vein tinted green."""
    rgb = np.repeat((panel.astype(np.float32) / 255.0)[..., None], 3, axis=2)
    g = gt_panel > 0
    rgb[g] = 0.35 * rgb[g] + np.array([0.0, 0.6, 0.0], dtype=np.float32)
    return np.clip(rgb, 0, 1)


def save_overlay(full_gray, full_mask, gt_full, title, out_png) -> None:
    """Grayscale image with predicted mask (red) and GT outline (cyan)."""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(full_gray, cmap="gray")
    pred = np.zeros((*full_mask.shape, 4), np.float32)
    pred[full_mask > 0] = (1.0, 0.0, 0.0, 0.45)
    ax.imshow(pred)
    ax.contour(gt_full > 0, levels=[0.5], colors="cyan", linewidths=0.8)
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_em(case: str, panel, reference, out_dir, brush) -> np.ndarray:
    """Run the edge_em pipeline, returning a cropped (panel-sized) binary mask."""
    vness = features.vesselness(panel)
    feats = features.feature_stack(panel, vness)

    strokes = scribble.load_or_collect_scribbles(
        case,
        panel,
        brush_radius=brush,
        reference=reference,
        cache_path=out_dir / f"{case}_scribbles.json",
    )
    if not strokes:
        raise SystemExit("No scribbles — cannot run the EM segmentation.")
    seeds = scribble.rasterize(strokes, panel.shape)

    posterior = segment.em_posterior(feats, seeds)
    return segment.regularize(panel, seeds, posterior)


def run_nnunet(case: str, img_path: Path, x0: int, args) -> np.ndarray:
    """Run nnunet/predict.py into a temp dir, returning a cropped binary mask.

    predict.py writes a full-frame (W, H, 1) nii; we transpose back to (H, W) and
    crop to the right panel so it lines up with the EM mask and ground truth.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            sys.executable,
            str(REPO_ROOT / "nnunet" / "predict.py"),
            str(img_path),
            "-o",
            tmp,
            "--configuration",
            args.configuration,
            "--folds",
            *args.folds,
            "--checkpoint",
            args.checkpoint,
            "--device",
            args.device,
        ]
        if args.trainer:
            cmd += ["--trainer", args.trainer]
        print(f"\n$ {' '.join(cmd)}\n", flush=True)
        subprocess.run(cmd, check=True)

        nii = Path(tmp) / f"{case}.nii"
        if not nii.exists():
            raise SystemExit(f"nnU-Net produced no prediction for '{case}'.")
        full = (np.squeeze(np.asarray(nib.load(str(nii)).dataobj)) > 0).T.astype(
            np.uint8
        )
    return full[:, x0:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "case", help="case name, e.g. 10m (needs pre-data img + data/.nii)"
    )
    parser.add_argument(
        "--brush", type=int, default=None, help="EM scribble brush radius (px)"
    )
    parser.add_argument(
        "--spacing", type=float, default=None, help="mm/px for output nii"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output dir (default: benchmark/{case}/). Never data/.",
    )
    # nnU-Net inference knobs (forwarded to nnunet/predict.py).
    parser.add_argument("--folds", nargs="+", default=["0"], help="folds to ensemble")
    parser.add_argument("--configuration", default="2d")
    parser.add_argument(
        "--trainer",
        default="nnUNetTrainer_5epochs",
        help="trainer used at train time (the only trained model in nnUNet_results)",
    )
    parser.add_argument("--checkpoint", default="checkpoint_final.pth")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "mps", "cuda"],
        help="compute device for nnU-Net (default cpu)",
    )
    args = parser.parse_args()
    case = args.case

    gt_nii = config.NII_DIR / f"{case}.nii"
    if not gt_nii.exists():
        raise SystemExit(f"No ground-truth label {gt_nii} — pick a case that has one.")

    out_dir = Path(args.out) if args.out else REPO_ROOT / "benchmark" / case
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_dir.resolve() == config.NII_DIR.resolve():
        raise SystemExit("Refusing to write into data/ — choose a different --out.")

    img_path = find_image(case)
    print(f"Case {case}: image {img_path.name}, GT {gt_nii.name}, out {out_dir}")

    # Shared geometry: load the panel once so EM, nnU-Net, and GT all align.
    panel, x0, (H, W) = features.load_panel_gray(img_path)
    # Load the ground-truth mask once in (H, W) (like main.py); the right-panel
    # crop is just a view that aligns with the EM / nnU-Net masks.
    gt_full = (np.squeeze(np.asarray(nib.load(str(gt_nii)).dataobj)) > 0).T.astype(
        np.uint8
    )
    gt_panel = gt_full[:, x0:]

    # Reference image (green = true vein) shown beside the scribble canvas so the
    # user knows where to paint. Saved to the benchmark dir, not data/.
    reference = make_reference(panel, gt_panel)
    ref_png = out_dir / f"{case}_reference.png"
    plt.imsave(str(ref_png), reference)
    print(f"  reference (green = true vein) -> {ref_png}")

    # EM first (interactive scribbles), then nnU-Net (may be slow on CPU).
    mask_em = run_em(case, panel, reference, out_dir, args.brush)
    mask_nnunet = run_nnunet(case, img_path, x0, args)

    spacing = segment.resolve_spacing(case, config.NII_DIR, args.spacing)
    full_gray = np.zeros((H, W), np.uint8)
    full_gray[:, x0 : x0 + panel.shape[1]] = panel

    print("\n  method    Dice     IoU    vessel_px")
    results = {}
    for name, mask_crop in (("nnunet", mask_nnunet), ("em", mask_em)):
        dice, iou = scores(mask_crop, gt_panel)
        results[name] = dice
        full = np.zeros((H, W), np.uint8)
        full[:, x0 : x0 + mask_crop.shape[1]] = mask_crop
        segment.write_mask_nii(full, out_dir / f"{case}_{name}.nii", spacing)
        save_overlay(
            full_gray,
            full,
            gt_full,
            f"{name}: Dice={dice:.3f} IoU={iou:.3f}",
            out_dir / f"{case}_{name}_overlay.png",
        )
        print(f"  {name:<8}  {dice:.3f}   {iou:.3f}   {int(mask_crop.sum())}")

    winner = max(results, key=results.get)
    delta = abs(results["nnunet"] - results["em"])
    print(f"\n  WINNER: {winner} by Dice +{delta:.3f}")
    print(f"  Overlays + masks + reference written to {out_dir} (data/ untouched).")


if __name__ == "__main__":
    main()
