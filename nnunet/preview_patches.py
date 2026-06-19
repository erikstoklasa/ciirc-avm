"""Render a contact sheet of the prepared training patches for a sanity check.

Each tile shows an ``imagesTr`` patch in grayscale with its ``labelsTr`` vein
mask tinted red on top, plus the case name. Useful to confirm that tiling,
the rotation/noise augmentation, and image/label alignment all look right
before committing to a training run.

Usage:
    uv run python nnunet/preview_patches.py                 # 30 random patches
    uv run python nnunet/preview_patches.py -n 48 --seed 1  # more, different draw
    uv run python nnunet/preview_patches.py --filter _r     # only rotated patches
    uv run python nnunet/preview_patches.py --filter _neg   # only negatives
    uv run python nnunet/preview_patches.py -o out.png --no-open
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")  # write a file; no interactive backend needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import config  # noqa: E402


def overlay(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Grayscale image as RGB with vein-mask pixels pushed toward red."""
    rgb = np.repeat(img[..., None].astype(np.float32), 3, axis=2)
    m = mask > 0
    rgb[m] = 0.5 * rgb[m] + 0.5 * np.array([255.0, 0.0, 0.0])
    return rgb.astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--num", type=int, default=30, help="Patches to show.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument(
        "--filter",
        default="",
        help="Only show cases whose name contains this substring "
        "(e.g. _r, _n, _neg, or a case prefix like 10m).",
    )
    parser.add_argument("--cols", type=int, default=6, help="Grid columns.")
    parser.add_argument(
        "-o",
        "--out",
        default=str(config.NNUNET_HOME / "preview_patches.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--no-open", action="store_true", help="Do not open the PNG when done."
    )
    args = parser.parse_args()

    config.setup_env()
    labels_dir = config.dataset_raw_dir() / "labelsTr"
    images_dir = config.dataset_raw_dir() / "imagesTr"
    cases = sorted(p.stem for p in labels_dir.glob(f"*{config.FILE_ENDING}"))
    if args.filter:
        cases = [c for c in cases if args.filter in c]
    if not cases:
        raise SystemExit(
            f"No patches found in {labels_dir} (filter={args.filter!r}). "
            "Run prepare_dataset.py first."
        )

    rng = np.random.default_rng(args.seed)
    n = min(args.num, len(cases))
    pick = sorted(rng.choice(len(cases), size=n, replace=False).tolist())
    chosen = [cases[i] for i in pick]

    # Each patch is shown as a pair of columns: plain image | labeled overlay.
    pairs = max(1, args.cols)  # patch-pairs per row
    rows = (n + pairs - 1) // pairs
    fig, axes = plt.subplots(rows, pairs * 2, figsize=(pairs * 2 * 1.7, rows * 2.0))
    axes = np.atleast_2d(axes)

    for k in range(rows * pairs):
        r, c = k // pairs, k % pairs
        ax_l, ax_r = axes[r, c * 2], axes[r, c * 2 + 1]
        if k >= n:
            ax_l.axis("off")
            ax_r.axis("off")
            continue
        case = chosen[k]
        img = np.asarray(
            Image.open(images_dir / f"{case}_0000{config.FILE_ENDING}").convert("L")
        )
        mask = np.asarray(Image.open(labels_dir / f"{case}{config.FILE_ENDING}"))
        ax_l.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax_l.set_title(case, fontsize=6)
        ax_r.imshow(overlay(img, mask))
        ax_r.set_title(f"{int((mask > 0).sum())} px", fontsize=6)
        ax_l.axis("off")
        ax_r.axis("off")

    fig.suptitle(
        f"{n} of {len(cases)} patches"
        + (f"  (filter={args.filter!r})" if args.filter else "")
        + "   —   red = vein label",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = Path(args.out)
    fig.savefig(out, dpi=130)
    print(f"Wrote {out}  ({n} patches)")

    if not args.no_open and sys.platform == "darwin":
        subprocess.run(["open", str(out)], check=False)


if __name__ == "__main__":
    main()
