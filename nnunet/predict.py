"""Segment veins in a DSA jpg with the trained nnU-Net and write a main.py-ready .nii.

Pipeline per image:
    jpg --grayscale--> {case}_0000.png  (nnU-Net input)
        --nnUNetv2_predict--> {case}.png (0/1 mask, H x W)
        --transpose + spacing--> data/{case}.nii  (W x H x 1, uint8)

The output .nii drops straight into main.py: you still click start/end and the
existing FMM/metrics code runs unchanged.

Orientation: nnU-Net works in (H, W); main.py expects the on-disk nii as
(W, H, 1) and transposes it back with ``.T``. So we write ``mask.T[..., None]``.

Spacing (mm/pixel) — needed for correct length/diameter/volume metrics:
    --spacing 0.0568     set it explicitly (from your Slicer scale-bar calibration)
    (omitted)            reuse spacing from an existing data/{case}.nii if present,
                         otherwise fall back to 1.0 with a warning.

Usage:
    uv run python nnunet/predict.py pre-data/27m.jpg --spacing 0.0568
    uv run python nnunet/predict.py img1.jpg img2.jpg --folds 0 1 2 3 4
    uv run python nnunet/predict.py some_folder/ -o data/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import config  # noqa: E402


def collect_inputs(paths: list[str]) -> list[Path]:
    """Expand files/directories into a flat list of jpg paths."""
    jpgs: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            jpgs.extend(sorted(path.glob("*.jpg")))
        elif path.suffix.lower() in {".jpg", ".jpeg"}:
            jpgs.append(path)
        else:
            print(f"  skip {path}: not a jpg or directory")
    return jpgs


def resolve_spacing(case: str, out_dir: Path, override: float | None) -> float:
    """Pick mm/pixel for the output nii. Isotropic single value."""
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
    """Write a full-size (H, W) binary mask as a (W, H, 1) uint8 nii.

    The mask is in the *original* (uncropped) image frame so the output aligns
    with the full jpg that main.py loads.
    """
    mask_hw = (mask_hw > 0).astype(np.uint8)
    disk = mask_hw.T[..., np.newaxis]  # (W, H, 1), matches main.py
    affine = np.diag([spacing, spacing, 1.0, 1.0]).astype(np.float64)
    img = nib.Nifti1Image(disk, affine)
    img.header.set_zooms((spacing, spacing, 1.0))
    nib.save(img, str(nii_path))
    return int(mask_hw.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="jpg file(s) or folder(s).")
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
        "--folds",
        nargs="+",
        default=["0"],
        help="Folds to ensemble. Must match what you trained. Default: 0.",
    )
    parser.add_argument("--configuration", default="2d")
    parser.add_argument("--trainer", default=None, help="Trainer used at train time.")
    parser.add_argument("--checkpoint", default="checkpoint_final.pth")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "mps", "cuda"],
        help="Compute device. Default cpu (no CUDA on macOS).",
    )
    args = parser.parse_args()

    config.setup_env()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jpgs = collect_inputs(args.inputs)
    if not jpgs:
        raise SystemExit("No jpg inputs found.")
    print(f"Segmenting {len(jpgs)} image(s): {', '.join(j.stem for j in jpgs)}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_in = Path(tmp) / "in"
        tmp_out = Path(tmp) / "out"
        tmp_in.mkdir()
        tmp_out.mkdir()

        # Stage grayscale PNGs, cropped to the right panel exactly as in training.
        # Remember each case's full size and seam offset to paste predictions back.
        geom: dict[str, tuple[tuple[int, int], int]] = {}  # case -> ((H, W), x0)
        for jpg in jpgs:
            gray = np.asarray(Image.open(jpg).convert("L"))  # (H, W)
            x0 = config.right_panel_x0(gray)
            crop = gray[:, x0:]
            geom[jpg.stem] = (gray.shape, x0)
            Image.fromarray(crop, mode="L").save(
                tmp_in / f"{jpg.stem}_0000{config.FILE_ENDING}"
            )

        cmd = config.nnunet_cmd("nnUNetv2_predict") + [
            "-i",
            str(tmp_in),
            "-o",
            str(tmp_out),
            "-d",
            str(config.DATASET_ID),
            "-c",
            args.configuration,
            "-f",
            *args.folds,
            "-chk",
            args.checkpoint,
            "-device",
            args.device,
        ]
        if args.trainer:
            cmd += ["-tr", args.trainer]
        print(f"\n$ {' '.join(cmd)}\n", flush=True)
        subprocess.run(cmd, check=True)

        # Paste each cropped prediction back into the full image frame, then
        # write a main.py-compatible nii aligned with the original jpg.
        print("\nWriting masks:")
        for jpg in jpgs:
            case = jpg.stem
            png = tmp_out / f"{case}{config.FILE_ENDING}"
            if not png.exists():
                print(f"  {case}: no prediction produced — skipping")
                continue
            (h, w), x0 = geom[case]
            pred_crop = (np.asarray(Image.open(png)) > 0).astype(np.uint8)
            full = np.zeros((h, w), dtype=np.uint8)
            full[:, x0 : x0 + pred_crop.shape[1]] = pred_crop
            spacing = resolve_spacing(case, out_dir, args.spacing)
            nii_path = out_dir / f"{case}.nii"
            n_px = write_mask_nii(full, nii_path, spacing)
            print(f"  {case}: {n_px} vessel px -> {nii_path}")

    print("\nDone. Run the analysis with:  uv run main.py")


if __name__ == "__main__":
    main()
