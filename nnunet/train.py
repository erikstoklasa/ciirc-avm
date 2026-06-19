"""Plan, preprocess and train the 2D nnU-Net for vein segmentation.

This is a thin wrapper around nnU-Net v2's CLI that injects the correct
environment variables (see config.py) so you don't have to export them by hand.

Steps it runs:
    1. nnUNetv2_plan_and_preprocess  (fingerprint + plans + preprocessed data)
    2. nnUNetv2_train                (one call per requested fold)

Usage:
    uv run python nnunet/train.py                 # train fold 0 (quick start)
    uv run python nnunet/train.py --folds 0 1 2 3 4   # full 5-fold CV (best)
    uv run python nnunet/train.py --folds all     # single model on all data
    uv run python nnunet/train.py --skip-preprocess --folds 1

Notes:
    * nnU-Net needs a CUDA GPU for practical training times. On CPU it will run
      but be extremely slow — fine only for a smoke test.
    * With ~26 cases, train all 5 folds for the most robust result, or fold 0
      alone to iterate quickly. predict.py must use whatever folds you trained.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--folds",
        nargs="+",
        default=["0"],
        help="Folds to train: any of 0-4, or 'all'. Default: 0.",
    )
    parser.add_argument(
        "--configuration",
        default="2d",
        help="nnU-Net configuration (these are 2D images). Default: 2d.",
    )
    parser.add_argument(
        "--trainer",
        default=None,
        help="Optional nnU-Net trainer, e.g. nnUNetTrainer_50epochs for a fast run.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "mps", "cuda"],
        help="Compute device. Default cpu (no CUDA on macOS; mps may hit "
        "unsupported-op errors in nnU-Net).",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip plan_and_preprocess (use if already done).",
    )
    args = parser.parse_args()

    config.setup_env()

    if not args.skip_preprocess:
        run(
            config.nnunet_cmd("nnUNetv2_plan_and_preprocess")
            + [
                "-d",
                str(config.DATASET_ID),
                "--verify_dataset_integrity",
            ]
        )

    for fold in args.folds:
        cmd = config.nnunet_cmd("nnUNetv2_train") + [
            str(config.DATASET_ID),
            args.configuration,
            fold,
            "-device",
            args.device,
        ]
        if args.trainer:
            cmd += ["-tr", args.trainer]
        run(cmd)

    print("\nTraining complete. Run inference with:")
    print("  uv run python nnunet/predict.py path/to/image.jpg")


if __name__ == "__main__":
    main()
