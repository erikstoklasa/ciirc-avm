"""Shared configuration for the nnU-Net vein-segmentation pipeline.

This module sets the three environment variables nnU-Net v2 requires
(``nnUNet_raw``, ``nnUNet_preprocessed``, ``nnUNet_results``) so that every
script in this folder agrees on where data and trained models live. Import it
*before* importing anything from ``nnunetv2`` or invoking the nnU-Net CLI.

All nnU-Net artifacts are kept inside ``nnunet/`` so they never collide with the
analysis pipeline in ``main.py``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# --- Project layout -------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
JPG_DIR = REPO_ROOT / "pre-data"  # source DSA images: {name}.jpg
NII_DIR = REPO_ROOT / "data"  # segmentation masks:  {name}.nii (also main.py I/O)

# --- nnU-Net dataset identity --------------------------------------------
DATASET_ID = 1
DATASET_NAME = f"Dataset{DATASET_ID:03d}_Veins"
FILE_ENDING = ".png"  # nnU-Net's 2D natural-image format

# Class map. nnU-Net needs integer labels; main.py only cares about >0.
LABELS = {"background": 0, "vessel": 1}

# --- Patch extraction (training) -----------------------------------------
# Instead of training on whole panels (26 large cases), we tile each panel into
# small square windows and keep only those that sit on/near the annotated veins.
# This turns 26 frames into hundreds of vessel-centric samples, so the network
# sees far more — and more balanced — training data. Inference is unaffected:
# nnU-Net slides this same patch size over the full image at predict time.
PATCH_SIZE = 128  # square patch side, pixels
PATCH_STRIDE = 64  # step between candidate windows (50% overlap)
PATCH_MIN_FG = 25  # keep a window only if it has >= this many vessel px
PATCH_NEG_PER_CASE = 2  # extra vessel-free windows/case to curb false positives

# --- Offline augmentation (training) -------------------------------------
# Emit extra rotated / noisy copies of each panel before tiling, multiplying the
# patch count and the appearance variety. This stacks on top of nnU-Net's own
# online augmentation. Set either tuple to () to disable that augmentation.
AUG_ROTATIONS = (-7, 7)  # degrees (counter-clockwise); slight tilts only
AUG_GAUSS_NOISE_STD = (10.0,)  # uint8 std of additive Gaussian noise per copy

# --- nnU-Net working directories (kept under nnunet/) ---------------------
NNUNET_HOME = REPO_ROOT / "nnunet"
NNUNET_RAW = NNUNET_HOME / "nnUNet_raw"
NNUNET_PREPROCESSED = NNUNET_HOME / "nnUNet_preprocessed"
NNUNET_RESULTS = NNUNET_HOME / "nnUNet_results"


def setup_env() -> None:
    """Point nnU-Net at our directories and make sure they exist."""
    for path in (NNUNET_RAW, NNUNET_PREPROCESSED, NNUNET_RESULTS):
        path.mkdir(parents=True, exist_ok=True)
    os.environ["nnUNet_raw"] = str(NNUNET_RAW)
    os.environ["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)
    os.environ["nnUNet_results"] = str(NNUNET_RESULTS)


def dataset_raw_dir() -> Path:
    return NNUNET_RAW / DATASET_NAME


# nnU-Net console-script -> "module:function" entry points. Invoking these via
# `python -c` is immune to PATH / console-script install-location problems that
# bite on Colab and under `uv run` (where the bare `nnUNetv2_*` command may not
# resolve). Verify with: importlib.metadata.entry_points(group='console_scripts').
NNUNET_ENTRYPOINTS = {
    "nnUNetv2_plan_and_preprocess": "nnunetv2.experiment_planning.plan_and_preprocess_entrypoints:plan_and_preprocess_entry",
    "nnUNetv2_train": "nnunetv2.run.run_training:run_training_entry",
    "nnUNetv2_predict": "nnunetv2.inference.predict_from_raw_data:predict_entry_point",
}


# Each DSA frame is a side-by-side pair: the left panel holds the radiologist's
# hand annotation, the right panel is the clean copy that was segmented. We crop
# to the right panel so the network sees the labeled vein once (not an identical
# unlabeled twin on the left) and patches aren't 2x too wide. The split is the
# darkest vertical column near image center (the gutter between panels).
PANEL_SEAM_SEARCH = 70  # +/- pixels around center to search for the seam


def right_panel_x0(gray_hw) -> int:
    """Return the left x-bound (column) of the right panel for an (H, W) image.

    Detects the inter-panel seam as the darkest column within
    +/-PANEL_SEAM_SEARCH px of center. Slice the image/mask as ``arr[:, x0:]``.
    Validated to leave all 26 training masks fully inside the right panel.
    """
    import numpy as np  # local import keeps config dependency-light

    h, w = gray_hw.shape[:2]
    center = w // 2
    lo = max(0, center - PANEL_SEAM_SEARCH)
    hi = min(w, center + PANEL_SEAM_SEARCH)
    col_mean = np.asarray(gray_hw, dtype=np.float32).mean(axis=0)
    return lo + int(np.argmin(col_mean[lo:hi]))


def nnunet_cmd(name: str) -> list[str]:
    """Return a command prefix that runs an nnU-Net entry point.

    Uses ``sys.executable -c "from <mod> import <fn>; sys.exit(<fn>())"`` so the
    call always uses the active interpreter and never depends on PATH or the
    presence of the ``nnUNetv2_*`` console scripts. nnU-Net's entry functions
    parse ``sys.argv[1:]`` via argparse, so callers append CLI args as usual.
    """
    module, func = NNUNET_ENTRYPOINTS[name].split(":")
    code = f"import sys; from {module} import {func}; sys.exit({func}())"
    return [sys.executable, "-c", code]
