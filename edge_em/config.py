"""Shared configuration for the edge_em semi-automatic vein segmentation.

This module is the classical / no-training-required sibling of ``nnunet/``. It
turns a DSA source image into a ``main.py``-ready binary ``.nii`` using a
multi-scale vesselness filter followed by a Gaussian-Mixture EM segmentation,
seeded by weak foreground/background scribbles.

Everything for this approach lives under ``edge_em/`` so it never collides with
``main.py`` or the ``nnunet/`` pipeline. The few small helpers borrowed from
``nnunet/`` (panel cropping, nii writing) are re-implemented here on purpose,
keeping this module self-contained rather than cross-importing.
"""

from __future__ import annotations

from pathlib import Path

# --- Project layout -------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
JPG_DIR = REPO_ROOT / "pre-data"  # source DSA images: {name}.jpg/.jpeg/.png
NII_DIR = REPO_ROOT / "data"  # segmentation masks: {name}.nii (main.py I/O)

# Source images may be any of these (loaded as grayscale).
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Class map. main.py only cares about >0; we keep the same convention as nnunet.
LABELS = {"background": 0, "vessel": 1}

# --- Stage 1: vesselness (Frangi) ----------------------------------------
# Sigma scales (px) for the multi-scale Frangi filter. Should bracket the
# expected vein half-widths in the cropped panel. Veins here are a few px to a
# couple dozen px wide, so 1..8 px covers thin and thick vessels.
FRANGI_SIGMAS = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
FRANGI_BLACK_RIDGES = True  # DSA veins are darker than background

# --- Stage 1: Canny edges (boundary cue for regularization) --------------
CANNY_SIGMA = 2.0

# --- Stage 2: Gaussian-Mixture EM ----------------------------------------
# One GMM per class (vessel / background), each a small mixture, fitted via EM
# on the per-pixel feature vectors of that class's scribble pixels.
GMM_COMPONENTS = 2  # mixture components per class
GMM_MAX_ITER = 200  # EM iterations
GMM_RANDOM_STATE = 0  # reproducible EM init

# --- Spatial regularization (random walker) ------------------------------
# Larger beta = sharper class boundaries (diffusion respects image contrast).
RW_BETA = 130
# Conjugate-gradient convergence tolerance. Lower = tighter convergence, which
# keeps the walker probabilities inside [0, 1]; skimage raises "probability range
# is outside [0, 1]" when they drift past prob_tol, which a high beta on a large
# frame can trigger. Tighter than skimage's 1e-3 default for robustness.
RW_TOL = 1e-4
# How strongly the EM posterior pulls the random walker. The posterior is
# injected as extra soft seeds; this is the probability threshold above/below
# which a pixel is offered as a soft vessel/background seed.
RW_PRIOR_FG_THRESH = 0.90
RW_PRIOR_BG_THRESH = 0.10

# --- Mask cleanup ---------------------------------------------------------
CLOSING_RADIUS = 2  # morphological closing radius to bridge small gaps
MIN_OBJECT_SIZE = 64  # remove connected components smaller than this (px)
# Keep only mask components that a foreground scribble touches. Veins analysed
# by main.py are a single connected vessel, so this drops the scattered
# false-positive blobs EM soft-seeding leaves in the background.
KEEP_SCRIBBLE_COMPONENTS = True
MAX_HOLE_SIZE = 64  # fill only enclosed holes smaller than this (px);
# keeps real loop interiors (tortuous veins) open

# --- Scribble paint tool --------------------------------------------------
DEFAULT_BRUSH_RADIUS = 4  # px radius of a scribble stroke when rasterized

# Panel seam detection (mirrors nnunet/config.py:right_panel_x0). Each DSA frame
# is a side-by-side pair; the right panel is the clean copy that was segmented.
PANEL_SEAM_SEARCH = 70  # +/- px around center to search for the darkest column


def right_panel_x0(gray_hw) -> int:
    """Return the left x-bound (column) of the right panel for an (H, W) image.

    Detects the inter-panel seam as the darkest column within
    +/-PANEL_SEAM_SEARCH px of center. Slice the image as ``arr[:, x0:]``.
    Re-implemented from ``nnunet/config.py`` so edge_em stays self-contained.
    """
    import numpy as np

    h, w = gray_hw.shape[:2]
    center = w // 2
    lo = max(0, center - PANEL_SEAM_SEARCH)
    hi = min(w, center + PANEL_SEAM_SEARCH)
    col_mean = np.asarray(gray_hw, dtype=np.float32).mean(axis=0)
    return lo + int(np.argmin(col_mean[lo:hi]))


def scribbles_path(case: str) -> Path:
    """Where the cached FG/BG scribbles for a case are stored."""
    return NII_DIR / f"{case}_scribbles.json"


def overlay_path(case: str) -> Path:
    """Where the inspection overlay PNG for a case is written."""
    return NII_DIR / f"{case}_overlay.png"


def nii_path(case: str) -> Path:
    """The main.py-ready mask output path for a case."""
    return NII_DIR / f"{case}.nii"
