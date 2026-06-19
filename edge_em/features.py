"""Stage 1: image features for the edge_em segmentation.

From a DSA source image we derive two cues that drive the EM segmentation:

* **Vesselness** -- a multi-scale Frangi filter response. Frangi analyses the
  Hessian eigenvalues at several scales and lights up dark tubular structures,
  which is exactly what spinal DSA veins are. This is the "SOTA edge detection"
  stage: purpose-built for vessels and CPU-cheap, unlike generic learned edge
  detectors.
* **Canny edges** -- a classic boundary map used later as a contrast cue for the
  random-walker spatial regularization.

The per-pixel feature stack handed to the EM step is ``[intensity, vesselness]``,
both scaled to [0, 1].
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from skimage.feature import canny  # noqa: E402
from skimage.filters import frangi  # noqa: E402

import config  # noqa: E402  (local module: edge_em/config.py)


def load_panel_gray(image_path) -> tuple[np.ndarray, int, tuple[int, int]]:
    """Load a DSA image as grayscale (H, W) and crop to the right panel.

    Accepts ``.jpg``/``.jpeg``/``.png`` (anything PIL can open as mode "L").
    Returns ``(panel, x0, (H, W))`` where ``panel`` is the cropped right panel,
    ``x0`` is the seam column to paste predictions back at, and ``(H, W)`` is the
    full (uncropped) frame size.
    """
    with Image.open(image_path) as im:
        gray = np.asarray(im.convert("L"), dtype=np.uint8)  # (H, W)
    full_shape = gray.shape
    x0 = config.right_panel_x0(gray)
    return gray[:, x0:], x0, full_shape


def _to_float01(gray: np.ndarray) -> np.ndarray:
    """Scale a uint8/float image to [0, 1] floats."""
    g = gray.astype(np.float32)
    lo, hi = float(g.min()), float(g.max())
    if hi <= lo:
        return np.zeros_like(g)
    return (g - lo) / (hi - lo)


def vesselness(gray: np.ndarray) -> np.ndarray:
    """Multi-scale Frangi vesselness, returned in [0, 1] (H, W) float."""
    response = frangi(
        _to_float01(gray),
        sigmas=config.FRANGI_SIGMAS,
        black_ridges=config.FRANGI_BLACK_RIDGES,
    )
    # Frangi output is unnormalized; rescale to [0, 1] for use as a feature.
    mx = float(response.max())
    if mx > 0:
        response = response / mx
    return response.astype(np.float32)


def edges(gray: np.ndarray) -> np.ndarray:
    """Canny edge map (boolean (H, W)) used as a boundary cue."""
    return canny(_to_float01(gray), sigma=config.CANNY_SIGMA)


def feature_stack(gray: np.ndarray, vness: np.ndarray) -> np.ndarray:
    """Build the per-pixel feature matrix for EM.

    Returns an (H*W, 2) array of ``[intensity, vesselness]`` in [0, 1]. Intensity
    is inverted so that "vessel-like" (dark in DSA) reads high, matching the
    vesselness channel's polarity.
    """
    intensity = 1.0 - _to_float01(gray)  # invert: dark veins -> high value
    feats = np.stack([intensity.ravel(), vness.ravel()], axis=1)
    return feats.astype(np.float32)
