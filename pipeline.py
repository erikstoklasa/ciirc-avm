"""End-to-end vein pipeline: image -> segment -> analyze -> report, in napari.

One interactive run per image takes a raw DSA frame all the way to a report,
using napari throughout so there are at most two windows per image:

    image (.jpg/.jpeg/.png)
      ├─ 0. upscale            frame bicubically upsampled --upscale x (default 2)
      │                        so EM runs on a finer grid; spacing divided to match
      ├─ 1. spacing            override > saved cache; if unresolved, measured
      │                        by hand inside the window below
      ├─ 2. segment  [napari]  ONE window: paint vein/background seeds, press 'e'
      │                        to run the edge_em EM (Frangi -> GMM-EM -> random
      │                        walker), refine the resulting 'vessel' mask, and
      │                        (if needed) measure the scale bar — then close
      ├─ 3. guarded save       data/{case}.nii  (refuses to overwrite silently)
      └─ 4. analysis [napari]  main.analyze_vein: click the vein path on the
                               skeleton -> FMM centerline -> data/{case}_qc.png

The segmentation window works in full-frame coordinates: the base image is the
whole DSA frame (for side-by-side datasets the left panel is the clinician's
tracing; for ``--dataset tju`` the separate ``{case} tracked.png`` is shown as a
toggleable reference and the whole raw frame is segmented). EM internally crops
to the clean panel and pastes the mask back. The vessel layer has native
brush/erase/fill/undo. Resolved spacing is cached to ``{case}_spacing.json`` and
painted seeds to ``{case}_seeds.png``, so detection and painting happen once per
case. Step 3 bakes the spacing into the NIfTI header; step 4 reuses ``main.py``
(now also napari) to pick the path. Metrics for every image are aggregated into
``data/metrics.xlsx``.

Usage:
    uv run python pipeline.py pre-data/10m.jpg
    uv run python pipeline.py pre-data/10m.jpg --spacing 0.0568   # skip measuring
    uv run python pipeline.py pre-data/10m.jpg --manual-spacing   # remeasure by hand
    uv run python pipeline.py pre-data/TJU/TJU1.png              # tju auto-detected
    uv run python pipeline.py pre-data/                          # whole folder
    uv run python pipeline.py --next                             # next image w/o a report
    uv run python pipeline.py pre-data/10m.jpg --force           # allow overwrite
    uv run python pipeline.py pre-data/10m.jpg --no-analyze      # segment only
    uv run python pipeline.py pre-data/10m.jpg --upscale 1        # native pixel grid
    uv run python pipeline.py pre-data/10m.jpg --upscale 3        # 3x finer voxel grid

``uv run main.py`` still works standalone to (re)analyze every mask in ``data/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# The EM segmentation lives under edge_em/ and imports its siblings (config,
# features, scribble) as top-level modules, so put that dir on the path first.
_EDGE_EM = Path(__file__).resolve().parent / "edge_em"
sys.path.insert(0, str(_EDGE_EM))

import config  # noqa: E402  (edge_em/config.py)
import features  # noqa: E402  (edge_em/features.py)
import scribble  # noqa: E402  (edge_em/scribble.py)
import segment  # noqa: E402  (edge_em/segment.py)

# Top-level scale-bar reader (repo root is on sys.path when run from here).
from spacing import spacing_from_measurement  # noqa: E402

# Dataset loaders (panel split vs separate label files, per-dataset scale style).
import datasets  # noqa: E402

# Repo-root analysis half (main.py). Its `if __name__ == "__main__"` guard keeps
# the batch glob dormant on import, and it shares no module names with edge_em/.
import main as analysis  # noqa: E402


def infer_dataset(image_path: Path) -> str:
    """Guess the dataset from the image's location/name (overridable by --dataset).

    Mirrors the on-disk conventions in ``datasets.py``: TJU lives in ``.../TJU/``
    as ``TJU*.png`` (raw + separate ``* tracked.png``), JAP in ``.../JAP/`` as
    ``JAP*.jpg``; everything else is the side-by-side ``current`` dataset.
    """
    parts = {p.lower() for p in image_path.parts}
    stem = image_path.stem.lower()
    if "tju" in parts or stem.startswith("tju"):
        return "tju"
    if "jap" in parts or stem.startswith("jap"):
        return "jap"
    return "current"


# ---------------------------------------------------------------------------
# Step 1: spacing
# ---------------------------------------------------------------------------
def spacing_cache_path(out_dir: Path, case: str) -> Path:
    """Where a case's resolved spacing is cached (next to its mask)."""
    return out_dir / f"{case}_spacing.json"


def load_cached_spacing(out_dir: Path, case: str) -> float | None:
    """Return a previously saved mm/px for the case, or None if absent/unreadable."""
    path = spacing_cache_path(out_dir, case)
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return float(json.load(f)["mm_per_px"])
    except (json.JSONDecodeError, KeyError, ValueError, OSError) as e:
        print(f"  {case}: ignoring unreadable spacing cache ({e})")
        return None


def load_cached_scale_points(out_dir: Path, case: str) -> list[list[float]] | None:
    """Return the cached scale-bar endpoints (native (row, col) pixels), or None.

    These are the two points whose distance set the saved spacing; the segment
    window draws them back onto the image so the spacing can be eyeballed.
    """
    path = spacing_cache_path(out_dir, case)
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            pts = json.load(f).get("endpoints")
        if not pts:
            return None
        return [[float(p[0]), float(p[1])] for p in pts]
    except (json.JSONDecodeError, ValueError, TypeError, IndexError, OSError):
        return None


def load_cached_spacing_label(out_dir: Path, case: str) -> str | None:
    """Human-readable description of a cached spacing, for display in the segment window.

    Built from the same evidence ``save_spacing`` writes (value/unit/length/source),
    so a reused cache shows e.g. "0.1012 mm/px (10 mm / 98.8 px, manual)" instead of
    a bare number. Falls back to just the number when those fields are absent
    (an ``--spacing`` override, or a cache saved before this detail was added).
    """
    path = spacing_cache_path(out_dir, case)
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        mm_per_px = float(data["mm_per_px"])
    except (json.JSONDecodeError, KeyError, ValueError, OSError):
        return None
    source = data.get("source", "?")
    value_mm, unit, length_px = data.get("value_mm"), data.get("unit"), data.get("length_px")
    if value_mm is None or unit is None or length_px is None:
        return f"{mm_per_px:.4f} mm/px  ({source})"
    return f"{mm_per_px:.4f} mm/px  ({value_mm:g} {unit} / {length_px:.1f} px, {source})"


def save_spacing(
    out_dir: Path, case: str, mm_per_px: float, source: str, est=None
) -> None:
    """Cache a case's spacing to ``{case}_spacing.json`` for reuse on re-runs.

    ``source`` records how it was obtained ("override"/"manual"); when an ``est``
    Spacing is given its evidence (measured px, physical length, note) is stored
    too, so the file is human-inspectable.
    """
    data: dict = {"mm_per_px": float(mm_per_px), "source": source}
    if est is not None:
        data.update(
            value_mm=est.value_mm,
            length_px=est.length_px,
            unit=est.unit,
            method=est.method,
            note=est.note,
        )
        if getattr(est, "endpoints", None) is not None:
            data["endpoints"] = [[float(p[0]), float(p[1])] for p in est.endpoints]
    path = spacing_cache_path(out_dir, case)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  {case}: saved spacing -> {path.name}")


def resolve_spacing(
    case: str,
    out_dir: Path,
    override: float | None,
    manual_force: bool,
) -> tuple[float | None, bool, list[list[float]] | None, str | None]:
    """Resolve mm/pixel non-interactively: override > saved cache > manual.

    Returns ``(mm_per_px, needs_manual, scale_points, spacing_label)``. When
    nothing is resolved, ``mm_per_px`` is None and ``needs_manual`` is True so the
    caller can offer the in-window manual measurement (``segment_in_napari``).
    ``scale_points`` are the measured scale-bar endpoints (native (row, col)
    pixels) when known, for the segment window to draw as a verification overlay;
    None otherwise. ``spacing_label`` is a human-readable description (e.g.
    "0.1012 mm/px (10 mm / 98.8 px, manual)") for the segment window to display,
    so the loaded spacing is visible even when ``scale_points`` is unavailable. An
    explicit ``--spacing`` always wins and is cached; a previously saved
    ``{case}_spacing.json`` is reused so measuring happens once per case;
    ``--manual-spacing`` forces a fresh in-window measurement.
    """
    if override is not None:
        print(f"  {case}: spacing override {override:.4f} mm/px")
        save_spacing(out_dir, case, override, "override")
        return override, False, None, f"{override:.4f} mm/px  (override)"

    if manual_force:
        print(f"  {case}: --manual-spacing; will measure in the napari window.")
        return None, True, None, None

    cached = load_cached_spacing(out_dir, case)
    if cached is not None:
        print(
            f"  {case}: using saved spacing {cached:.4f} mm/px "
            f"({spacing_cache_path(out_dir, case).name})"
        )
        return (
            cached,
            False,
            load_cached_scale_points(out_dir, case),
            load_cached_spacing_label(out_dir, case),
        )

    print(f"  {case}: no saved spacing — will offer manual measurement.")
    return None, True, None, None


# ---------------------------------------------------------------------------
# Step 2: EM segmentation (reuses edge_em building blocks)
# ---------------------------------------------------------------------------
def _upscale_array(arr: np.ndarray, factor: int) -> np.ndarray:
    """Bicubically upsample an image array (grayscale or RGB) by an integer factor."""
    if factor <= 1:
        return arr
    h, w = arr.shape[:2]
    im = Image.fromarray(arr).resize((w * factor, h * factor), Image.BICUBIC)
    return np.asarray(im, dtype=arr.dtype)


def _seed_region(seeds_full: np.ndarray, x0: int, W: int) -> tuple[int, int]:
    """Pick the column range to segment from where the seeds were actually painted.

    The side-by-side datasets put the clinician's tracing on one side of the seam
    (``x0``) and the clean image on the other — but which side is *not* consistent
    (JAP10's clean panel is on the right, JAP14's is on the left). Rather than
    hardcode "right panel", segment whichever seam-side panel holds BOTH a vein and
    a background seed; if neither side has both (seeds split across the seam, or a
    single-panel frame like TJU where ``x0 == 0``), fall back to the whole frame so
    no painted seed is cropped away. Returns ``(c0, c1)`` column bounds.
    """
    if x0 <= 0:
        return 0, W
    for c0, c1 in ((x0, W), (0, x0)):
        side = seeds_full[:, c0:c1]
        if (side == scribble.FG_LABEL).any() and (side == scribble.BG_LABEL).any():
            return c0, c1
    return 0, W


def _resample_grid(arr: np.ndarray, shape: tuple[int, int], *, reduce_max: bool):
    """Resample a 2-D label/mask array onto ``shape`` for nearest-neighbour reuse.

    Cached seeds and masks are stored at whatever ``--upscale`` was used when they
    were made; a later run at a different factor needs them on the current grid.
    Resamples when the two shapes are clean integer multiples — nearest-replicate
    to go finer, block-reduce to go coarser (max for a mask, so any vessel sub-voxel
    survives; top-left for labels, to keep label identities). Returns ``arr``
    unchanged when shapes already match, or None when they are not a common
    integer multiple (so the caller can ignore an incompatible cache).
    """
    if arr.shape == shape:
        return arr
    (H, W), (h, w) = shape, arr.shape
    if H % h == 0 and W % w == 0 and H // h == W // w and H // h > 1:  # finer
        k = H // h
        return np.repeat(np.repeat(arr, k, axis=0), k, axis=1)
    if h % H == 0 and w % W == 0 and h // H == w // W and h // H > 1:  # coarser
        k = h // H
        blocks = arr.reshape(H, k, W, k)
        return blocks.max(axis=(1, 3)) if reduce_max else blocks[:, 0, :, 0]
    return None


def _load_for_segmentation(
    image_path: Path, dataset: str, upscale: int = 1
) -> tuple[np.ndarray, int, tuple[int, int], np.ndarray, np.ndarray | None]:
    """Resolve the panel to segment, its paste offset, and the labeled reference.

    Datasets differ in layout: ``current``/``jap`` are side-by-side frames whose
    right panel is the clean copy and left panel is the clinician's tracing;
    ``tju`` stores the raw and the traced versions in two *separate* files. Returns
    ``(panel_gray, x0, (H, W), full_rgb, reference_rgb)`` where ``panel_gray`` is
    the grayscale region segmented, ``x0`` is the column it is pasted back at,
    ``full_rgb`` is the frame shown for refinement, and ``reference_rgb`` is the
    labeled image shown beside the scribble canvas (None if unavailable).

    When ``upscale > 1`` the whole frame is bicubically upsampled here, as the
    first pipeline step, so vesselness/EM/random-walker all run on the finer grid;
    ``x0`` and ``(H, W)`` are scaled to match. The right-panel seam is detected on
    the *native* image (its search window is fixed-width) and then scaled, rather
    than re-detected on the larger frame.
    """
    if dataset == "tju":
        # Single-panel raw image: segment the whole frame (no panel crop), and
        # take the reference from the separate "{case} tracked.png" file.
        loader = datasets.TJUDataset(image_path)
        full_rgb = loader.get_raw()
        panel = np.asarray(Image.fromarray(full_rgb).convert("L"), dtype=np.uint8)
        try:
            reference = loader.get_labeled()
        except (FileNotFoundError, OSError):
            print(f"  {image_path.stem}: no tracked reference file found.")
            reference = None
        x0 = 0
    else:
        # Side-by-side: segment the clean right panel, reference is the left tracing.
        with Image.open(image_path) as im:
            full_rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
        gray = np.asarray(Image.fromarray(full_rgb).convert("L"), dtype=np.uint8)
        x0 = config.right_panel_x0(gray)  # expects 2-D grayscale (H, W)
        panel = gray[:, x0:]
        reference = full_rgb[:, :x0]

    if upscale > 1:
        panel = _upscale_array(panel, upscale)
        full_rgb = _upscale_array(full_rgb, upscale)
        if reference is not None:
            reference = _upscale_array(reference, upscale)
        x0 *= upscale

    return panel, x0, full_rgb.shape[:2], full_rgb, reference


def seeds_cache_path(out_dir: Path, case: str) -> Path:
    """Where a case's painted FG/BG seeds are cached (full-frame, 0/1/2 PNG)."""
    return out_dir / f"{case}_seeds.png"


def load_cached_seeds(out_dir: Path, case: str, shape: tuple[int, int]):
    """Load saved seeds for a case, or None if absent / wrong shape / unreadable."""
    path = seeds_cache_path(out_dir, case)
    if not path.exists():
        return None
    try:
        arr = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    except (OSError, ValueError):
        return None
    resampled = _resample_grid(arr, shape, reduce_max=False)
    if resampled is None:
        print(f"  {case}: cached seeds {arr.shape} != frame {shape}; ignoring.")
        return None
    if resampled.shape != arr.shape:
        print(f"  {case}: resampled cached seeds {arr.shape} -> {shape} for this run.")
    print(f"  {case}: loaded saved seeds ({seeds_cache_path(out_dir, case).name}).")
    return np.ascontiguousarray(resampled, dtype=np.uint8)


def save_cached_seeds(out_dir: Path, case: str, seeds: np.ndarray) -> None:
    """Persist full-frame seeds (0=unset, 1=vein, 2=background) for reuse."""
    path = seeds_cache_path(out_dir, case)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(seeds.astype(np.uint8), mode="L").save(path)
    print(f"  {case}: saved seeds -> {path.name}")


def load_existing_mask(out_dir: Path, case: str, shape: tuple[int, int]):
    """Load a previously saved ``{case}.nii`` as a full-frame (H, W) mask, or None.

    Mirrors ``main.py:load_and_preprocess``: the on-disk nii is (W, H, 1), so we
    squeeze and transpose back to (H, W). Returns None if absent, unreadable, or a
    different frame size, so the user can keep refining an existing mask in napari.
    """
    nii_path = out_dir / f"{case}.nii"
    if not nii_path.exists():
        return None
    try:
        import nibabel as nib

        data = np.asarray(nib.load(str(nii_path)).dataobj)
    except Exception as e:  # noqa: BLE001 - any read error -> start blank
        print(f"  {case}: could not read existing mask ({e}); starting blank.")
        return None
    mask = (np.squeeze(data) > 0).T.astype(np.uint8)
    # An earlier run at a different --upscale stored the mask on a coarser/finer
    # grid; resample it (nearest up, block-max down) so the editing loop still
    # works. It is re-saved at this run's resolution.
    resampled = _resample_grid(mask, shape, reduce_max=True)
    if resampled is None:
        print(f"  {case}: existing mask {mask.shape} != frame {shape}; not loading.")
        return None
    if resampled.shape != mask.shape:
        print(f"  {case}: resampled existing mask {mask.shape} -> {shape} to refine.")
    print(f"  {case}: loaded existing vessel mask ({nii_path.name}) to refine.")
    return np.ascontiguousarray(resampled, dtype=np.uint8)


def segment_in_napari(
    image_path: Path,
    case: str,
    dataset: str,
    brush_radius: int | None,
    out_dir: Path,
    spacing_known: float | None,
    allow_manual_scale: bool,
    upscale: int = 1,
    scale_points: list[list[float]] | None = None,
    spacing_label: str | None = None,
):
    """One napari window for the whole segmentation stage.

    Paint a ``seeds`` Labels layer (1=vein, 2=background), press ``e`` (or the
    dock button) to run the EM segmentation into a ``vessel`` layer, then refine
    that mask in place — all without reopening a window. When the spacing is not
    yet known and ``allow_manual_scale`` is set, a ``scale`` Points layer + ``m``
    let you measure the scale bar (click its two ends, type its length).

    When the spacing is already resolved (override/cache), a "Loaded spacing" dock
    label always shows ``spacing_label`` (e.g. "0.1012 mm/px (10 mm / 98.8 px,
    manual)") so the value is visible even when no overlay can be drawn. If
    ``scale_points`` (the measured legend endpoints, in native (row, col) pixels)
    is also known, it is additionally drawn as a read-only ``scale (detected)``
    overlay so the user can confirm the spacing locked onto the real scale bar.

    Returns ``(full_rgb, full_mask, measured_spacing)`` (the last is a
    ``spacing.Spacing`` if measured here, else None), or None if no usable mask
    was produced. Seeds are cached to ``{case}_seeds.png`` for re-runs.
    """
    import napari
    from magicgui.widgets import Container, Label, LineEdit, PushButton

    _, x0, (H, W), full_rgb, reference = _load_for_segmentation(
        image_path, dataset, upscale
    )
    # Segment whichever seam-side panel the seeds land in (see _seed_region), so
    # the region isn't a hardcoded side. Vesselness is computed lazily per region
    # and cached, so re-running EM as the seeds are adjusted recomputes nothing.
    # The Frangi sigmas are scaled by the upscale factor so the same physical
    # vessel widths are captured on the finer grid (just localized more precisely).
    full_gray = np.asarray(Image.fromarray(full_rgb).convert("L"), dtype=np.uint8)
    sigmas = tuple(s * upscale for s in config.FRANGI_SIGMAS)
    vness_cache: dict[tuple[int, int], np.ndarray] = {}

    seeds0 = load_cached_seeds(out_dir, case, (H, W))
    if seeds0 is None:
        seeds0 = np.zeros((H, W), dtype=np.uint8)

    # Scale the brush with the frame so painting covers the same anatomy at any
    # upscale (a 4px brush on a 2x frame would otherwise feel half as wide).
    brush = (brush_radius or config.DEFAULT_BRUSH_RADIUS) * upscale
    state: dict = {"spacing": None}

    viewer = napari.Viewer(title=f"Segment vessel — {case}")
    viewer.add_image(full_rgb, name="DSA")
    if dataset == "tju" and reference is not None:
        viewer.add_image(reference, name="tracked (reference)", opacity=0.5)
    seeds_layer = viewer.add_labels(seeds0.copy(), name="seeds")
    seeds_layer.selected_label = 1
    seeds_layer.brush_size = brush
    seeds_layer.mode = "paint"
    vessel0 = load_existing_mask(out_dir, case, (H, W))
    if vessel0 is None:
        vessel0 = np.zeros((H, W), dtype=np.uint8)
    vessel_layer = viewer.add_labels(vessel0, name="vessel")

    # When the spacing is already known, draw the measured legend span (endpoints
    # in native pixels, scaled to this frame) so it can be eyeballed against the
    # actual scale bar. A line + its two endpoints; not meant to be edited.
    if scale_points:
        pts = np.asarray(scale_points, dtype=float) * upscale
        viewer.add_shapes(
            [pts[:2]],
            shape_type="line",
            edge_color="red",
            edge_width=max(2, 2 * upscale),
            name="scale span",
        )
        viewer.add_points(
            pts, name="scale (detected)", ndim=2, size=10 * upscale, face_color="red"
        )

    def run_em() -> None:
        seeds_full = np.asarray(seeds_layer.data, dtype=np.uint8)
        c0, c1 = _seed_region(seeds_full, x0, W)
        seeds_c = seeds_full[:, c0:c1]
        if (
            not (seeds_c == scribble.FG_LABEL).any()
            or not (seeds_c == scribble.BG_LABEL).any()
        ):
            fg_f = int((seeds_full == scribble.FG_LABEL).sum())
            bg_f = int((seeds_full == scribble.BG_LABEL).sum())
            print(
                "  paint BOTH vein (label 1) and background (label 2) first."
                f"\n    seeds in whole frame: FG={fg_f} BG={bg_f}"
            )
            return
        gray_c = full_gray[:, c0:c1]
        if (c0, c1) not in vness_cache:
            print(f"  {case}: computing vesselness on cols {c0}:{c1}...")
            vness_cache[(c0, c1)] = features.vesselness(gray_c, sigmas=sigmas)
        feats_c = features.feature_stack(gray_c, vness_cache[(c0, c1)])
        posterior = segment.em_posterior(feats_c, seeds_c)
        mask_c = segment.regularize(gray_c, seeds_c, posterior)
        full_mask = np.zeros((H, W), dtype=np.uint8)
        full_mask[:, c0:c1] = mask_c
        vessel_layer.data = full_mask
        vessel_layer.selected_label = 1
        print(
            f"  {case}: EM produced {int((full_mask > 0).sum())} vessel px "
            f"(segmented cols {c0}:{c1})."
        )

    def measure_scale() -> None:
        pts = np.asarray(scale_layer.data)
        if len(pts) < 2:
            scale_status.value = "place two points on the 'scale' layer first"
            return
        # Points are clicked on the upscaled frame; divide by the factor so the
        # spacing comes out in mm per *native* pixel (the cache/finalize convention;
        # run_one then divides by the factor again to get mm per voxel).
        length_px = float(np.linalg.norm(pts[1] - pts[0])) / upscale
        text = length_edit.value.strip()
        if not text:
            scale_status.value = (
                f"{length_px:.1f} px — type the bar length, then Measure"
            )
            return
        sp = spacing_from_measurement(length_px, text)
        if sp is None:
            scale_status.value = f"couldn't read '{text}' — try '7 mm' or '1 cm'"
            return
        # Record the clicked ends in native (row, col) pixels (divide out the
        # upscale, matching the mm/native-px convention) so a later run can redraw
        # them as the verification overlay.
        sp.endpoints = (
            (float(pts[0][0]) / upscale, float(pts[0][1]) / upscale),
            (float(pts[1][0]) / upscale, float(pts[1][1]) / upscale),
        )
        state["spacing"] = sp
        scale_status.value = (
            f"{sp.mm_per_px:.4f} mm/px  ({sp.value_mm:g} mm / {length_px:.1f} px)"
        )
        print(f"  {case}: {sp}")

    # Dock controls + an on-canvas step guide.
    widgets = []
    if spacing_known is not None:
        widgets.append(
            Label(value=f"Loaded spacing: {spacing_label or f'{spacing_known:.4f} mm/px'}")
        )
    run_btn = PushButton(text="Run EM (e)")
    run_btn.clicked.connect(run_em)
    viewer.bind_key("e", lambda v: run_em(), overwrite=True)
    widgets.append(run_btn)

    if allow_manual_scale:
        scale_layer = viewer.add_points(
            name="scale", ndim=2, size=8, face_color="yellow"
        )
        scale_layer.mode = "add"
        length_edit = LineEdit(label="bar length", tooltip="e.g. '7 mm' or '1 cm'")
        scale_status = Label(value="place two points on the 'scale' layer")
        scale_btn = PushButton(text="Measure scale (m)")

        def _show_px(*_) -> None:
            pts = np.asarray(scale_layer.data)
            scale_status.value = (
                f"{float(np.linalg.norm(pts[1] - pts[0])) / upscale:.1f} px"
                " — type length, then Measure"
                if len(pts) >= 2
                else "place two points on the 'scale' layer"
            )

        scale_layer.events.data.connect(_show_px)
        scale_btn.clicked.connect(measure_scale)
        viewer.bind_key("m", lambda v: measure_scale(), overwrite=True)
        widgets += [length_edit, scale_btn, scale_status]

    # On-canvas step guide — scale measurement first (when needed), then segment.
    steps = []
    if allow_manual_scale:
        steps.append("Place 2 points on 'scale', type its length, press Measure scale")
    steps += [
        "Paint 'seeds': label 1 = vein, 2 = background\n"
        "   (brush adds, eraser removes, Ctrl+Z undoes)",
        "Press 'e' (or Run EM) to segment",
        "Refine the 'vessel' layer with the brush",
    ]
    guide = [f"{i}) {s}" for i, s in enumerate(steps, 1)]
    if scale_points:
        guide.append(
            "Red 'scale (detected)' marks the measured legend — check it sits on "
            "the scale bar (the spacing is wrong if it doesn't)."
        )
    guide.append("Close the window when done (saves the mask).")

    viewer.window.add_dock_widget(
        Container(widgets=widgets), area="right", name="actions"
    )
    viewer.text_overlay.visible = True
    viewer.text_overlay.font_size = 10
    viewer.text_overlay.text = "\n".join(guide)

    print(f"  {case}: napari open — follow the on-screen steps, then close to save.")
    napari.run()

    seeds_final = np.asarray(seeds_layer.data, dtype=np.uint8)
    if seeds_final.any():
        save_cached_seeds(out_dir, case, seeds_final)

    # If there's no mask yet (no EM run, no existing mask loaded), try EM from the
    # painted seeds so a paint-then-close flow still works.
    if int((np.asarray(vessel_layer.data) > 0).sum()) == 0:
        run_em()
    full_mask = (np.asarray(vessel_layer.data) > 0).astype(np.uint8)
    if full_mask.sum() == 0:
        print(f"  {case}: no mask produced — skipping save.")
        return None
    return full_rgb, full_mask, state["spacing"]


# ---------------------------------------------------------------------------
# Step 4: guarded save
# ---------------------------------------------------------------------------
def _compare_existing(
    full_mask: np.ndarray, nii_path: Path, spacing: float
) -> tuple[bool, bool]:
    """Compare a mask/spacing against the on-disk nii, voxels and spacing separately.

    Returns ``(mask_equal, spacing_equal)``. ``mask_equal`` compares the full
    binary voxel grid (every pixel) in exactly the layout ``write_mask_nii``
    produces; ``spacing_equal`` compares the header spacing as float32 (how the
    NIfTI header stores it) to avoid a spurious mismatch from the float64->float32
    round-trip. Any read error or shape mismatch returns ``(False, False)`` so the
    caller falls back to the normal guarded path.
    """
    import nibabel as nib

    try:
        img = nib.load(str(nii_path))
        existing = np.asarray(img.dataobj)
        existing_spacing = float(img.header.get_zooms()[0])
    except Exception as e:  # noqa: BLE001 - unreadable existing -> not comparable
        print(f"  {nii_path.stem}: could not compare existing mask ({e}).")
        return False, False

    # Match write_mask_nii's layout: binary (H, W) -> (W, H, 1) uint8.
    new_vol = (full_mask > 0).astype(np.uint8).T[..., np.newaxis]
    existing_vol = (existing > 0).astype(np.uint8)
    if existing_vol.shape != new_vol.shape:
        return False, False
    mask_equal = np.array_equal(existing_vol, new_vol)
    spacing_equal = float(np.float32(spacing)) == existing_spacing
    return mask_equal, spacing_equal


def guarded_save(
    full_mask: np.ndarray, nii_path: Path, spacing: float, force: bool
) -> bool:
    """Write the mask to ``nii_path``, refusing to overwrite without confirmation.

    The pipeline always targets ``data/{case}.nii`` (so ``main.py`` finds it). The
    overwrite prompt guards against clobbering existing *mask* edits, so it keys on
    the voxels only:

    - identical voxels and spacing -> nothing to do, no prompt, no write;
    - identical voxels but different spacing -> no mask is at risk, so reconcile
      the header silently (no prompt);
    - different voxels -> overwrite with ``--force`` (warned), else ask to confirm
      (a declined or non-interactive prompt keeps the existing file).
    """
    if nii_path.exists():
        mask_equal, spacing_equal = _compare_existing(full_mask, nii_path, spacing)
        if mask_equal and spacing_equal:
            print(f"  {nii_path.stem}: identical to existing mask — keeping it.")
            return False
        if mask_equal:
            # Only the spacing differs: no mask edits at stake, reconcile silently.
            print(f"  {nii_path.stem}: mask unchanged; updating spacing only.")
        else:
            print(f"  WARNING: {nii_path} already exists.")
            if force:
                print("  --force given; overwriting.")
            elif not sys.stdin.isatty():
                print("  Not overwriting (no --force, non-interactive). Skipped.")
                return False
            else:
                resp = input(f"  Overwrite {nii_path.name}? [y/N] ").strip().lower()
                if resp not in ("y", "yes"):
                    print("  Skipped (existing mask kept).")
                    return False

    n_px = segment.write_mask_nii(full_mask, nii_path, spacing)
    print(f"  {nii_path.stem}: {n_px} vessel px -> {nii_path}")
    return True


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_one(image_path: Path, out_dir: Path, args) -> tuple[str, dict] | None:
    """Segment one image and (by default) analyze it in the same run.

    Returns ``(nii_path, metrics)`` when analysis produced metrics, else None.
    The returned tuples are aggregated by ``main()`` into ``metrics.xlsx``.
    """
    case = image_path.stem
    dataset = args.dataset or infer_dataset(image_path)
    print(f"\n=== {case} ({image_path.name})  [dataset: {dataset}] ===")

    # 1. spacing, non-interactively (override > cache). If unresolved, the manual
    # measurement happens inside the napari window in step 2.
    spacing, needs_manual, scale_points, spacing_label = resolve_spacing(
        case, out_dir, args.spacing, args.manual_spacing,
    )
    allow_manual = needs_manual and not args.no_manual_spacing and sys.stdin.isatty()

    # 2. segmentation + refinement (+ optional scale measurement) in ONE window.
    # The frame is upsampled --upscale x inside, so vesselness/EM/random-walker all
    # run on the finer grid; full_mask and full_rgb come back at that resolution.
    result = segment_in_napari(
        image_path, case, dataset, args.brush, out_dir, spacing, allow_manual,
        args.upscale, scale_points=scale_points, spacing_label=spacing_label,
    )
    if result is None:
        return None
    full_rgb, full_mask, measured = result

    # Finalize spacing. `spacing` (override/cache) and a manual measurement are
    # both in mm per *native* pixel, so divide by the upscale factor to get the
    # per-voxel spacing of the finer grid we segmented on — keeping every main.py
    # metric in mm. The existing-nii/1.0 fallback is already per-voxel, so it is not
    # divided again.
    native_spacing = spacing
    if native_spacing is None and measured is not None:
        native_spacing = measured.mm_per_px
        save_spacing(out_dir, case, native_spacing, "manual", measured)

    if native_spacing is not None:
        spacing = native_spacing / args.upscale
    else:
        print(f"  {case}: falling back to existing/1.0 spacing.")
        spacing = segment.resolve_spacing(case, out_dir, None)

    if args.upscale > 1:
        print(
            f"  {case}: upscaled {args.upscale}x -> {full_mask.shape} grid, "
            f"{int((full_mask > 0).sum())} vessel voxels, "
            f"spacing {spacing:.4f} mm/voxel."
        )

    # 3. guarded save (the segmentation is reviewed on the QC page from step 4)
    nii_path = out_dir / f"{case}.nii"
    guarded_save(full_mask, nii_path, spacing, args.force)

    # 4. analysis: napari path pick -> FMM centerline -> metrics + {case}_qc.png.
    # Runs by default on the mask we just wrote (or an existing one we kept).
    if args.no_analyze:
        return None
    if not nii_path.exists():
        print(f"  {case}: no mask written — skipping analysis.")
        return None

    print(f"  {case}: analyzing {nii_path.name} (click the vein path)...")
    metrics = analysis.analyze_vein(str(nii_path))
    if metrics is None:
        return None
    return str(nii_path), metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="image file(s) or folder(s). Optional with --next "
        "(defaults to every known dataset).",
    )
    parser.add_argument(
        "--next",
        action="store_true",
        help="process only the next image still missing a report (no need to name "
        "a file; searches the given inputs, or every known dataset if none).",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        default=str(config.NII_DIR),
        help="Where to write .nii masks. Default: data/ (so main.py finds them).",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        choices=["current", "jap", "tju"],
        help="source dataset (default: auto-detected from the path — TJU/ -> tju, "
        "JAP/ -> jap, else current). Sets the panel layout. 'tju' segments the "
        "whole raw frame and shows the separate '{case} tracked.png' as the "
        "scribble reference.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=None,
        help="mm per pixel (isotropic). Skips the in-window measurement when given.",
    )
    parser.add_argument(
        "--manual-spacing",
        action="store_true",
        help="remeasure the scale bar by hand even if a spacing is already cached "
        "(click its two ends, then type its physical length).",
    )
    parser.add_argument(
        "--no-manual-spacing",
        action="store_true",
        help="don't offer the in-window scale measurement when nothing is cached "
        "(batch).",
    )
    parser.add_argument(
        "--brush",
        type=int,
        default=None,
        help="initial napari seed brush size in px "
        f"(default {config.DEFAULT_BRUSH_RADIUS}).",
    )
    parser.add_argument(
        "--upscale",
        type=int,
        default=config.UPSCALE_FACTOR,
        metavar="N",
        help="bicubically upsample the frame N x as the first step, so vesselness, "
        "EM and the random walker all run on the finer grid (each pixel -> NxN "
        "voxels; Frangi sigmas scaled by N; spacing divided by N so mm metrics are "
        f"unchanged). 1 disables it. Default {config.UPSCALE_FACTOR}. Higher N is "
        "~N^2 x the EM/random-walker cost.",
    )
    parser.add_argument(
        "--no-analyze",
        action="store_true",
        help="stop after saving the .nii; skip the click-path analysis + report.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing data/{case}.nii without asking.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.next:
        # With explicit inputs, expand them as usual; with none, sweep every
        # dataset's cases (covers the JAP/ and TJU/ subdirs and skips the TJU
        # "* tracked" reference files), which a flat folder glob would miss.
        if args.inputs:
            images = segment.collect_inputs(args.inputs)
        else:
            images = [
                loader.image_path
                for cls in datasets.DATASETS.values()
                for loader in cls.cases()
            ]
        pending = [
            p
            for p in images
            if not os.path.exists(analysis.get_report_path(str(out_dir / f"{p.stem}.nii")))
        ]
        if not pending:
            print("No images left — every image already has a report.")
            return
        image_path = pending[0]
        print(f"Next image: {image_path}")
        print(f"{len(pending)} image(s) still missing reports (this one included).")
        images = [image_path]
    else:
        images = segment.collect_inputs(args.inputs)
        if not images:
            raise SystemExit("No image inputs found.")
        print(
            f"Pipeline over {len(images)} image(s): "
            f"{', '.join(i.stem for i in images)}"
        )

    all_metrics = []
    for image_path in images:
        result = run_one(image_path, out_dir, args)
        if result is not None:
            all_metrics.append(result)

    if all_metrics:
        analysis.export_all_metrics(all_metrics, str(out_dir))
        analysis.export_qc_pdf(str(out_dir))
        print(f"\nDone. QC pages, qc_report.pdf + {out_dir / 'metrics.xlsx'} / .csv written.")
    elif args.no_analyze:
        print("\nDone (segmentation only). Run the analysis with:  uv run main.py")
    else:
        print("\nDone. No metrics produced.")


if __name__ == "__main__":
    main()
