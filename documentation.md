# CIIRC Veins Analysis

A Python tool for analyzing vein structures from NIfTI segmentation masks. Computes geometric metrics along vein paths—length, tortuosity, diameter, volume, curvature, and turn counts—and generates visual reports.

## Workflow

```
3D Slicer (manual segmentation)          main.py (automated analysis)
─────────────────────────────────        ─────────────────────────────
slicer/scaling.py → calibrate spacing    load_and_preprocess()
→ paint mask → export .nii               skeletonize_and_graph()
                                         get_user_path() [interactive]
                                         compute_centerline_fmm()
                                         calculate_metrics()
                                         count_turns()
                                         visualize_results()
                                         export_all_metrics_xlsx()
```

## Directory Structure

| Path | Purpose |
|---|---|
| `main.py` | Single entry point: full vein analysis pipeline |
| `slicer/scaling.py` | 3D Slicer script for calibrating image spacing from a scale bar |
| `data/` | Input `.nii` files, plus generated `_report.png`, `_clicks.json`, and `metrics.xlsx` |
| `pre-data/` | Reference JPG images (source photos for each `.nii`) |
| `saved-scenes/` | 3D Slicer scene files (`.mrb`) |
| `archive/` | Old/undersampled test files |

## Requirements

- Python 3.12+
- Dependencies: `matplotlib`, `networkx`, `nibabel`, `scikit-image`, `scikit-fmm`, `scipy`, `openpyxl`

## Installation

```bash
uv sync
```

## Usage

### 1. Preprocessing (in 3D Slicer)

1. Open an image in 3D Slicer
2. Place two fiducial markers on a 10mm scale bar
3. Run `slicer/scaling.py` in the Slicer Python console to calibrate pixel spacing
4. Create and paint a vein segmentation mask
5. Export the segmentation as `.nii` into the `data/` directory

### 2. Analysis

```bash
uv run main.py
```

The script processes all `.nii` files in `data/` that don't yet have a `_report.png`. For each file, an interactive matplotlib window opens showing the vein mask and skeleton graph:

- **Left click** — select points along the vein path
- **Right click** — remove last point
- **Enter** — confirm path and proceed with analysis

Output files saved to `data/`:
- `*_report.png` — 2x2 visualization (anatomy, diameter profile, curvature profile)
- `*_clicks.json` — saved click points (skips interactive step on re-runs)
- `metrics.xlsx` — aggregate spreadsheet of all computed metrics

Delete `_report.png` files to reprocess.

## Key Configuration

Constants in `main.py` lines 19–29:

| Constant | Default | Description |
|---|---|---|
| `MANUAL_PIXEL_SIZE` | `None` | Override auto-detected pixel spacing |
| `FMM_WAYPOINT_INTERVAL_MM` | `2.0` | Waypoint spacing for FMM centerline refinement |
| `DIAMETER_SMOOTHING_SIGMA` | `3.0` | Gaussian sigma for diameter profile smoothing |
| `CURVATURE_SMOOTHING_SIGMA` | `5.0` | Gaussian sigma for curvature calculation |
| `CORRIDOR_RADIUS_FACTOR` | `1.2` | FMM corridor expansion factor |
| `MIN_CORRIDOR_RADIUS` | `1.5` | Minimum FMM corridor radius (px) |
| `FMM_STEP_SIZE` | `0.5` | Gradient descent step size for FMM backtracking |

## Pipeline Details

1. **Load & Preprocess** — Reads `.nii` via nibabel, extracts binary mask and mm pixel dimensions
2. **Skeletonize & Graph** — Medial axis transform via `skimage.morphology.medial_axis`; builds weighted `networkx.Graph`
3. **User Path Selection** — Interactive matplotlib click-to-select, shortest path between clicked nodes
4. **FMM Centerline Refinement** — Fast Marching Method (`skfmm.travel_time`) with gradient-descent backtracking for subpixel path
5. **Metrics** — Length, tortuosity (path/chord ratio), diameter (distance transform, Gaussian-smoothed), volume (cylindrical approximation), curvature (2nd derivative)
6. **Turn Counts** — Cumulative heading changes around curvature peaks; thresholds at 90°, 180°, 270°
7. **Visualization** — 2x2 matplotlib figure with anatomy overlay, diameter profile, and curvature plot
8. **Export** — Aggregate metrics written to `data/metrics.xlsx` via openpyxl