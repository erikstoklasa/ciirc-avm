# CIIRC Veins Analysis

A Python tool for analyzing vein structures from NIfTI segmentation masks. Computes geometric metrics along vein paths—length, tortuosity, diameter, volume, curvature, and turn counts—and generates a QC report per case plus one PDF for the dataset.

## Workflow

```
3D Slicer (manual segmentation)          main.py (automated analysis)
─────────────────────────────────        ─────────────────────────────
slicer/scaling.py → calibrate spacing    load_and_preprocess()
→ paint mask → export .nii               skeletonize_and_graph()
                                         get_user_path() [interactive]
                                         compute_centerline_fmm()
                                         calculate_metrics()
                                           ├ count_curves()
                                           └ shape_features()
                                         visualize_qc()
                                         export_all_metrics()  (.xlsx + .csv)
```

## Directory Structure

| Path | Purpose |
|---|---|
| `main.py` | Single entry point: full vein analysis pipeline |
| `slicer/scaling.py` | 3D Slicer script for calibrating image spacing from a scale bar |
| `data/` | Input `.nii` files, plus generated `_qc.png`, `_clicks.json`, `qc_report.pdf`, `metrics.xlsx`, and `metrics.csv` |
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

The script processes all `.nii` files in `data/` that don't yet have a `_qc.png`. For each file, an interactive matplotlib window opens showing the vein mask and skeleton graph:

- **Left click** — select points along the vein path
- **Right click** — remove last point
- **Enter** — confirm path and proceed with analysis

Output files saved to `data/`:
- `*_qc.png` — one-page QC report (calibration, segmentation, path, measurements, profiles, automatic flags)
- `*_clicks.json` — saved click points (skips interactive step on re-runs)
- `metrics.xlsx` — aggregate spreadsheet of all computed metrics (rounded, for people)
- `metrics.csv` — the same rows unrounded, with snake_case keys plus `case`, `dataset`, `qc_status` (for models)
- `qc_report.pdf` — every `_qc.png` in one PDF, cover page lists flagged cases first

Delete `_qc.png` files to reprocess.

## Key Configuration

Constants in `main.py` lines 19–29:

| Constant | Default | Description |
|---|---|---|
| `MANUAL_PIXEL_SIZE` | `None` | Override auto-detected pixel spacing |
| `MEDIAL_AXIS_SEED` | `0` | Seed for `medial_axis` tie-breaking, so skeletons and metrics are reproducible |
| `FMM_WAYPOINT_INTERVAL_MM` | `2.0` | Waypoint spacing for FMM centerline refinement |
| `DIAMETER_SMOOTHING_SIGMA` | `3.0` | Gaussian sigma for diameter profile smoothing |
| `CURVATURE_SMOOTHING_SIGMA` | `5.0` | Gaussian sigma for curvature calculation |
| `CORRIDOR_RADIUS_FACTOR` | `1.2` | FMM corridor expansion factor |
| `MIN_CORRIDOR_RADIUS` | `1.5` | Minimum FMM corridor radius (px) |
| `FMM_STEP_SIZE` | `0.5` | Gradient descent step size for FMM backtracking |

## Pipeline Details

1. **Load & Preprocess** — Reads `.nii` via nibabel, extracts binary mask and mm pixel dimensions
2. **Skeletonize & Graph** — Medial axis transform via `skimage.morphology.medial_axis`; builds weighted `networkx.Graph`
3. **User Path Selection** — Click points in napari; shortest graph path through them. The start/end clicks are path nodes themselves, joined to the skeleton by centre-seeking geodesics (`skimage.graph.MCP_Geometric`, cost 1/distance-to-wall); waypoints snap to the nearest skeleton node
4. **FMM Centerline Refinement** — Fast Marching Method (`skfmm.travel_time`) with gradient-descent backtracking for subpixel path; both ends are then carried on along their direction to the mask wall (unless no wall is near, i.e. a mid-vessel click)
5. **Metrics** — Length, tortuosity (path/chord ratio), diameter (distance transform, Gaussian-smoothed; min/mean/max/sd skip one median diameter at each tip), volume (cylindrical approximation), curvature (2nd derivative)
6. **Turn Counts** — Cumulative heading changes around curvature peaks; thresholds at 90°, 180°, 270°
7. **Shape features** — scale-robust, length-normalized features for classification (see below)
8. **QC report** — one page per case: calibration on the legend, raw/tracing, mask + path, width lines and numbered turns, profiles, automatic flags
9. **Export** — one row per vein (`METRIC_COLUMNS` in `main.py` defines the columns) to `data/metrics.xlsx` and `data/metrics.csv`

## Shape Features

Added to the same row as the metrics above, by `shape_features()`. Diameter features use the
diameter profile resampled to uniform arc length (every mm weighs the same) with one median
diameter trimmed off each end (`DIAMETER_END_TRIM_DIAMETERS`), where the distance transform
falls off toward the mask tips. Ratios to the median cancel errors in the estimated pixel
spacing; counts are divided by the vein length so they don't grow with how much was clicked.

| Key | Definition | Unit |
|---|---|---|
| `diam_median`, `diam_p10`, `diam_p90` | median / 10th / 90th percentile diameter | mm |
| `diam_cv` | std / mean diameter | – |
| `ectasia_ratio` | P90 / median: broad dilation | – |
| `narrowing_ratio` | P10 / median: narrowing | – |
| `dilated_frac` | share of length with diameter > `DILATION_RATIO` (1.5) × median: focal varices | – |
| `diam_change_per_cm` | Σ\|Δd\| / median / length: caliber irregularity (beading) | 1/cm |
| `diam_skew` | skewness of the diameter (local bulges → right skew) | – |
| `diam_kurtosis` | excess kurtosis — exploratory, noisy on a smoothed profile | – |
| `turning_deg_per_cm` | total turning / length (sum-of-angles metric, Bullitt et al. 2003) | °/cm |
| `inflections_per_cm` | inflection count / length | 1/cm |
| `turns90_per_cm`, `turns180_per_cm` | turn counts / length | 1/cm |
| `icm` | (inflections + 1) × tortuosity (inflection-count metric, Bullitt et al. 2003) | – |
| `curv_capped_frac` | share of curvature samples hitting the physical cap (projection folds); a QC covariate | – |

For a model, read `metrics.csv` and pick a handful of non-redundant columns:
`avg_curvature` ≈ `turning_deg_per_cm` up to a constant, and `volume` ≈ length × diameter².