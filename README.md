# CIIRC Veins Analysis

A Python tool for segmenting and analyzing vein structures from DSA (digital
subtraction angiography) images. It turns a raw image into a NIfTI vessel
mask, then computes geometric metrics along the vein path — length,
tortuosity, diameter, volume, and curvature — and generates a one-page QC
report per case plus a single PDF for the whole dataset.

## Features

- Semi-automatic segmentation (`edge_em/`): Frangi vesselness → Gaussian-Mixture
  EM → random-walker regularization, seeded by a few painted scribbles. No
  trained model required.
- Automatic scale-bar measurement/calibration (`spacing.py`), cached per case.
- Skeletonizes the mask and builds a graph; interactive path selection in
  napari.
- Refines the centerline with the Fast Marching Method (FMM).
- Computes metrics: length, tortuosity, diameter, volume, curvature, turn
  counts, plus length-normalized shape features (see
  [Metrics reference](#metrics-reference)).
- Generates a one-page QC report per case (`_qc.png`), bundles them into
  `qc_report.pdf`, and aggregates all cases into `metrics.xlsx`.

## Installation

This project uses `uv` for dependency management (Python 3.12+).

```bash
# Install core dependencies
uv sync

# Also install the classical segmentation stack (Frangi + EM), used by
# pipeline.py / edge_em/
uv sync --extra edge-em
```

## Usage

### 1. End-to-end pipeline (recommended)

`pipeline.py` takes a raw DSA frame all the way to a report in one interactive
run: upscale → measure/load spacing → segment (napari) → save mask → analyze
(napari) → report.

```bash
# One image
uv run python pipeline.py pre-data/10m.jpg

# Skip the scale-bar measurement if you already know the spacing
uv run python pipeline.py pre-data/10m.jpg --spacing 0.0568

# A whole folder, or the next image still missing a report
uv run python pipeline.py pre-data/
uv run python pipeline.py --next

# Segment only (skip the click-path analysis)
uv run python pipeline.py pre-data/10m.jpg --no-analyze
```

Each image opens two napari windows:

- **Segment** — paint `1` (vein) / `2` (background) seeds on the `seeds`
  layer, press `e` to run the Frangi + EM + random-walker segmentation, then
  refine the resulting `vessel` layer with the brush. If spacing hasn't been
  measured yet, place two points on the `scale` layer, type the bar's physical
  length, and press `m`. Close the window to save.
- **Analyze** — click points along the vein path on the `endpoints` layer, in
  order; select + Delete to remove a wrong point. Close the window to run the
  FMM refinement and generate the report.

Painted seeds, measured spacing, and clicked path points are all cached
(`{case}_scribbles.json`, `{case}_spacing.json`, `{case}_clicks.json`), so
re-running a case skips the corresponding interactive step.

### 2. Analysis only

If `data/` already has masks (e.g. from a previous pipeline run or exported
from 3D Slicer), `main.py` re-runs just the analysis half over every `.nii`
file that doesn't yet have a report:

```bash
uv run main.py
```

### 3. Re-exporting all metrics

`main.py` and `pipeline.py` only accumulate metrics for cases processed in
that run, so re-running after every case already has a report produces a
`metrics.xlsx` with just one row. `reanalyze_cached.py` rebuilds the full
spreadsheet (and `qc_report.pdf`) non-interactively from every mask with
cached click points:

```bash
MPLBACKEND=Agg uv run python reanalyze_cached.py
```

### Output files (in `data/`)

| File | Produced by | Contents |
|---|---|---|
| `{case}.nii` | `pipeline.py` / `edge_em/segment.py` | Binary vessel mask, spacing baked into the header |
| `{case}_scribbles.json` | segmentation step | Cached foreground/background paint strokes |
| `{case}_spacing.json` | spacing step | Resolved mm/px and the measured scale-bar endpoints |
| `{case}_clicks.json` | analysis step | Cached vein-path click points |
| `{case}_qc.png` | `main.py` | One-page QC report (see below) |
| `qc_report.pdf` | `main.py` / `pipeline.py` / `reanalyze_cached.py` | Every `_qc.png` in one PDF, with a cover page listing cases, flagged first |
| `metrics.xlsx` | `main.py` / `pipeline.py` / `reanalyze_cached.py` | Aggregate spreadsheet of all computed metrics and shape features, plus `QC Status` / `QC Flags` |
| `metrics.csv` | same | Same rows unrounded, snake_case headers, `case` / `dataset` / `qc_status` columns — ready for `pd.read_csv` |

Delete a `_qc.png` to have a case reprocessed.

### QC report

Each `{case}_qc.png` walks a reviewer through numbered checks:

1. **Calibration** — the measured line drawn on the image's own scale bar, with
   its mm (only when the scale-bar endpoints were saved).
2. **Segmentation** — the raw crop, and for TJU the clinician's tracing.
3. **Mask + path** — mask outline, centerline colored start→end with direction
   arrows (denser in tight turns), numbered click points.
4. **Measurements** — diameter drawn as width lines across the vessel, the
   minimum diameter in red, detected turns numbered `T1…` with their angle.
5. **Profiles** — diameter and curvature vs distance, colored like the path,
   turns numbered as on the image.
6. **Summary** — metrics and automatic flags: missing/implausible/mismatched
   spacing, missing source image, multiple mask pieces, mask touching the image
   border, clicks far from the path, sub-3 px minimum diameter, and heavy
   curvature capping (projection folds). Thresholds are the `QC_*` constants
   in `main.py`. A flag means "look here", not "wrong".

## Metrics reference

One row per vein in `metrics.xlsx` / `metrics.csv`, in the order of
`METRIC_COLUMNS` in `main.py`. The CSV header is the key, and the XLSX header
is the label in parentheses. Everything is computed along the clicked,
FMM-refined centerline in physical units, using the per-axis pixel spacing.

### Core metrics (`calculate_metrics`)

| Key (XLSX header) | Definition | Unit |
|---|---|---|
| `length` (Length) | Arc length of the centerline | mm |
| `tortuosity` (Tortuosity) | Path length / straight-line chord between the ends (1 = straight) | – |
| `avg_diameter`, `min_diameter`, `max_diameter`, `std_diameter` | Mean / min / max / SD of the diameter profile. Diameter = 2 × the spacing-aware distance transform at each path point, Gaussian-smoothed (`DIAMETER_SMOOTHING_SIGMA`). One median diameter is skipped at each tip, where the distance transform falls to 0 | mm |
| `volume` (Volume) | Sum of cylinders, π(d/2)² × segment length, over the whole path | mm³ |
| `avg_curvature`, `max_curvature`, `std_curvature` | Mean / max / SD of \|κ\|, where κ is the analytic curvature of a smoothing cubic B-spline fit to the centerline (`SPLINE_SMOOTHING_FACTOR`), **after capping** (see below) | 1/mm |
| `inflection_count` (Inflections) | Number of turn-direction reversals between consecutive significant bends. A bend is a constant-sign arc of κ that turns ≥ `MIN_ARC_ANGLE_DEG` (20°). Ripples below `INFLECTION_DEADBAND_FRAC` × max\|κ\| are ignored | count |
| `total_curvature` (Total Turning) | Total absolute turning ∫\|κ\| ds over the vein | ° |
| `turns_90`, `turns_180`, `turns_270` | Bends whose turn angle ∫κ ds over the arc is ≥ 90° / 180° / 270°. These are cumulative, so a 200° bend counts in both `turns_90` and `turns_180` | count |

### Shape features (`shape_features`)

These are scale-robust and length-normalized, for classification. Diameter
features use the profile resampled to uniform arc length, with one median
diameter trimmed from each end (`DIAMETER_END_TRIM_DIAMETERS`). Ratios to the
median cancel errors in the pixel spacing. Counts are divided by the vein
length, so they don't grow with how much of the vein was clicked.

| Key | Definition | Unit |
|---|---|---|
| `diam_median`, `diam_p10`, `diam_p90` | Median / 10th / 90th percentile diameter | mm |
| `diam_cv` | SD / mean diameter | – |
| `ectasia_ratio` | P90 / median: broad dilation | – |
| `narrowing_ratio` | P10 / median: narrowing | – |
| `dilated_frac` | Share of length with diameter > `DILATION_RATIO` (1.5) × median: focal varices | 0–1 |
| `diam_change_per_cm` | Σ\|Δd\| / median / length: caliber irregularity (beading) | 1/cm |
| `diam_skew` | Skewness of the diameter (local bulges give a right skew) | – |
| `diam_kurtosis` | Excess kurtosis. Exploratory, and noisy on a smoothed profile | – |
| `turning_deg_per_cm` | `total_curvature` / length (sum-of-angles metric, Bullitt et al. 2003) | °/cm |
| `inflections_per_cm` | `inflection_count` / length | 1/cm |
| `turns90_per_cm`, `turns180_per_cm` | `turns_90`, `turns_180` / length | 1/cm |
| `icm` | (inflections + 1) × tortuosity (inflection-count metric, Bullitt et al. 2003) | – |
| `curv_capped_frac` (Curvature Capped Fraction) | Share of centerline points whose curvature had to be capped at the physical limit (see below) | 0–1 |

### Curvature capping and `curv_capped_frac`

The images are 2D projections of 3D vessels. Where a vein runs toward or away
from the camera, the projection folds it back over itself into a near-hairpin.
That fold produces a curvature spike that is physically impossible: its radius
of curvature is far smaller than the vessel itself. A single spike like this
would dominate `max_curvature`, `std_curvature` and the ∫|κ|ds turning totals.

A centerline cannot bend tighter than its own lumen, so at each point |κ| is
clipped to

    κ_max = 1 / (CURVATURE_CAP_RADIUS_FACTOR × r_local)

`r_local` is the local lumen radius from the distance map, and the factor
defaults to 1.0, the hard physical floor. The signed curvature is clipped, so
inflection signs (and so the bend counts) are preserved.

#### How `r_local` is computed

`r_local` is the same per-point radius that feeds the diameter profile
(`path_radii_mm` in `calculate_metrics`):

1. **Spacing-aware distance map.** A Euclidean distance transform of the
   binary mask is computed with `distance_transform_edt(binary_mask,
   sampling=(ps_y, ps_x))`. Every foreground pixel then holds its distance to
   the nearest background pixel, in **millimetres**, with anisotropic spacing
   handled correctly (the mask is laid out rows = Y, cols = X, so the sampling
   order is `(ps_y, ps_x)`). This map is recomputed here on purpose. The
   pixel-space map from `medial_axis` is not reused, because it would bias
   radii when `ps_x ≠ ps_y`.
2. **Sample along the centerline.** Each point of the final FMM centerline
   (subpixel, extended to the wall) is rounded to its nearest pixel, and the
   distance map is read there:
   `r_local[i] = dist_map_mm[round(y_i), round(x_i)]`.
   On a well-centred path, that distance to the nearest wall is the lumen
   radius.
3. **Unsmoothed.** The diameter profile is Gaussian-smoothed afterwards
   (`DIAMETER_SMOOTHING_SIGMA`), but that smoothing goes into a separate array.
   The cap uses the raw radii. No tip trimming is applied either, so the cap
   covers every path point.

The cap is then `κ_max[i] = 1 / (CURVATURE_CAP_RADIUS_FACTOR · r_local[i])`,
which has the same units (1/mm) as the spline curvature. Edge behaviour:

- **Near the tips**, where the path runs into the mask wall, `r_local` drops
  toward 0 and κ_max grows very large. The cap is effectively loose there.
- **Where `r_local = 0`** (a point that rounds onto a background pixel), κ_max
  is set to ∞, meaning no cap at that point. This avoids dividing by zero.
- A narrower stretch has a smaller radius, so its cap is less strict: tighter
  bends are allowed there. So the cap adapts to the local calibre and is not
  one global threshold.

#### What `CURVATURE_CAP_RADIUS_FACTOR` does

`CURVATURE_CAP_RADIUS_FACTOR` (in `main.py`, default `1.0`) sets the tightest
bend the centerline is allowed to make, measured in local vessel radii. In
radius-of-curvature terms (R = 1/κ), the cap enforces

    R ≥ CURVATURE_CAP_RADIUS_FACTOR × r_local

| Factor | Smallest allowed radius of curvature | Effect |
|---|---|---|
| `1.0` (default) | one lumen radius | Hard physical floor. A tighter bend would make the inner wall of the vessel fold over itself, so only impossible curvature is clipped |
| `> 1.0` (e.g. `2.0`) | that many radii | More aggressive. It also clips bends that are physically possible but very sharp, and `curv_capped_frac` rises |
| `< 1.0` | under one radius | Looser than physics allows. Projection-fold spikes start to come through again |
| `0` or very small | – | κ_max becomes huge or infinite, which in effect turns the cap off |

Worked example: on a 4 mm vein (`r_local` = 2 mm) with the default factor,
κ_max = 1 / (1.0 × 2 mm) = 0.5 mm⁻¹, so no bend is tighter than a 2 mm radius.
With factor `2.0`, κ_max is 0.25 mm⁻¹, which means a 4 mm radius.

The default is the only value with a physical justification. Raise it only if
you want to suppress sharp-but-real bends on purpose, and keep it fixed across
a dataset. Changing it changes `max_curvature`, `std_curvature`,
`avg_curvature`, the ∫|κ|ds turning totals, the turn counts and
`curv_capped_frac`, so values computed with different factors can't be
compared.

`curv_capped_frac` is the fraction of path points where |κ| exceeded κ_max
before clipping. It is not a vein-shape feature. It measures how much the
curvature metrics were affected by projection artifacts, or by a jagged
centerline or mask:

- **≈ 0**: the curvature stats are raw geometry.
- **High**: the vein has foreshortened or folded stretches, so its
  curvature/turn metrics understate the real 3D tortuosity in those places.
  Above `QC_CURV_CAPPED_WARN_FRAC` (5 %) the QC report flags the case.

Use it as a QC covariate, for example to exclude or down-weight cases, rather
than as a classifier input.

### QC columns

| Key (XLSX header) | Definition |
|---|---|
| `case` / `dataset` (File / Dataset) | Case name and source dataset (`current`, `JAP`, `TJU`) |
| `qc_status` (QC Status) | `OK`, or `REVIEW` when any automatic check fired |
| `qc_flags` (QC Flags) | The fired checks, `|`-separated (see the QC report summary above) |

## Datasets

Source images live under `pre-data/` and come from three sources, each with a
different label/raw layout (`datasets.py` is the single source of truth for
these conventions and is auto-detected from the path):

- **current** — `pre-data/{n}m.jpg`, one side-by-side frame (left panel:
  clinician's tracing, right panel: clean copy).
- **JAP** — `pre-data/JAP/JAP{n}.jpg`, same side-by-side layout.
- **TJU** — `pre-data/TJU/TJU{n}.png` (raw) + `TJU{n} tracked.png` (traced),
  as two separate files.

```bash
uv run python datasets.py current pre-data/10m.jpg
uv run python datasets.py tju pre-data/TJU/TJU1.png --save /tmp/out
```

## Manual segmentation with 3D Slicer (alternative)

Masks can also be produced by hand in 3D Slicer instead of `pipeline.py`:

1. Open image
2. Add point list — rename to F
3. Under "control points" expand "advanced options"
4. Click twice add new control point
5. Under the new two points click on the first one - the right most dot and place the point
6. Place second point
7. Run `slicer/scaling.py` in 3D Slicer's python console — make sure the length of image label is the same as the one in the script - 10.0 mm
8. Click on segmentations and add new segment, draw segmentation
9. Oversample the segment by clicking right of the segment volume button, selecting volume and then factor 4
10. Export to .nii file
11. Close scene and save it to a medical bundle

## Directory Structure

| Path | Purpose |
|---|---|
| `pipeline.py` | End-to-end pipeline: segment → analyze → report |
| `main.py` | Analysis half: skeleton → path selection → FMM → metrics → report |
| `edge_em/` | Semi-automatic segmentation: Frangi vesselness + Gaussian-Mixture EM + random walker |
| `spacing.py` | Parses a hand-measured scale-bar length into mm/px |
| `datasets.py` | Per-source dataset loaders (label/raw split conventions) |
| `reanalyze_cached.py` | Rebuilds `metrics.xlsx` and `qc_report.pdf` from every mask with cached click points |
| `slicer/scaling.py` | 3D Slicer script for calibrating spacing from a scale bar (manual workflow) |
| `pre-data/` | Source DSA images (`current`, `JAP/`, `TJU/`) |
| `data/` | Generated `.nii` masks, click/scribble/spacing caches, `_qc.png`, `qc_report.pdf`, `metrics.xlsx` |
| `saved-scenes/` | 3D Slicer scene files (`.mrb`) |
| `archive/` | Old/undersampled test files |
