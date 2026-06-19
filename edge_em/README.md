# edge_em — semi-automatic vein segmentation (Frangi → EM)

A classical, **no-training** alternative to `nnunet/`: **DSA image → binary vessel `.nii`**,
seeded by a few foreground/background scribbles. The output `.nii` drops straight into
`main.py` — you still click start/end and the existing FMM + metrics code runs unchanged.

Runs entirely on CPU; no model download. This module is fully separate from `main.py` and
`nnunet/` — all its code lives under `edge_em/`.

## How it works

```
image (.jpg/.jpeg/.png)
  ├─ Stage 1  Frangi vesselness (multi-scale) + Canny edges      features.py
  ├─ weak in  FG/BG scribbles (cached to data/{case}_scribbles.json)   scribble.py
  ├─ Stage 2  Gaussian-Mixture EM per class → posterior P(vessel)      segment.py
  ├─ Stage 3  random-walker regularization (scribbles + posterior)     segment.py
  └─ output   data/{case}.nii  (+ data/{case}_edge_em_overlay.png)
```

* **Frangi vesselness** is the "edge detection" stage — a multi-scale Hessian filter built for
  dark tubular structures, far better suited to veins than a generic learned edge detector and
  cheap on CPU.
* **EM** is a real Gaussian mixture per class (vessel / background), fitted via
  expectation-maximization (`sklearn.mixture.GaussianMixture`) on the scribbled pixels' features
  `[inverted intensity, vesselness]`.
* **Random walker** (`skimage.segmentation.random_walker`) turns the posterior + scribbles into a
  clean, connected mask, pinned by your scribbles and the high-confidence EM pixels.

## Install

```bash
uv sync --extra edge-em      # adds scikit-learn + pillow (Frangi/Canny/random_walker are in scikit-image already)
```

## Usage

```bash
uv run python edge_em/segment.py pre-data/10m.jpg
uv run python edge_em/segment.py pre-data/10m.png --spacing 0.0568
uv run python edge_em/segment.py pre-data/ -o /tmp/out        # whole folder, write elsewhere
```

In the scribble window:

- **left-drag** — paint a stroke of the active class
- **`f`** — switch to foreground (vein), **`b`** — switch to background
- **Enter** (or close the window) — finish

Scribbles are saved to `data/{case}_scribbles.json` and reused on re-run (delete that file to
re-draw). Then:

```bash
uv run main.py     # click start/end, get metrics
```

## Spacing

Like `nnunet/`, the segmentation only produces pixels. Pass `--spacing` (mm/pixel from your Slicer
scale-bar calibration) for calibrated metrics. If a `data/{case}.nii` already exists, its spacing is
reused automatically.

## Output & overwrite note

By default the mask is written to `data/{case}.nii` so `main.py` finds it — this **overwrites** any
existing mask for that case (e.g. a hand-painted one). Use `-o <dir>` to write elsewhere when you
want to compare against an existing label. An inspection overlay is always written to
`data/{case}_edge_em_overlay.png`.

## Tuning (`config.py`)

| Constant | Default | Description |
|---|---|---|
| `FRANGI_SIGMAS` | `(1..8)` | Vesselness scales (px); bracket vein half-widths |
| `CANNY_SIGMA` | `2.0` | Canny smoothing for the boundary cue |
| `GMM_COMPONENTS` | `2` | Mixture components per class (EM) |
| `RW_BETA` | `130` | Random-walker edge sensitivity (higher = sharper) |
| `RW_PRIOR_FG/BG_THRESH` | `0.90 / 0.10` | Posterior confidence to add soft seeds |
| `MIN_OBJECT_SIZE` | `64` | Drop connected components smaller than this (px) |
| `DEFAULT_BRUSH_RADIUS` | `4` | Scribble stroke radius (px); override with `--brush` |
