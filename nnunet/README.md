# nnU-Net Vein Segmentation

Automates the manual Slicer painting step: **DSA `.jpg` → binary vessel `.nii`**.
The output `.nii` drops straight into `main.py` — you still click start/end and
the existing FMM + metrics code runs unchanged.

This module is fully separate from `main.py`; all nnU-Net data and models live
under `nnunet/`.

## Install

```bash
uv sync --extra nnunet      # adds nnunetv2 + torch + pillow
```

### Device (macOS has no CUDA)

nnU-Net defaults to CUDA, which doesn't exist on Apple Silicon. Both `train.py`
and `predict.py` default to `--device cpu`. `--device mps` (Apple GPU) is faster
but nnU-Net hits unsupported-op errors on some layers, so `cpu` is the safe
default. On a CUDA box, pass `--device cuda`.

CPU training is slow — pair it with a short trainer for anything interactive:
```bash
uv run python nnunet/train.py --trainer nnUNetTrainer_5epochs    # smoke test
uv run python nnunet/train.py --trainer nnUNetTrainer_100epochs  # rough model
```
Available: `nnUNetTrainer_{5,10,20,50,100,250,500,750}epochs` (default trainer is
1000 epochs — impractical on CPU). Inference is fine on CPU for one image at a time.

## Workflow

```
prepare_dataset.py     pre-data/*.jpg + data/*.nii  ->  nnUNet_raw/Dataset001_Veins/
train.py               plan + preprocess + train the 2D model
predict.py             new.jpg  ->  data/new.nii   (then run main.py as usual)
```

### 1. Build the training set
```bash
uv run python nnunet/prepare_dataset.py
```
Pairs every `pre-data/{name}.jpg` with `data/{name}.nii` (26 labeled cases),
converts to nnU-Net's 2D PNG format, and writes `dataset.json`.

### 2. Train
```bash
uv run python nnunet/train.py --folds 0 1 2 3 4    # full 5-fold (best)
uv run python nnunet/train.py                       # fold 0 only (quick start)
uv run python nnunet/train.py --trainer nnUNetTrainer_50epochs   # fast smoke test
```

### 3. Segment a new image
```bash
uv run python nnunet/predict.py pre-data/new.jpg --spacing 0.0568
uv run python nnunet/predict.py pre-data/new.jpg --folds 0 1 2 3 4   # match training
```
Then:
```bash
uv run main.py     # click start/end, get metrics
```

## Spacing (important)

Each DSA image has its own mm/pixel scale (from the scale bar). The model only
produces pixels — it cannot know the scale — so pass `--spacing` (the value from
your Slicer scale-bar calibration). Without it, lengths/diameters/volumes are
uncalibrated. If you re-segment an image that already has a `data/{name}.nii`,
the existing spacing is reused automatically.

## Orientation

On-disk nii is `(W, H, 1)`; `main.py` transposes to `(H, W)`. The scripts train
in `(H, W)` (matching the jpg) and write predictions back as `(W, H, 1)`, so the
round trip is consistent.
