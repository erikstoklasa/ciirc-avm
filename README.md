# CIIRC Veins Analysis

A Python tool for analyzing vein structures from NIfTI segmentation files. It calculates geometric metrics such as length, tortuosity, diameter, and curvature.

## Features

- Loads NIfTI (`.nii`) segmentation masks.
- Skeletonizes the structure and builds a graph.
- Allows interactive manual selection of the vein path.
- Refines the centerline using the Fast Marching Method.
- Computes metrics: Length, Tortuosity, Diameter, Volume, Curvature.
- Generates visual reports (`_report.png`).

## Installation

This project uses `uv` for dependency management.

```bash
# Install dependencies
uv sync
```

Alternatively, if you are using `pip`, it is recommended to create a virtual environment first:

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install dependencies
pip install nibabel numpy matplotlib networkx scikit-fmm scikit-image scipy
```

## Usage

1. Place your NIfTI segmentation files (`.nii`) in the `data/` directory.
2. Run the main script:

```bash
uv run main.py
```

3. The script will iterate through files in `data/`. For each file without an existing report:
   - An interactive plot will open.
   - **Left Click** to select points along the vein path.
   - **Right Click** to remove the last point.
   - Press **Enter** to finish selection.

4. The script will generate a report image (e.g., `seg_report.png`) in the `data/` directory.

## Segmentation steps with 3d slicer
1. Open image
2. Add point list - rename to F
3. Under "control points" expand "advanced options"
4. Click twice add new control point
5. Under the new two points click on the first one - the right most dot and place the point
6. Place second point
7. Run "scaling.py" script in 3d slicer python console - make sure the length of image label is the same as the one in the script - 10.0 mm
8. Click on segmentations and add new segment, draw segmentation
9. Oversample the segment by clicking right of the segment volume button, selecting volume and then factor 4
10. Export to .nii file
11. Close scene and save it to a medical bundle
