"""Convert the (jpg, nii) vein pairs into an nnU-Net v2 raw dataset of patches.

For every image that has BOTH a source ``pre-data/{name}.jpg`` and a label
``data/{name}.nii``, this crops the segmented (right) panel and then tiles it
into small square patches centered on the annotated veins, writing one nnU-Net
case per patch:

    nnUNet_raw/Dataset001_Veins/
        imagesTr/{case}_0000.png   grayscale patch  (PATCH_SIZE, PATCH_SIZE)
        labelsTr/{case}.png        binary mask 0/1  (PATCH_SIZE, PATCH_SIZE)
        dataset.json

Why patches: 26 whole frames make a tiny, vessel-sparse dataset. Tiling around
the labels yields hundreds of vessel-rich patches (more, better-balanced data),
and offline augmentation (slight rotations + Gaussian noise; see config.py)
multiplies and diversifies them further. nnU-Net's planner then picks a small
patch size, and at predict time it slides that window over the full image — so
predict.py needs no change.

Orientation note: main.py treats the on-disk nii as (W, H, 1) and transposes
with ``.T`` to get (H, W) for processing/display. We therefore read the label
as ``nii.squeeze().T`` so that image and mask share the same (H, W) frame as the
jpg. predict.py applies the inverse transpose when writing masks back out.

Usage:
    uv run python nnunet/prepare_dataset.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# Allow running as `uv run python nnunet/prepare_dataset.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import nibabel as nib
import numpy as np
from PIL import Image

import config  # local module (nnunet/config.py)


def find_pairs() -> list[str]:
    """Return sorted case names that have both a jpg and a nii of matching size."""
    names = []
    for nii_path in sorted(config.NII_DIR.glob("*.nii")):
        name = nii_path.stem
        jpg_path = config.JPG_DIR / f"{name}.jpg"
        if not jpg_path.exists():
            continue

        # Verify the image and mask describe the same pixel grid.
        with Image.open(jpg_path) as im:
            w, h = im.size  # PIL: (width, height)
        nii = nib.load(str(nii_path))
        nw, nh = nii.shape[0], nii.shape[1]  # on-disk (W, H, 1)
        if (nw, nh) != (w, h):
            print(f"  skip {name}: jpg {w}x{h} != nii {nw}x{nh}")
            continue
        names.append(name)
    return names


def load_image_hw(jpg_path) -> np.ndarray:
    """Load a DSA jpg as a single-channel grayscale uint8 array of shape (H, W)."""
    with Image.open(jpg_path) as im:
        gray = im.convert("L")
        return np.asarray(gray, dtype=np.uint8)  # (H, W)


def load_mask_hw(nii_path) -> np.ndarray:
    """Load a nii mask into (H, W) binary uint8, matching main.py's orientation."""
    nii = nib.load(str(nii_path))
    data = np.squeeze(np.asarray(nii.dataobj))  # (W, H)
    mask = (data > 0).T.astype(np.uint8)  # (H, W), same as main.py
    return mask


def _rotate(arr: np.ndarray, angle: float, *, nearest: bool) -> np.ndarray:
    """Rotate (H, W) array CCW by ``angle`` degrees, keeping the same shape.

    Pixels rotated in from outside the frame are filled with 0. We track that
    fill with a companion validity mask so tiling can reject patches touching it.
    """
    resample = Image.NEAREST if nearest else Image.BILINEAR
    im = Image.fromarray(arr).rotate(
        angle, resample=resample, expand=False, fillcolor=0
    )
    return np.asarray(im)


def panel_variants(
    img: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Build (suffix, img, mask, valid) variants of one panel for tiling.

    ``valid`` is a (H, W) uint8 mask of pixels that are genuine image data (1)
    rather than rotation fill (0). The base/noise variants are valid everywhere.
    """
    h, w = img.shape
    ones = np.ones((h, w), dtype=np.uint8)
    variants = [("", img, mask, ones)]

    for ang in config.AUG_ROTATIONS:
        variants.append(
            (
                f"_r{ang:+d}".replace("+", "p").replace("-", "m"),
                _rotate(img, ang, nearest=False),
                _rotate(mask, ang, nearest=True),
                _rotate(ones, ang, nearest=True),
            )
        )

    for std in config.AUG_GAUSS_NOISE_STD:
        noisy = np.clip(
            img.astype(np.float32) + rng.normal(0.0, std, img.shape), 0, 255
        ).astype(np.uint8)
        variants.append((f"_n{int(round(std))}", noisy, mask, ones))

    return variants


def _windows(h: int, w: int) -> list[tuple[int, int]]:
    """Top-left (y, x) corners of PATCH_SIZE windows covering an (H, W) frame."""
    p, s = config.PATCH_SIZE, config.PATCH_STRIDE

    def starts(extent: int) -> list[int]:
        if extent <= p:
            return [0]  # frame smaller than a patch: single (later padded) window
        pos = list(range(0, extent - p + 1, s))
        if pos[-1] != extent - p:
            pos.append(extent - p)  # ensure the far edge is covered
        return pos

    return [(y, x) for y in starts(h) for x in starts(w)]


def _crop(arr: np.ndarray, y: int, x: int) -> np.ndarray:
    """Crop a PATCH_SIZE square at (y, x), zero-padding if the frame is smaller."""
    p = config.PATCH_SIZE
    patch = arr[y : y + p, x : x + p]
    if patch.shape != (p, p):
        out = np.zeros((p, p), dtype=arr.dtype)
        out[: patch.shape[0], : patch.shape[1]] = patch
        return out
    return patch


def extract_patches(
    img: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Tile a panel into (suffix, img_patch, mask_patch) windows around veins.

    Keeps every fully-valid window with >= PATCH_MIN_FG vessel pixels across all
    augmented variants, plus PATCH_NEG_PER_CASE vessel-free windows (from the
    un-augmented panel) so the network also sees plausible negatives.
    """
    out: list[tuple[str, np.ndarray, np.ndarray]] = []

    for vsuffix, vimg, vmask, vvalid in panel_variants(img, mask, rng):
        for i, (y, x) in enumerate(_windows(*img.shape)):
            mpatch = _crop(vmask, y, x)
            if int(mpatch.sum()) < config.PATCH_MIN_FG:
                continue
            if not _crop(vvalid, y, x).all():
                continue  # window clips rotation fill — skip it
            out.append((f"{vsuffix}_p{i:03d}", _crop(vimg, y, x), mpatch))

    # A few vessel-free patches from the original panel reduce false positives.
    windows = _windows(*img.shape)
    empties = [(y, x) for (y, x) in windows if int(_crop(mask, y, x).sum()) == 0]
    rng.shuffle(empties)
    for j, (y, x) in enumerate(empties[: config.PATCH_NEG_PER_CASE]):
        out.append((f"_neg{j:02d}", _crop(img, y, x), _crop(mask, y, x)))

    return out


def main() -> None:
    config.setup_env()

    raw = config.dataset_raw_dir()
    images_tr = raw / "imagesTr"
    labels_tr = raw / "labelsTr"
    if raw.exists():
        print(f"Removing existing {raw}")
        shutil.rmtree(raw)
    images_tr.mkdir(parents=True)
    labels_tr.mkdir(parents=True)

    # The preprocessed cache (fingerprint, plans, .b2nd tensors, splits) is
    # derived from the raw cases. Once we re-tile into patches it's stale and
    # the planner/trainer would silently reuse old full-frame data, so wipe it.
    preprocessed = config.NNUNET_PREPROCESSED / config.DATASET_NAME
    if preprocessed.exists():
        print(f"Removing stale preprocessed cache {preprocessed}")
        shutil.rmtree(preprocessed)

    names = find_pairs()
    if not names:
        raise SystemExit("No (jpg, nii) pairs found — nothing to convert.")

    print(f"Found {len(names)} labeled pairs: {', '.join(names)}")
    for name in names:
        img = load_image_hw(config.JPG_DIR / f"{name}.jpg")
        mask = load_mask_hw(config.NII_DIR / f"{name}.nii")
        if img.shape != mask.shape:
            print(f"  skip {name}: shape mismatch img {img.shape} vs mask {mask.shape}")
            continue

        # Crop to the right (segmented) panel; seam from the image, applied to both.
        x0 = config.right_panel_x0(img)
        img = img[:, x0:]
        mask = mask[:, x0:]
        if mask.sum() == 0:
            print(
                f"  WARNING {name}: no vessel pixels after crop (x0={x0}) — check seam"
            )
            continue

        # Seed per case so the dataset is reproducible across runs.
        rng = np.random.default_rng(abs(hash(name)) % (2**32))
        patches = extract_patches(img, mask, rng)
        for suffix, ip, mp in patches:
            case = f"{name}{suffix}"
            Image.fromarray(ip, mode="L").save(
                images_tr / f"{case}_0000{config.FILE_ENDING}"
            )
            Image.fromarray(mp, mode="L").save(
                labels_tr / f"{case}{config.FILE_ENDING}"
            )
        print(f"  {name}: {len(patches)} patches")

    n_written = len(list(labels_tr.glob(f"*{config.FILE_ENDING}")))

    # dataset.json — uses nnU-Net v2's natural-image (.png) 2D reader/writer.
    from nnunetv2.dataset_conversion.generate_dataset_json import (
        generate_dataset_json,
    )

    generate_dataset_json(
        output_folder=str(raw),
        channel_names={0: "grayscale"},
        labels=config.LABELS,
        num_training_cases=n_written,
        file_ending=config.FILE_ENDING,
        dataset_name=config.DATASET_NAME,
        description="Spinal DSA vein segmentation (single grayscale channel).",
    )

    print(f"\nWrote {n_written} cases to {raw}")
    print("Next: uv run python nnunet/train.py")


if __name__ == "__main__":
    main()
