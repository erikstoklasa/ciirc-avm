"""Dataset loaders: a uniform way to get the scale, the labeled image, and the
raw (unlabeled) image out of each DSA source, regardless of how that source is
laid out on disk.

Three datasets feed this project and each stores the same three things
differently:

* **current** (``pre-data/{n}m.jpg``) -- one side-by-side frame. The **left**
  panel carries the clinician's red vein tracing; the **right** panel is the
  clean copy. Scale is a green diamond bar (``X mm``).
* **JAP** (``pre-data/JAP/JAP{n}.jpg``) -- same side-by-side layout: **left**
  panel has the orange tracing + yellow annotations, **right** panel is clean.
  Scale is a centimetre tick ruler along the bottom.
* **TJU** (``pre-data/TJU/TJU{n}.png``) -- the labeled and raw versions are two
  *separate* files: ``TJU{n} tracked.png`` is the traced one, ``TJU{n}.png`` is
  raw. Scale is a generic labelled bar (``X mm`` / ``X cm``).

The abstraction is a single image/case at a time:

    loader = TJUDataset("pre-data/TJU/TJU1.png")
    loader.get_scale()     # -> spacing.Spacing | None   (mm per pixel)
    loader.get_labeled()   # -> np.ndarray (H, W, 3)     traced version
    loader.get_raw()       # -> np.ndarray (H, W, 3)     clean version

Each concrete loader implements those three accessors its own way; the scale
detection itself is delegated to ``spacing.estimate_spacing`` with the detector
that matches the dataset's legend style.

CLI:
    uv run python datasets.py current pre-data/10m.jpg
    uv run python datasets.py jap pre-data/JAP/JAP1.jpg --debug
    uv run python datasets.py tju pre-data/TJU/TJU1.png
"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Iterator

import numpy as np
from PIL import Image

from spacing import Spacing, estimate_spacing

REPO_ROOT = Path(__file__).resolve().parent
PRE_DATA = REPO_ROOT / "pre-data"


# ---------------------------------------------------------------------------
# Shared image helpers
# ---------------------------------------------------------------------------
def _load_rgb(path: str | Path) -> np.ndarray:
    """Load an image file as an (H, W, 3) uint8 RGB array."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


# Panel seam detection, mirroring edge_em/config.py:right_panel_x0 (kept local so
# this module is self-contained, same convention the other modules follow). Each
# side-by-side DSA frame is split at its darkest near-centre column: the left
# part is the traced panel, the right part is the clean copy.
PANEL_SEAM_SEARCH = 70  # +/- px around centre to search for the divider column


def _panel_seam_x(gray_or_rgb: np.ndarray) -> int:
    """Column index of the divider between the left and right panels."""
    arr = gray_or_rgb
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    h, w = arr.shape[:2]
    center = w // 2
    lo = max(0, center - PANEL_SEAM_SEARCH)
    hi = min(w, center + PANEL_SEAM_SEARCH)
    col_mean = arr.astype(np.float32).mean(axis=0)
    return lo + int(np.argmin(col_mean[lo:hi]))


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------
class DatasetLoader(ABC):
    """One DSA case, with a uniform way to read its scale, label, and raw image.

    A loader wraps a single source image (``image_path``). Subclasses implement
    the three accessors below according to how their dataset stores things.

    Class attributes a subclass sets:
        name            short dataset id ("current" / "jap" / "tju")
        image_dir       default directory the dataset's images live in
        spacing_method  spacing.py detector matching the legend style
                        ("green" / "ruler" / "bar")
    """

    name: ClassVar[str]
    image_dir: ClassVar[Path]
    spacing_method: ClassVar[str]  # the one spacing.py detector this dataset uses

    def __init__(self, image_path: str | Path):
        self.image_path = Path(image_path)
        self.case = self.image_path.stem

    # -- scale: one detector per dataset, no auto-detection ---------------
    def scale_source(self) -> Path:
        """The image file the scale legend is read from (default: the source)."""
        return self.image_path

    def get_scale(self, **kwargs) -> Spacing | None:
        """Physical scale (mm per pixel), read with this dataset's detector."""
        return estimate_spacing(
            self.scale_source(), method=self.spacing_method, **kwargs
        )

    # -- the label/raw split, which each dataset stores differently -------
    @abstractmethod
    def get_labeled(self) -> np.ndarray:
        """The labeled/traced version of the image, as (H, W, 3) RGB."""

    @abstractmethod
    def get_raw(self) -> np.ndarray:
        """The raw, unlabeled version of the image, as (H, W, 3) RGB."""

    @classmethod
    def cases(cls) -> Iterator[DatasetLoader]:
        """Yield a loader for every image of this dataset on disk."""
        raise NotImplementedError(f"{cls.__name__} does not enumerate cases")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(case={self.case!r})"


# ---------------------------------------------------------------------------
# Side-by-side panel datasets (current + JAP)
# ---------------------------------------------------------------------------
class _PanelDataset(DatasetLoader):
    """Datasets where labeled and raw are the two panels of one frame.

    The frame is split at the inter-panel seam: the left panel holds the
    clinician's tracing, the right panel is the clean copy.
    """

    glob_pattern: ClassVar[str] = "*"

    def __init__(self, image_path: str | Path):
        super().__init__(image_path)
        self._rgb: np.ndarray | None = None
        self._seam: int | None = None

    def _split(self) -> tuple[np.ndarray, int]:
        """Frame + panel-seam column, decoded once and reused by both accessors."""
        if self._rgb is None:
            self._rgb = _load_rgb(self.image_path)
            self._seam = _panel_seam_x(self._rgb)
        return self._rgb, self._seam  # type: ignore[return-value]

    def get_labeled(self) -> np.ndarray:
        rgb, seam = self._split()
        return rgb[:, :seam]

    def get_raw(self) -> np.ndarray:
        rgb, seam = self._split()
        return rgb[:, seam:]

    @classmethod
    def cases(cls) -> Iterator[DatasetLoader]:
        for path in sorted(cls.image_dir.glob(cls.glob_pattern)):
            yield cls(path)


class CurrentDataset(_PanelDataset):
    """Current dataset: ``pre-data/{n}m.jpg``, green diamond scale bar."""

    name = "current"
    image_dir = PRE_DATA
    spacing_method = "green"
    glob_pattern = "*m.jpg"


class JAPDataset(_PanelDataset):
    """JAP dataset: ``pre-data/JAP/JAP{n}.jpg``, bottom centimetre tick ruler."""

    name = "jap"
    image_dir = PRE_DATA / "JAP"
    spacing_method = "ruler"
    glob_pattern = "JAP*.jpg"


# ---------------------------------------------------------------------------
# TJU: labeled and raw are separate files
# ---------------------------------------------------------------------------
class TJUDataset(DatasetLoader):
    """TJU dataset: ``pre-data/TJU/TJU{n}.png`` (raw) + ``TJU{n} tracked.png``.

    The loader can be pointed at either file; both the raw and the tracked
    versions are resolved from the case name. Scale is a generic labelled bar.
    """

    name = "tju"
    image_dir = PRE_DATA / "TJU"
    spacing_method = "bar"

    def __init__(self, image_path: str | Path):
        super().__init__(image_path)
        # Normalise the case so we can find both files whether we were handed
        # "TJU1.png" or "TJU1 tracked.png".
        self.case = self.case.removesuffix(" tracked")

    @property
    def raw_path(self) -> Path:
        return self.image_dir / f"{self.case}.png"

    @property
    def labeled_path(self) -> Path:
        return self.image_dir / f"{self.case} tracked.png"

    def scale_source(self) -> Path:
        # Read the legend off the raw image (the tracing can occlude the bar).
        return self.raw_path

    def get_labeled(self) -> np.ndarray:
        return _load_rgb(self.labeled_path)

    def get_raw(self) -> np.ndarray:
        return _load_rgb(self.raw_path)

    @classmethod
    def cases(cls) -> Iterator[DatasetLoader]:
        # One loader per raw image; skip the "* tracked" companions.
        for path in sorted(cls.image_dir.glob("TJU*.png")):
            if path.stem.endswith(" tracked"):
                continue
            yield cls(path)


# ---------------------------------------------------------------------------
# Registry / factory
# ---------------------------------------------------------------------------
DATASETS: dict[str, type[DatasetLoader]] = {
    cls.name: cls for cls in (CurrentDataset, JAPDataset, TJUDataset)
}


def load(dataset: str, image_path: str | Path) -> DatasetLoader:
    """Build the loader for ``dataset`` ("current"/"jap"/"tju") over ``image_path``."""
    try:
        return DATASETS[dataset](image_path)
    except KeyError:
        raise ValueError(
            f"Unknown dataset {dataset!r}; choose from {sorted(DATASETS)}."
        ) from None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "dataset", choices=sorted(DATASETS), help="which dataset loader"
    )
    parser.add_argument("image", help="path to a source image of that dataset")
    parser.add_argument(
        "--save",
        metavar="DIR",
        default=None,
        help="write {case}_labeled.png and {case}_raw.png to DIR",
    )
    args = parser.parse_args()

    loader = load(args.dataset, args.image)
    print(f"{loader}  ({loader.name})")

    labeled, raw = loader.get_labeled(), loader.get_raw()
    print(f"  labeled: {labeled.shape}")
    print(f"  raw:     {raw.shape}")

    scale = loader.get_scale()
    print(f"  scale:   {scale if scale is not None else 'not detected'}")

    if args.save:
        out = Path(args.save)
        out.mkdir(parents=True, exist_ok=True)
        Image.fromarray(labeled).save(out / f"{loader.case}_labeled.png")
        Image.fromarray(raw).save(out / f"{loader.case}_raw.png")
        print(f"  wrote {loader.case}_labeled.png / {loader.case}_raw.png -> {out}")


if __name__ == "__main__":
    main()
