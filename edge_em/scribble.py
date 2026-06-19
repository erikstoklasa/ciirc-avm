"""Weak user input: foreground/background scribbles for the EM segmentation.

The user drags strokes over the cropped DSA panel to mark a little **vein**
(foreground) and a little **background**. Those scribbles seed the per-class
Gaussian mixtures (Stage 2) and pin the random-walker regularization.

Scribbles are cached to ``data/{case}_scribbles.json`` and reused on re-run, so
the interactive step happens once per case -- the same load-or-prompt-then-save
idiom ``main.py:get_user_path`` uses for click points. The cache stores each
stroke as polyline coordinates (in cropped-panel pixels), its class, and the
brush radius, so it is human-inspectable and rasterizes deterministically.

Class convention (matches skimage.segmentation.random_walker seed labels):
    1 = vessel (foreground)
    2 = background
    0 = unlabeled
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from skimage.draw import disk, line  # noqa: E402

import config  # noqa: E402  (local module: edge_em/config.py)

FG_LABEL = 1
BG_LABEL = 2


class _ScribbleCollector:
    """Mouse-drag paint tool over a single grayscale panel.

    Left-drag paints strokes of the active class. Press ``v`` for foreground
    (vein), ``b`` for background, ``u`` (or ctrl+z) to undo the last stroke, and
    Enter/close to finish.
    """

    def __init__(self, panel: np.ndarray, brush_radius: int, reference=None):
        self.panel = panel
        self.brush_radius = brush_radius
        self.active_label = FG_LABEL
        self.strokes: list[dict] = []
        self._current: list[list[float]] | None = None
        # matplotlib artists for the in-progress stroke, and one such list per
        # completed stroke, so 'u' can remove a whole stroke's drawing on undo.
        self._current_artists: list = []
        self._stroke_artists: list[list] = []

        # Optional read-only reference panel (e.g. ground truth overlaid) shown on
        # the left, like main.py:get_user_path, so the user knows where to paint.
        if reference is not None:
            self.fig, (ax_ref, self.ax) = plt.subplots(1, 2, figsize=(20, 10))
            ax_ref.imshow(reference)
            ax_ref.set_title("Reference (where the true vein is) — paint on the RIGHT")
            ax_ref.axis("off")
            h, w = panel.shape
            ax_ref.set_xlim(0, w)
            ax_ref.set_ylim(h, 0)
        else:
            self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.ax.imshow(panel, cmap="gray")
        self._set_title()

        # Crop view to the (usually centered) vessel region with some padding.
        h, w = panel.shape
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(h, 0)

        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _color(self) -> str:
        return "lime" if self.active_label == FG_LABEL else "red"

    def _set_title(self) -> None:
        which = "FOREGROUND (vein)" if self.active_label == FG_LABEL else "BACKGROUND"
        self.ax.set_title(
            f"Painting: {which}\n"
            "LEFT-DRAG to paint. Press 'v'=foreground (vein), 'b'=background, "
            "'u'=undo. ENTER (or close window) to finish."
        )
        self.fig.canvas.draw_idle()

    def _on_key(self, event) -> None:
        if event.key == "v":
            self.active_label = FG_LABEL
            self._set_title()
        elif event.key == "b":
            self.active_label = BG_LABEL
            self._set_title()
        elif event.key in ("u", "ctrl+z"):
            self._undo()
        elif event.key in ("enter", "return"):
            plt.close(self.fig)

    def _undo(self) -> None:
        """Remove the most recent completed stroke and its drawing."""
        if not self.strokes:
            return
        self.strokes.pop()
        for artist in self._stroke_artists.pop():
            artist.remove()
        self.fig.canvas.draw_idle()

    def _on_press(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None:
            return
        if event.button != 1:  # left button only
            return
        self._current = [[float(event.xdata), float(event.ydata)]]
        self._current_artists = []
        # Mark the start so a single click (no drag) is visible and undoable.
        (dot,) = self.ax.plot(
            event.xdata,
            event.ydata,
            marker="o",
            markersize=self.brush_radius,
            markerfacecolor=self._color(),
            markeredgecolor=self._color(),
            linestyle="None",
        )
        self._current_artists.append(dot)
        self.fig.canvas.draw_idle()

    def _on_motion(self, event) -> None:
        if self._current is None or event.inaxes != self.ax or event.xdata is None:
            return
        pt = [float(event.xdata), float(event.ydata)]
        prev = self._current[-1]
        self._current.append(pt)
        (seg,) = self.ax.plot(
            [prev[0], pt[0]],
            [prev[1], pt[1]],
            color=self._color(),
            linewidth=self.brush_radius,
            solid_capstyle="round",
        )
        self._current_artists.append(seg)
        self.fig.canvas.draw_idle()

    def _on_release(self, event) -> None:
        if self._current is None:
            return
        if len(self._current) >= 1:
            self.strokes.append(
                {
                    "label": self.active_label,
                    "radius": self.brush_radius,
                    "points": self._current,
                }
            )
            self._stroke_artists.append(self._current_artists)
        self._current = None
        self._current_artists = []

    def run(self) -> list[dict]:
        print(
            "Scribble window opened. Paint a bit of vein ('v') and background "
            "('b'), 'u' to undo, then press Enter."
        )
        plt.show(block=True)
        return self.strokes


def collect_scribbles(
    panel: np.ndarray, brush_radius: int, reference=None
) -> list[dict]:
    """Open the interactive paint tool and return the collected strokes."""
    return _ScribbleCollector(panel, brush_radius, reference=reference).run()


def load_or_collect_scribbles(
    case: str,
    panel: np.ndarray,
    brush_radius: int | None = None,
    reference=None,
    cache_path: Path | None = None,
) -> list[dict]:
    """Load cached scribbles for a case, or collect them interactively and save.

    Mirrors ``main.py:get_user_path`` caching: if the cache file exists it is
    loaded; otherwise the paint tool opens and the result is saved. ``cache_path``
    overrides the default ``data/{case}_scribbles.json`` (e.g. to keep benchmark
    scribbles out of ``data/``). ``reference`` is shown read-only beside the panel.
    """
    if brush_radius is None:
        brush_radius = config.DEFAULT_BRUSH_RADIUS

    path = cache_path if cache_path is not None else config.scribbles_path(case)
    if path.exists():
        print(f"Loading saved scribbles from {path}...")
        with open(path, "r") as f:
            strokes = json.load(f)
        print(f"Loaded {len(strokes)} saved strokes.")
        return strokes

    strokes = collect_scribbles(panel, brush_radius, reference=reference)
    if not strokes:
        print("No scribbles drawn.")
        return []

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(strokes, f)
    print(f"Saved {len(strokes)} strokes to {path}")
    return strokes


def rasterize(strokes: list[dict], shape: tuple[int, int]) -> np.ndarray:
    """Rasterize strokes into an (H, W) seed-label array (0/1/2).

    For each stroke the polyline is drawn and dilated by a disk of the stroke's
    brush radius. Foreground (1) is stamped after background (2) so that, where
    they overlap, the foreground wins.
    """
    seeds = np.zeros(shape, dtype=np.uint8)
    h, w = shape

    # Stamp background first, then foreground, so FG overrides on overlap.
    for label in (BG_LABEL, FG_LABEL):
        for stroke in strokes:
            if stroke["label"] != label:
                continue
            r = int(stroke.get("radius", config.DEFAULT_BRUSH_RADIUS))
            pts = stroke["points"]
            for i in range(len(pts)):
                x, y = pts[i]
                _stamp_disk(seeds, y, x, r, label, h, w)
                if i > 0:
                    x0, y0 = pts[i - 1]
                    _stamp_line(seeds, y0, x0, y, x, r, label, h, w)
    return seeds


def _stamp_disk(seeds, cy, cx, r, label, h, w) -> None:
    rr, cc = disk((cy, cx), max(r, 1), shape=(h, w))
    seeds[rr, cc] = label


def _stamp_line(seeds, y0, x0, y1, x1, r, label, h, w) -> None:
    """Draw the segment (x0,y0)->(x1,y1) then thicken it with disks."""
    ry0, rx0 = int(round(y0)), int(round(x0))
    ry1, rx1 = int(round(y1)), int(round(x1))
    ry0 = min(max(ry0, 0), h - 1)
    rx0 = min(max(rx0, 0), w - 1)
    ry1 = min(max(ry1, 0), h - 1)
    rx1 = min(max(rx1, 0), w - 1)
    rr, cc = line(ry0, rx0, ry1, rx1)
    for yy, xx in zip(rr, cc):
        _stamp_disk(seeds, yy, xx, r, label, h, w)
