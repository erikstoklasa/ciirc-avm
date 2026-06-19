"""Estimate physical pixel spacing (mm/px) from a DSA image's scale legend.

Length/diameter/volume metrics in ``main.py`` need to know how many millimetres a
pixel covers. That number is usually printed on the image as a scale legend. This
module finds the legend, measures its length in pixels, OCR-reads its physical
length, and returns ``mm_per_px = physical_mm / length_px``.

Three legend styles seen across the datasets are supported:

* ``green`` -- the ``{n}m.jpg`` cases: a green line with diamond endpoints and a
  green ``X.X mm`` / ``X cm`` label (the same green as the panel seam). The bar
  length is measured diamond-centre to diamond-centre. *Most reliable.*
* ``ruler`` -- the JAP cases: a grayscale tick ruler along the bottom edge
  labelled ``cm``. Spacing = one tick interval = 1 cm.
* ``bar`` -- TJU and grayscale diamond/dashed bars labelled ``X mm`` / ``X cm``.
  The bar is found near its OCR-located label and measured end to end.

The printed number is read by a vision model served by a local Ollama (default
``glm-ocr:q8_0``, the 8B quantized GLM-OCR). It is far more accurate on these
tiny antialiased labels than a generic OCR engine, but requires ``ollama serve``
to be running with the model pulled. The resulting spacing is sanity-gated to a
plausible range so a misread number is rejected rather than silently trusted.

Public API:
    estimate_spacing(image_path, method, ...) -> Spacing | None
    method is one of "green" / "ruler" / "bar"

The detector is chosen by the caller, not guessed: each dataset has one legend
style, so its loader (``datasets.py``) passes the matching ``method`` directly.

CLI:
    uv run python spacing.py pre-data/12m.jpg --method green
    uv run python spacing.py pre-data/JAP/JAP1.jpg --method ruler --debug
    uv run python spacing.py pre-data/TJU/TJU1.png --method bar --debug
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
from PIL import Image

# Defaults for the Ollama OCR backend (the only backend).
DEFAULT_OLLAMA_MODEL = "glm-ocr:q8_0"  # 8B quantized GLM-OCR
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
_OCR_PROMPT = (
    "Read the text in this image exactly as printed. It is a scale-bar label "
    "such as '10.0 mm' or '1 cm'. Output only the text, nothing else."
)

# Plausible bounds so a misread label is rejected, not trusted.
PLAUSIBLE_MM_PER_PX = (0.005, 0.4)
PLAUSIBLE_VALUE_MM = (0.5, 100.0)

# A physical length printed on the legend: "10.1 mm", "1.0 cm", "7mm".
_LEN_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(mm|cm)\b", re.IGNORECASE)
# Lenient form: number + a short letter run the OCR mangled (e.g. "mim", "enn"
# for mm, "7m" for 7mm). Used only on label-only crops, never full pages.
_LEN_RE_LENIENT = re.compile(r"(\d+(?:[.,]\d+)?)\s*([a-zA-Z]{1,4})")


@dataclass
class Spacing:
    """A spacing estimate plus the evidence behind it (so it can be sanity-checked)."""

    mm_per_px: float
    value_mm: float  # physical length read from the legend, in mm
    length_px: float  # measured legend length, in pixels
    unit: str  # "mm" or "cm" as printed
    method: str  # "green" | "ruler" | "bar"
    confidence: float  # rough 0..1 self-assessment
    note: str = ""  # human-readable detail / OCR text

    def __str__(self) -> str:
        return (
            f"{self.mm_per_px:.4f} mm/px  "
            f"({self.value_mm:g} mm over {self.length_px:.1f} px, "
            f"method={self.method}, conf={self.confidence:.2f})"
            + (f"  [{self.note}]" if self.note else "")
        )


# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------
def _ocr_ollama(
    img: np.ndarray,
    model: str = DEFAULT_OLLAMA_MODEL,
    host: str = DEFAULT_OLLAMA_HOST,
    upscale: int = 4,
    timeout: float = 300.0,
    keep_alive: str = "10m",
) -> str:
    """OCR a crop with a local Ollama vision model (e.g. glm-ocr).

    ``timeout`` must cover the first call's model load (cold start can take
    minutes on CPU). ``keep_alive`` keeps the model resident so later calls in a
    batch are fast.
    """
    pil = Image.fromarray(img)
    if upscale > 1:
        pil = pil.resize((pil.width * upscale, pil.height * upscale), Image.LANCZOS)
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    payload = {
        "model": model,
        "prompt": _OCR_PROMPT,
        "images": [b64],
        "stream": False,
        "keep_alive": keep_alive,
        "options": {"temperature": 0},
    }
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())["response"].strip()
    except (urllib.error.URLError, OSError) as e:
        raise SystemExit(
            f"Ollama OCR failed ({e}). Is `ollama serve` running and '{model}' pulled?"
        ) from e


def _make_ocr(model: str, host: str, timeout: float = 300.0):
    """Return an ``ocr(img, upscale=) -> str`` callable backed by Ollama."""
    return lambda img, upscale=4: _ocr_ollama(
        img, model=model, host=host, upscale=upscale, timeout=timeout
    )


def _plausible(spacing_value: Spacing) -> bool:
    """True if the estimate falls in physically sensible ranges."""
    lo, hi = PLAUSIBLE_MM_PER_PX
    vlo, vhi = PLAUSIBLE_VALUE_MM
    return (lo <= spacing_value.mm_per_px <= hi) and (
        vlo <= spacing_value.value_mm <= vhi
    )


def _parse_length(text: str, lenient: bool = False) -> tuple[float, str, str] | None:
    """Return (value_mm, unit, raw_match) for the first 'N mm'/'N cm' in ``text``.

    With ``lenient`` (for label-only crops where OCR mangles the unit), accept a
    number followed by any short letter run: the unit must contain an 'm'
    (both mm and cm end in m); a 'c' anywhere makes it cm, otherwise mm.
    """
    m = _LEN_RE.search(text)
    if m:
        value, unit = float(m.group(1).replace(",", ".")), m.group(2).lower()
        return value * (10.0 if unit == "cm" else 1.0), unit, m.group(0)
    if not lenient:
        return None
    for m in _LEN_RE_LENIENT.finditer(text):
        letters = m.group(2).lower()
        if "m" not in letters:  # not a length unit; skip
            continue
        unit = "cm" if "c" in letters else "mm"
        value = float(m.group(1).replace(",", "."))
        return value * (10.0 if unit == "cm" else 1.0), unit, m.group(0)
    return None


# ---------------------------------------------------------------------------
# Style 1: green diamond bar  ({n}m.jpg)
# ---------------------------------------------------------------------------
def _green_mask(rgb: np.ndarray) -> np.ndarray:
    """Boolean mask of the green annotation overlay (line, diamonds, text)."""
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    return (g - np.maximum(r, b) > 40) & (g > 80)


def _bar_length_px(xs: np.ndarray, ys: np.ndarray) -> float:
    """Diamond-centre to diamond-centre length of one bar component.

    The component is a thin horizontal line with a taller diamond at each end.
    Column pixel-counts peak at the two diamonds; the distance between those two
    peaks (not the outer bounding box) is the measured length.
    """
    x0, x1 = int(xs.min()), int(xs.max())
    heights = np.array([(xs == x).sum() for x in range(x0, x1 + 1)])
    tall = heights >= 0.6 * heights.max()
    cols = np.arange(x0, x1 + 1)
    mid = (x0 + x1) / 2
    left = cols[tall & (cols < mid)]
    right = cols[tall & (cols >= mid)]
    if len(left) == 0 or len(right) == 0:
        return float(x1 - x0)  # fall back to outer width
    return float(right.mean() - left.mean())


def _label_crop(
    rgb: np.ndarray, y0: int, y1: int, x0: int, x1: int
) -> np.ndarray | None:
    """Black-text-on-white crop of a green label, from the green channel + Otsu.

    Using the antialiased green-channel intensity (not the hard colour mask)
    preserves thin features like the decimal point, which OCR otherwise drops.
    """
    from skimage.filters import threshold_otsu

    g = rgb[y0:y1, x0:x1, 1].astype(np.float32)
    if g.size == 0 or np.ptp(g) < 1e-3:
        return None
    norm = (255 * (g - g.min()) / (np.ptp(g) + 1e-6)).astype(np.uint8)
    try:
        th = threshold_otsu(norm)
    except ValueError:
        return None
    return np.where(norm > th, np.uint8(0), np.uint8(255))  # bright green -> black text


def detect_green(rgb: np.ndarray, ocr, debug: dict | None = None) -> Spacing | None:
    """Detect the green diamond scale bar and read its label via ``ocr``."""
    mask = _green_mask(rgb)
    if mask.sum() < 50:
        return None
    lbl, n = ndi.label(mask)

    best: Spacing | None = None
    for i in range(1, n + 1):
        ys, xs = np.where(lbl == i)
        w, h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        if w < 40 or w < 3 * h:  # keep only long, thin, horizontal bars
            continue
        length_px = _bar_length_px(xs, ys)

        # OCR the label sitting just to the right of the bar.
        cy = int(ys.mean())
        pad = max(h, 8)
        y0, y1 = max(0, cy - 2 * pad), min(mask.shape[0], cy + 2 * pad)
        x_text0 = xs.max() + 2
        x_text1 = min(mask.shape[1], xs.max() + 2 + 11 * h)
        canvas = _label_crop(rgb, y0, y1, x_text0, x_text1)
        if canvas is None:
            continue
        parsed = _parse_length(ocr(canvas, upscale=6), lenient=True)
        if parsed is None:
            continue
        value_mm, unit, raw = parsed

        if debug is not None:
            debug.setdefault("green_bars", []).append(
                (int(xs.min()), int(xs.max()), cy, length_px, raw)
            )
        cand = Spacing(
            mm_per_px=value_mm / length_px,
            value_mm=value_mm,
            length_px=length_px,
            unit=unit,
            method="green",
            confidence=0.9,
            note=f"OCR '{raw}'",
        )
        if not _plausible(cand):
            continue
        # Prefer the longest bar (least relative measurement error).
        if best is None or length_px > best.length_px:
            best = cand
    return best


# ---------------------------------------------------------------------------
# Style 2: bottom tick ruler  (JAP)
# ---------------------------------------------------------------------------
def detect_ruler(gray: np.ndarray, ocr, debug: dict | None = None) -> Spacing | None:
    """Detect a grayscale tick ruler along the bottom and measure one tick interval.

    Assumes a centimetre ruler (the JAP legend prints 'cm'): one tick interval is
    1 cm = 10 mm. The interval is the robust median spacing of evenly placed ticks.
    """
    h, w = gray.shape
    strip = gray[int(h * 0.90) :, :].astype(np.float32)

    # Baseline = the row with the most bright pixels (the long horizontal rule).
    bright = (strip > 140).sum(axis=1)
    base = int(np.argmax(bright))
    if bright[base] < 0.2 * w:
        return None

    # Ticks are short vertical marks just above the baseline.
    band = strip[max(0, base - 10) : max(1, base - 1), :]
    col = (band > 130).mean(axis=0)
    on = col >= 0.5
    # Cluster contiguous "on" columns into tick centres.
    ticks: list[int] = []
    x = 0
    while x < w:
        if on[x]:
            x0 = x
            while x < w and on[x]:
                x += 1
            ticks.append((x0 + x - 1) // 2)
        else:
            x += 1
    if len(ticks) < 3:
        return None

    # Robust tick interval: median of gaps, keeping only gaps near that median
    # (drops the big jump between the two side-by-side panel rulers and noise).
    gaps = np.diff(ticks)
    med = float(np.median(gaps))
    good = gaps[(gaps > 0.5 * med) & (gaps < 1.5 * med)]
    if len(good) == 0:
        return None
    interval = float(np.median(good))

    # Confirm the unit by OCR-ing the bottom strip ('cm' expected).
    unit_mm, unit = 10.0, "cm"
    parsed = _parse_length(ocr(strip.astype(np.uint8), upscale=3))
    if parsed is not None:
        unit_mm, unit = (10.0, "cm") if parsed[1] == "cm" else (1.0, "mm")

    if debug is not None:
        debug["ruler"] = (int(h * 0.90) + base, ticks, interval)
    return Spacing(
        mm_per_px=unit_mm / interval,
        value_mm=unit_mm,
        length_px=interval,
        unit=unit,
        method="ruler",
        confidence=0.6,
        note=f"{len(good) + 1} ticks, 1 tick = 1 {unit}",
    )


# ---------------------------------------------------------------------------
# Style 3: generic labelled bar  (TJU, grayscale diamond/dashed bars)
# ---------------------------------------------------------------------------
# detect_bar scans a coarse grid of overlapping tiles with the vision OCR. The
# tile whose text contains an "N mm"/"N cm" token localizes the legend (this
# replaces tesseract's bounding boxes), and the bar is measured inside that
# small tile -- which also bounds the span so a stray full-width structure can't
# masquerade as the bar.
BAR_TILE_GRID = (2, 3)  # (rows, cols) of tiles scanned for the label
BAR_TILE_OVERLAP = 0.30  # fractional overlap so a label/bar isn't split


def _bar_tiles(h: int, w: int) -> list[tuple[int, int, int, int]]:
    """Overlapping (y0, y1, x0, x1) tiles tiling an (h, w) image."""
    rows, cols = BAR_TILE_GRID
    th, tw = h / rows, w / cols
    oy, ox = th * BAR_TILE_OVERLAP, tw * BAR_TILE_OVERLAP
    tiles = []
    for r in range(rows):
        for c in range(cols):
            y0, y1 = max(0, int(r * th - oy)), min(h, int((r + 1) * th + oy))
            x0, x1 = max(0, int(c * tw - ox)), min(w, int((c + 1) * tw + ox))
            tiles.append((y0, y1, x0, x1))
    return tiles


BAR_LABEL_STRIPS = 4  # horizontal strips a matched tile is split into
BAR_BAND_PAD_FRAC = 0.6  # strip-heights of padding around the label strip


def _label_band(tile: np.ndarray, ocr) -> tuple[int, int]:
    """Row range of ``tile`` holding the length label (so the bar search is local).

    The matched tile still spans a lot of anatomy; the scale bar sits right by
    its 'N mm' label, not in the busiest vessel row. Split the tile into
    horizontal strips, OCR each, and return a padded band around the first strip
    whose text parses as a length. Falls back to the whole tile if none parse.
    """
    th = tile.shape[0]
    sh = th / BAR_LABEL_STRIPS
    pad = int(sh * BAR_BAND_PAD_FRAC)
    for s in range(BAR_LABEL_STRIPS):
        a, b = int(s * sh), int((s + 1) * sh)
        if _parse_length(ocr(tile[a:b], upscale=4), lenient=True):
            return max(0, a - pad), min(th, b + pad)
    return 0, th


def _widest_horizontal_span(region: np.ndarray) -> tuple[int, np.ndarray] | None:
    """Row with the widest run of pixels deviating from its median; (row, xs).

    The bar is a horizontal structure, so the row holding it has the widest
    spread of stand-out (bright or dark) pixels. ``xs`` are that row's deviating
    columns; ``xs.max() - xs.min()`` is the end-to-end length in pixels.
    """
    best_row, best_xs = None, None
    for ry in range(region.shape[0]):
        row = region[ry]
        dev = np.abs(row - np.median(row))
        on = np.where(dev > max(20.0, 0.5 * dev.max()))[0]
        if len(on) < 5:
            continue
        span = on.max() - on.min()
        if best_xs is None or span > (best_xs[-1] - best_xs[0]):
            best_row, best_xs = ry, on
    if best_xs is None:
        return None
    return best_row, best_xs


def detect_bar(gray: np.ndarray, ocr, debug: dict | None = None) -> Spacing | None:
    """Find a 'X mm'/'X cm' label by tile-scanning OCR, then measure its bar.

    Works on faint grayscale bars (dashed or solid, with small endpoint markers).
    Each tile is OCR-read; the first tile yielding a length token localizes the
    legend, and the bar is the widest horizontal run of stand-out pixels within
    that tile. Distractor text without a unit (e.g. "20.0 pixels") is rejected by
    the length parser. The longest plausible bar across tiles wins (least
    relative measurement error).
    """
    h, w = gray.shape
    best: Spacing | None = None
    for y0, y1, x0, x1 in _bar_tiles(h, w):
        tile = gray[y0:y1, x0:x1]
        parsed = _parse_length(ocr(tile, upscale=3), lenient=True)
        if parsed is None:
            continue
        value_mm, unit, raw = parsed

        # Narrow to the label's row band so the bar (not anatomy) is measured.
        by0, by1 = _label_band(tile, ocr)
        span = _widest_horizontal_span(tile[by0:by1].astype(np.float32))
        if span is None:
            continue
        ry, xs = span
        length_px = float(xs.max() - xs.min())
        if length_px < 10:
            continue

        cand = Spacing(
            mm_per_px=value_mm / length_px,
            value_mm=value_mm,
            length_px=length_px,
            unit=unit,
            method="bar",
            confidence=0.5,
            note=f"OCR '{raw}'",
        )
        if not _plausible(cand):
            continue
        if best is None or length_px > best.length_px:
            best = cand
            if debug is not None:
                debug["bar"] = (
                    y0 + by0 + ry,
                    x0 + int(xs.min()),
                    x0 + int(xs.max()),
                    raw,
                )
    return best


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
METHODS = ("green", "ruler", "bar")


def estimate_spacing(
    image_path: str | Path,
    method: str,
    ocr_model: str = DEFAULT_OLLAMA_MODEL,
    ocr_host: str = DEFAULT_OLLAMA_HOST,
    ocr_timeout: float = 300.0,
    debug_dir: str | Path | None = None,
) -> Spacing | None:
    """Estimate mm/px from an image's scale legend using one specific detector.

    ``method`` names the legend style of the dataset the image comes from:
    "green" (diamond bar), "ruler" (cm tick ruler) or "bar" (generic labelled
    bar). There is no auto-detection — each dataset loader already knows its own
    style, so exactly one detector runs. The printed number is read by the Ollama
    vision model ``ocr_model`` on ``ocr_host``. Returns None if the legend is not
    found or the reading is implausible.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}.")

    image_path = Path(image_path)
    debug: dict | None = {} if debug_dir is not None else None
    ocr_fn = _make_ocr(ocr_model, ocr_host, timeout=ocr_timeout)

    if method == "green":
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        result = detect_green(rgb, ocr_fn, debug)
    else:  # "ruler" / "bar" work on grayscale
        gray = np.asarray(Image.open(image_path).convert("L"))
        detect = detect_ruler if method == "ruler" else detect_bar
        result = detect(gray, ocr_fn, debug)

    if debug_dir is not None:
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        _save_debug(rgb, debug or {}, result, Path(debug_dir), image_path.stem)
    return result


# ---------------------------------------------------------------------------
# Manual fallback: measure the scale bar by hand when OCR fails
# ---------------------------------------------------------------------------
def spacing_from_measurement(length_px: float, text: str) -> Spacing | None:
    """Build a manual ``Spacing`` from a measured bar length and a typed length.

    ``length_px`` is the bar's pixel length; ``text`` its physical length, free-
    form ("10 mm", "1 cm", or a bare number taken as mm). Returns None if the bar
    is degenerate or no length can be read. An implausible result is warned about
    but still returned (a hand measurement is user-asserted ground truth).
    """
    if length_px < 1.0:
        return None
    text = text.strip()
    if not text:
        return None
    parsed = _parse_length(text) or _parse_length(text, lenient=True)
    if parsed is not None:
        value_mm, unit, _ = parsed
    else:  # accept a bare number as millimetres
        try:
            value_mm, unit = float(text.replace(",", ".")), "mm"
        except ValueError:
            return None
    sp = Spacing(
        mm_per_px=value_mm / length_px,
        value_mm=value_mm,
        length_px=length_px,
        unit=unit,
        method="manual",
        confidence=1.0,
        note=f"manual {value_mm:g} mm / {length_px:.1f} px",
    )
    if not _plausible(sp):
        print(
            f"  Note: {sp.mm_per_px:.4f} mm/px is outside the usual range — "
            "using your value anyway."
        )
    return sp


def pick_spacing_interactive(image_path: str | Path, prompt=input) -> Spacing | None:
    """Let the user measure the scale bar by hand and return a ``Spacing``.

    Opens the image; the user clicks the two ends of the printed scale bar. Either
    endpoint can be nudged afterwards by clicking near it, and the matplotlib
    toolbar's zoom/pan give pixel-precise placement. The live pixel length is shown
    in the title. On Enter the window closes and the user types the bar's physical
    length ("10 mm", "1 cm"); ``mm_per_px = physical_mm / pixel_length``. Returns
    None if cancelled (fewer than two points, or no length entered).
    """
    import matplotlib.pyplot as plt

    img = np.asarray(Image.open(image_path).convert("RGB"))
    pts: list[list[float]] = []

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.imshow(img)
    (line,) = ax.plot([], [], "-o", color="yellow", lw=1.5, ms=7, mfc="red")
    ax.axis("off")

    def _length() -> float:
        if len(pts) < 2:
            return 0.0
        (x0, y0), (x1, y1) = pts[0], pts[1]
        return float(np.hypot(x1 - x0, y1 - y0))

    def _redraw() -> None:
        line.set_data([p[0] for p in pts], [p[1] for p in pts])
        if len(pts) < 2:
            tip = "Click the TWO ends of the scale bar."
        else:
            tip = (
                f"Length = {_length():.1f} px.  Click near an end to nudge it; "
                "ENTER to accept."
            )
        ax.set_title(tip + "   (use the toolbar to zoom/pan for precision)")
        fig.canvas.draw_idle()

    def _on_click(event) -> None:
        if event.inaxes != ax or event.xdata is None:
            return
        tb = getattr(fig.canvas, "toolbar", None)
        if tb is not None and getattr(tb, "mode", ""):
            return  # a zoom/pan tool is active — don't drop a measurement point
        p = [float(event.xdata), float(event.ydata)]
        if len(pts) < 2:
            pts.append(p)
        else:  # move whichever endpoint is closer, so the user can refine
            d0 = np.hypot(p[0] - pts[0][0], p[1] - pts[0][1])
            d1 = np.hypot(p[0] - pts[1][0], p[1] - pts[1][1])
            pts[0 if d0 <= d1 else 1] = p
        _redraw()

    def _on_key(event) -> None:
        if event.key in ("enter", "return"):
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", _on_click)
    fig.canvas.mpl_connect("key_press_event", _on_key)
    _redraw()
    print("Manual scale: click the two ends of the scale bar, then press Enter.")
    plt.show(block=True)

    length_px = _length()
    if length_px < 1.0:
        print("  Manual scale cancelled (need two points).")
        return None

    for _ in range(3):
        text = prompt(
            f"  Measured {length_px:.1f} px. Physical length (e.g. '10 mm', '1 cm'): "
        ).strip()
        if not text:
            print("  Manual scale cancelled.")
            return None
        sp = spacing_from_measurement(length_px, text)
        if sp is not None:
            return sp
        print("  Could not read a length; try like '10 mm' or '1 cm'.")
    return None


def _save_debug(rgb, debug, result, out_dir: Path, stem: str) -> None:
    """Write an annotated PNG showing what was detected."""
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(rgb)
    for x0, x1, cy, length_px, raw in debug.get("green_bars", []):
        ax.plot([x0, x1], [cy, cy], "-", color="yellow", lw=2)
        ax.text(x1 + 5, cy, f"{raw} / {length_px:.0f}px", color="yellow", fontsize=9)
    if "ruler" in debug:
        row, ticks, interval = debug["ruler"]
        ax.plot([min(ticks), max(ticks)], [row, row], "-", color="cyan", lw=2)
        for t in ticks:
            ax.plot([t, t], [row - 8, row + 8], "-", color="cyan", lw=1)
        ax.text(
            min(ticks), row - 14, f"tick={interval:.1f}px", color="cyan", fontsize=10
        )
    if "bar" in debug:
        row, x0, x1, raw = debug["bar"]
        ax.plot([x0, x1], [row, row], "-", color="magenta", lw=2)
        ax.text(x0, row - 10, f"{raw} / {x1 - x0}px", color="magenta", fontsize=10)
    ax.set_title(str(result) if result else "no spacing detected")
    ax.axis("off")
    png = out_dir / f"{stem}_spacing.png"
    fig.savefig(png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  debug overlay -> {png}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="DSA image with a scale legend")
    parser.add_argument(
        "--method",
        required=True,
        choices=list(METHODS),
        help="detector matching the legend style of the image's dataset",
    )
    parser.add_argument(
        "--ocr-model",
        default=DEFAULT_OLLAMA_MODEL,
        help=f"Ollama vision model (default: {DEFAULT_OLLAMA_MODEL})",
    )
    parser.add_argument(
        "--ocr-host", default=DEFAULT_OLLAMA_HOST, help="Ollama base URL"
    )
    parser.add_argument(
        "--ocr-timeout",
        type=float,
        default=300.0,
        help="seconds to wait per Ollama call (cover cold-start model load)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="write an annotated overlay next to the image (in ./benchmark/spacing/)",
    )
    args = parser.parse_args()

    debug_dir = Path("benchmark") / "spacing" if args.debug else None
    result = estimate_spacing(
        args.image,
        method=args.method,
        ocr_model=args.ocr_model,
        ocr_host=args.ocr_host,
        ocr_timeout=args.ocr_timeout,
        debug_dir=debug_dir,
    )
    if result is None:
        raise SystemExit(
            f"No scale legend found in {args.image} (try --method / --debug)."
        )
    print(result)


if __name__ == "__main__":
    main()
