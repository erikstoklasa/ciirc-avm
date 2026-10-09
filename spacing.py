"""Physical pixel spacing (mm/px) for a DSA image, measured by hand.

Length/diameter/volume metrics in ``main.py`` need to know how many millimetres a
pixel covers. ``spacing_from_measurement`` turns a hand-clicked scale-bar length
(in pixels) plus its typed physical length (e.g. "10 mm", "1 cm") into
``mm_per_px = physical_mm / length_px``. The resulting spacing is sanity-gated to
a plausible range so a typo is flagged rather than silently trusted.

Public API:
    spacing_from_measurement(length_px, text) -> Spacing | None
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Plausible bounds so a mistyped value is flagged, not silently trusted.
PLAUSIBLE_MM_PER_PX = (0.005, 0.4)
PLAUSIBLE_VALUE_MM = (0.5, 100.0)

# A physical length typed by the user: "10.1 mm", "1.0 cm", "7mm".
_LEN_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(mm|cm)\b", re.IGNORECASE)
# Lenient form: number + a short letter run (e.g. a typo'd unit like "7m").
_LEN_RE_LENIENT = re.compile(r"(\d+(?:[.,]\d+)?)\s*([a-zA-Z]{1,4})")


@dataclass
class Spacing:
    """A spacing estimate plus the evidence behind it (so it can be sanity-checked)."""

    mm_per_px: float
    value_mm: float  # physical length typed by the user, in mm
    length_px: float  # measured bar length, in pixels
    unit: str  # "mm" or "cm" as typed
    method: str  # "manual"
    confidence: float  # rough 0..1 self-assessment
    note: str = ""  # human-readable detail
    # The two image points whose distance is ``length_px``, in (row, col) =
    # (y, x) native-pixel order. Kept so the measured span can be drawn back onto
    # the image (in napari) to verify the click landed on the real legend.
    endpoints: tuple[tuple[float, float], tuple[float, float]] | None = None

    def __str__(self) -> str:
        return (
            f"{self.mm_per_px:.4f} mm/px  "
            f"({self.value_mm:g} mm over {self.length_px:.1f} px, "
            f"method={self.method}, conf={self.confidence:.2f})"
            + (f"  [{self.note}]" if self.note else "")
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

    With ``lenient``, accept a number followed by any short letter run: the unit
    must contain an 'm' (both mm and cm end in m); a 'c' anywhere makes it cm,
    otherwise mm. Tolerates a typo'd unit in free-form user input.
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
