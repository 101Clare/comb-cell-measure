"""Millimetre morphometry of honeycomb cells: ruler calibration, flat-to-flat widths and size classes.

Scale. Two points clicked on a ruler and the known distance between them give
    S = known_mm / |p2 - p1|  (mm per pixel).
`ruler_tick_period` gives an independent check: the millimetre ticks along the same line form a regular
pattern whose period (from the autocorrelation of the intensity profile) is the number of pixels per mm.

Cell width. Beekeepers size cells by the distance across the cell between opposite walls, usually measured as
"N cells = L mm" along a ruler, which is the centre-to-centre spacing of the cells, i.e. from the middle of one
wall to the middle of the opposite wall. hexengine.refine fits each wall's centreline, so for a snapped cell:
    D_k = distance between the lines of side k and side k+3 (k = 0, 1, 2), measured both ways and averaged
    D   = mean of the three, anisotropy = max - min.
For a regular hexagon of circumradius R, D = sqrt(3) * R. The "inner" width (the open space, excluding the wall)
is approximately D minus the measured wall thickness; at thumbnail resolution that subtraction is crude.

Size classes (Apis mellifera, indicative ranges as supplied for this pilot; values between ranges are reported as
"between classes"):
    4.80-5.15 mm  natural / small-cell worker
    5.20-5.45 mm  standard worker
    5.50-6.15 mm  intermediate / storage
    6.20-6.60 mm  drone
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

BENCHMARKS = [  # (low mm, high mm, label)
    (4.80, 5.15, "natural / small-cell worker"),
    (5.20, 5.45, "standard worker"),
    (5.50, 6.15, "intermediate / storage"),
    (6.20, 6.60, "drone"),
]


@dataclass
class Scale:
    """Image scale from two ruler points (image px) and the known distance between them (mm)."""

    p1: tuple[float, float]
    p2: tuple[float, float]
    known_mm: float

    @property
    def length_px(self) -> float:
        return float(np.hypot(self.p2[0] - self.p1[0], self.p2[1] - self.p1[1]))

    @property
    def mm_per_px(self) -> float:
        if self.length_px <= 0 or self.known_mm <= 0:
            raise ValueError("ruler points must differ and the known distance must be positive")
        return self.known_mm / self.length_px

    @property
    def px_per_mm(self) -> float:
        return 1.0 / self.mm_per_px


def classify(width_mm: float) -> str:
    """Size class for a cell width in mm (see BENCHMARKS)."""
    for lo, hi, label in BENCHMARKS:
        if lo <= width_mm <= hi:
            return label
    if width_mm < BENCHMARKS[0][0]:
        return "below natural worker range"
    if width_mm > BENCHMARKS[-1][1]:
        return "above drone range"
    return "between classes"


def flat_to_flat_px(corners: np.ndarray) -> np.ndarray:
    """The three opposite-wall distances (px) of a hexagon given its six corners in order around it."""
    c = np.asarray(corners, float)
    out = []
    for k in range(3):
        a0, a1 = c[k], c[(k + 1) % 6]
        b0, b1 = c[(k + 3) % 6], c[(k + 4) % 6]
        def dist(p, q0, q1):
            d = q1 - q0
            return abs(d[0] * (p[1] - q0[1]) - d[1] * (p[0] - q0[0])) / np.hypot(*d)
        # midpoint of each side to the opposite side's line, averaged (exact for parallel sides)
        out.append(0.5 * (dist(0.5 * (a0 + a1), b0, b1) + dist(0.5 * (b0 + b1), a0, a1)))
    return np.array(out)


def ruler_tick_period(image: np.ndarray, p1, p2, strip_px: float = 6.0, expect_px_per_mm: float | None = None,
                      search: float = 0.35) -> float | None:
    """Pixels per mm from the millimetre ticks along the ruler line p1 -> p2 (None if no clear period).

    The intensity profile is sampled along the line (averaged across a strip either side), detrended, and
    autocorrelated; the strongest peak near the expected period (within +-search of it) is refined to sub-pixel.
    The period is measured over the whole line, so it averages out the printing and blur of single ticks.
    """
    g = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) if image.ndim == 3 else image.astype(np.float32)
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    L = float(np.hypot(*(p2 - p1)))
    u = (p2 - p1) / L
    n = np.array([-u[1], u[0]])
    ts = np.arange(0, L, 0.25)
    prof = np.zeros_like(ts)
    for off in np.linspace(-strip_px, strip_px, 9):
        pts = (p1 + ts[:, None] * u + off * n).astype(np.float32)
        prof += cv2.remap(g, pts[:, :1], pts[:, 1:], cv2.INTER_LINEAR).ravel()
    prof = prof - cv2.GaussianBlur(prof.reshape(1, -1), (0, 0), 40).ravel()
    ac = np.correlate(prof, prof, mode="full")[len(prof) - 1:]
    ac /= ac[0] + 1e-9
    lags = np.arange(len(ac)) * 0.25
    if expect_px_per_mm is None:
        return None
    lo, hi = expect_px_per_mm * (1 - search), expect_px_per_mm * (1 + search)
    sel = np.nonzero((lags >= lo) & (lags <= hi))[0]
    if len(sel) < 3:
        return None
    k = sel[np.argmax(ac[sel])]
    if ac[k] < 0.05:
        return None
    if 0 < k < len(ac) - 1:
        y0, y1, y2 = ac[k - 1], ac[k], ac[k + 1]
        den = y0 - 2 * y1 + y2
        shift = 0.5 * (y0 - y2) / den if den < 0 else 0.0
    else:
        shift = 0.0
    # refine with the 10th multiple (1 cm) if visible: dividing a longer lag by 10 cuts the error tenfold
    base = (k + shift) * 0.25
    k10 = int(round(base * 10 / 0.25))
    win = range(max(1, k10 - 8), min(len(ac) - 1, k10 + 9))
    if len(ac) > k10 + 9 and win:
        j = max(win, key=lambda i: ac[i])
        if ac[j] > 0.05:
            y0, y1, y2 = ac[j - 1], ac[j], ac[j + 1]
            den = y0 - 2 * y1 + y2
            sh = 0.5 * (y0 - y2) / den if den < 0 else 0.0
            return float((j + sh) * 0.25 / 10.0)
    return float(base)


@dataclass
class CellMeasure:
    width_mm: float          # mean flat-to-flat, wall centre to wall centre
    widths_mm: np.ndarray    # the three opposite-wall distances
    anisotropy_mm: float     # max - min of the three
    inner_mm: float          # width minus measured wall thickness (approximate)
    size_class: str


def measure_cell(corners: np.ndarray, scale: Scale, wall_width_px: float = 0.0) -> CellMeasure:
    w = flat_to_flat_px(corners) * scale.mm_per_px
    mean = float(w.mean())
    return CellMeasure(width_mm=mean, widths_mm=w, anisotropy_mm=float(w.max() - w.min()),
                       inner_mm=mean - wall_width_px * scale.mm_per_px, size_class=classify(mean))
