"""Polygons with any number of drawn sides: sub-pixel corners, and the geometry of a cyclic ring.

Built for compass-and-straightedge drawings (hand-drafted spiral constructions), where the outer figure is a 12-gon whose
sides are all drawn but whose "every second corner" hexagons are not. The snap is refine.refine_hexagon
generalised to n sides, with two additions that drawings need:

* a narrow default search (5 px). Construction drawings have chords running into every corner, and a wide search
  lets the snap jump onto a chord. The price is that rough corners must be marked to within about 3 px (mark them
  on a zoomed, gridded view);
* `occluded`: for a side partly hidden (a video play button, a ruler, a hand), only the listed fractions of the
  side are sampled, e.g. {3: [(0.0, 0.2), (0.75, 1.0)]} uses just the two visible ends of side 4.

`cyclic_ring` then describes the corners: a circle fit, side lengths, the angular step between neighbouring
corners and, for an even n, the two alternate-corner sub-polygons (for a 12-gon, hexagons A and B) and the angle
between them. A regular 12-gon steps 30 / 30 degrees; the drawings tested step about 41 / 19.

`step_candidates` compares a measured step (with its standard error) against named constructible angles.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import refine

DEFAULT_SEARCH_PX = 5.0

# Angles (degrees) a compass-and-straightedge construction reaches easily, for step_candidates.
CONSTRUCTIBLE = {
    "30 (regular 12-gon)": 30.0,
    "arctan(sqrt3/2)": float(np.degrees(np.arctan(np.sqrt(3) / 2))),     # 40.893
    "arccos(3/4)": float(np.degrees(np.arccos(0.75))),                   # 41.410
    "arctan(1/2)*2": float(np.degrees(2 * np.arctan(0.5))),              # 53.130
    "45": 45.0,
    "36 (pentagon)": 36.0,
    "40 (nonagon)": 40.0,
    "arcsin(2/3)": float(np.degrees(np.arcsin(2 / 3))),                  # 41.810
    "arctan(7/8)": float(np.degrees(np.arctan(7 / 8))),                  # 41.186
}


@dataclass
class SnappedPolygon:
    points: np.ndarray        # (n, 2) corners, same order as the rough input
    corner_sigma: np.ndarray  # (n,) bootstrap px
    side_rms: np.ndarray      # (n,) px, side i joins corners i and i+1
    inlier_frac: np.ndarray   # (n,)
    shift: np.ndarray         # (n,) px moved from the rough input
    wall_width_px: float = 0.0
    suspect: list[int] = field(default_factory=list)   # corners that probably snapped onto another line


def snap_polygon(image: np.ndarray, rough, kind: str = "dark_ridge", scale: float = 1.5,
                 search: float = DEFAULT_SEARCH_PX, occluded: dict[int, list[tuple[float, float]]] | None = None,
                 bootstrap: int = 200, seed: int = 0, suspect_shift_px: float = 8.0,
                 suspect_sigma_px: float = 2.5) -> SnappedPolygon:
    """Snap n roughly marked corners (in order around the polygon) onto the drawn sides."""
    rough = np.asarray(rough, float)
    n = len(rough)
    if n < 3:
        raise ValueError("need at least 3 corners")
    occluded = occluded or {}
    fmap = refine.feature_map(image, kind, scale)
    found, lines, rms, frac, widths = [], [], np.zeros(n), np.zeros(n), []
    for i in range(n):
        a, b = rough[i], rough[(i + 1) % n]
        pts, w = [], []
        for t0, t1 in occluded.get(i, [(0.25, 0.75)]):
            p, ww = refine._side_points(fmap, a + t0 * (b - a), a + t1 * (b - a), search, stations=20, span=0.9)
            if len(p):
                pts.append(p)
                w.append(ww)
        if not pts or sum(len(p) for p in pts) < 6:
            raise ValueError(f"side {i + 1}: line not found; mark the corners more closely")
        pts, w = np.concatenate(pts), np.concatenate(w)
        c, d, r, keep = refine._robust_line(pts)
        found.append(pts[keep])
        widths.append(w[keep])
        lines.append((c, d))
        rms[i] = float(np.sqrt(np.mean(r ** 2)))
        frac[i] = float(keep.mean())
    corners = np.array([refine._intersect(*lines[(i - 1) % n], *lines[i]) for i in range(n)])
    rng = np.random.default_rng(seed)
    boots = np.zeros((bootstrap, n, 2))
    for k in range(bootstrap):
        bl = [refine._fit_line(p[rng.integers(0, len(p), len(p))])[:2] for p in found]
        boots[k] = [refine._intersect(*bl[(i - 1) % n], *bl[i]) for i in range(n)]
    sigma = np.sqrt(np.mean(np.sum((boots - boots.mean(axis=0)) ** 2, axis=2), axis=0))
    shift = np.hypot(*(corners - rough).T)
    suspect = [int(i) for i in np.nonzero((shift > suspect_shift_px) | (sigma > suspect_sigma_px))[0]]
    return SnappedPolygon(points=corners, corner_sigma=sigma, side_rms=rms, inlier_frac=frac, shift=shift,
                          wall_width_px=float(np.median(np.concatenate(widths))), suspect=suspect)


def fit_circle(points, exclude=()) -> tuple[np.ndarray, float, float]:
    """Algebraic least-squares circle through the points (minus `exclude`): (centre, radius, radial RMS / radius)."""
    p = np.asarray(points, float)
    p = p[[i for i in range(len(p)) if i not in set(exclude)]]
    if len(p) < 3:
        raise ValueError("need at least 3 points for a circle")
    A = np.c_[2 * p, np.ones(len(p))]
    cx, cy, k = np.linalg.lstsq(A, (p ** 2).sum(1), rcond=None)[0]
    R = float(np.sqrt(k + cx ** 2 + cy ** 2))
    rad = np.hypot(p[:, 0] - cx, p[:, 1] - cy)
    return np.array([cx, cy]), R, float(np.std(rad) / R)


@dataclass
class Ring:
    centre: np.ndarray
    radius_px: float
    radial_rms_frac: float      # corners' spread about the circle, fraction of the radius
    sides_px: np.ndarray        # side i joins corners i and i+1
    steps_deg: np.ndarray       # angular step from corner i to i+1 about the centre
    long_step_deg: float | None = None   # even n: mean of the larger alternate steps (the A-to-B rotation)
    long_step_se_deg: float | None = None
    short_step_deg: float | None = None
    long_short_side_ratio: float | None = None
    alternate_sets: tuple[list[int], list[int]] | None = None   # corner indices of sub-polygons A and B


def cyclic_ring(points, exclude=()) -> Ring:
    """Geometry of corners that lie on one circle (order around the ring). `exclude` drops corners from the circle
    fit only (for instance ones flagged as suspect); steps and sides still use every corner."""
    p = np.asarray(points, float)
    n = len(p)
    c, R, rms = fit_circle(p, exclude)
    sides = np.hypot(*(np.roll(p, -1, axis=0) - p).T)
    ang = np.degrees(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]))
    steps = (np.roll(ang, -1) - ang) % 360
    if steps.mean() > 180:                       # corners listed the other way round
        steps = (ang - np.roll(ang, -1)) % 360
    ring = Ring(centre=c, radius_px=R, radial_rms_frac=rms, sides_px=sides, steps_deg=steps)
    if n % 2 == 0:
        even, odd = steps[0::2], steps[1::2]
        long_, short = (even, odd) if even.mean() >= odd.mean() else (odd, even)
        ring.long_step_deg = float(long_.mean())
        ring.long_step_se_deg = float(long_.std(ddof=1) / np.sqrt(len(long_))) if len(long_) > 1 else None
        ring.short_step_deg = float(short.mean())
        se, so = sides[0::2], sides[1::2]
        ring.long_short_side_ratio = float(max(se.mean(), so.mean()) / min(se.mean(), so.mean()))
        ring.alternate_sets = (list(range(0, n, 2)), list(range(1, n, 2)))
    return ring


def step_candidates(step_deg: float, se_deg: float, candidates: dict[str, float] | None = None,
                    within_se: float = 2.0) -> list[dict]:
    """Named angles within `within_se` standard errors of a measured step, closest first."""
    cands = CONSTRUCTIBLE if candidates is None else candidates
    se = max(float(se_deg), 1e-9)
    out = [{"name": k, "angle_deg": round(v, 3), "z": round((step_deg - v) / se, 2)} for k, v in cands.items()]
    return sorted([o for o in out if abs(o["z"]) <= within_se], key=lambda o: abs(o["z"]))
