"""Sub-pixel corner refinement by fitting a straight line along each side of a roughly marked hexagon.

Why: a click is only as good as the eye (a few pixels), and on a hexagon with a 60 px radius a 2 px click error
already adds ~0.03 to the residual, the whole "strong" threshold. A side, though, is tens of pixels long. If we
find the wall (or edge) along the whole side and fit a straight line to it, the line is located far more
precisely than any single click, and two neighbouring side lines intersect at the corner.

How, for each side V_i -> V_{i+1} of the rough hexagon:
1. take ~40 stations along the middle 70 % of the side (the ends are skipped: junctions are blobby);
2. at each station, sample a feature map along the perpendicular, within +-search px, and take the sub-pixel
   position of its maximum: the wall's centreline (ridge maps) or the boundary (edge map);
3. fit a line through those points by total least squares, drop outliers (> 3 robust sigma), refit.
Each corner is the intersection of its two side lines.

Feature maps (choose to suit the image):
  "dark_ridge"   dark line on brighter ground (basalt gaps, turtle seams, honeycomb walls in backlight)
  "bright_ridge" bright line on darker ground (soap-film walls, lit wax walls, snowflake ribs)
  "edge"         a step between two brightnesses (tile and cube-face boundaries)
`scale` is the Gaussian scale in px; set it to about half the wall width.

Quality outputs, per corner and per side:
  corner_sigma   bootstrap standard deviation of each corner position (px): the marking uncertainty
  side_rms       RMS distance of the found wall points from their fitted line (px): how straight the side is
  inlier_frac    share of stations that agreed with the line: low means the wall was not found cleanly
  wall_width_px  median full width at half maximum of the wall/edge response across the sides (px): an estimate
                 of how thick the wall is in the photo (it sets the source-pixel edge margin in inference)

Quality gate (`quality`): a refinement is trusted only if the corners are located to within 2.5 % of the hexagon's
radius, at least 75 % of the wall samples on every side agree with that side's line, and the corners moved on
average no more than 15 % of the radius from the clicks. The first two came from the baseline test, where they
separated every visibly wrong outline from the correct ones; the third stops a snap onto a neighbouring line.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

FEATURES = ("dark_ridge", "bright_ridge", "edge")
QC_SIGMA_FRAC = 0.025
QC_INLIER_MIN = 0.75
QC_SHIFT_FRAC = 0.15     # mean corner movement from the clicks, as a fraction of the radius
SHIFT_PENALTY = 0.25     # refine_auto: px of corner uncertainty one px of movement from the clicks is worth


@dataclass
class Refined:
    points: np.ndarray        # (6, 2) refined corners, same order as the input
    corner_sigma: np.ndarray  # (6,) px
    side_rms: np.ndarray      # (6,) px, side i joins corners i and i+1
    inlier_frac: np.ndarray   # (6,)
    shift: np.ndarray         # (6,) px moved from the rough input
    wall_width_px: float = 0.0
    radius_px: float = 0.0    # mean distance of the refined corners from their centroid
    kind: str = ""
    scale: float = 0.0
    qc_reasons: list[str] = field(default_factory=list)

    @property
    def qc_pass(self) -> bool:
        return not self.qc_reasons

    @property
    def sigma_px(self) -> float:
        """One number for the corner uncertainty: the median over the six corners."""
        return float(np.median(self.corner_sigma))


def quality(r: Refined, max_shift_frac: float = QC_SHIFT_FRAC) -> list[str]:
    """Reasons the refinement should not be trusted (empty list = passed).

    max_shift_frac: how far (mean, as a fraction of the radius) the corners may move from the starting points.
    The default suits clicks; scripts that start from deliberately rough corners can relax it."""
    reasons = []
    if r.radius_px <= 0 or r.sigma_px / r.radius_px > QC_SIGMA_FRAC:
        reasons.append("corners not well located (walls blurred, curved or not straight)")
    if r.inlier_frac.min() < QC_INLIER_MIN:
        reasons.append("a side's wall was not found cleanly")
    if r.radius_px > 0 and float(np.mean(r.shift)) / r.radius_px > max_shift_frac:
        reasons.append("corners moved far from the clicks (probably locked onto a different line)")
    return reasons


def _grey(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image)
    if img.ndim == 3:
        img = cv2.cvtColor(img[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return img.astype(np.float32) / (255.0 if img.dtype == np.uint8 else 1.0)


def feature_map(image: np.ndarray, kind: str, scale: float) -> np.ndarray:
    """Non-negative map that peaks on the wall centreline (ridges) or boundary (edge)."""
    if kind not in FEATURES:
        raise ValueError(f"kind must be one of {FEATURES}")
    g = cv2.GaussianBlur(_grey(image), (0, 0), scale)
    if kind == "edge":
        gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
        return cv2.magnitude(gx, gy)
    # Hessian eigenvalues: a dark line has a large positive curvature across it, a bright line a large negative one.
    hxx = cv2.Sobel(g, cv2.CV_32F, 2, 0)
    hyy = cv2.Sobel(g, cv2.CV_32F, 0, 2)
    hxy = cv2.Sobel(g, cv2.CV_32F, 1, 1)
    root = np.sqrt((hxx - hyy) ** 2 + 4 * hxy ** 2)
    lam_hi, lam_lo = 0.5 * (hxx + hyy + root), 0.5 * (hxx + hyy - root)
    return np.maximum(lam_hi, 0) if kind == "dark_ridge" else np.maximum(-lam_lo, 0)


def _sample(fmap: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Bilinear samples of fmap at (N, 2) float points (x, y); 0 outside the image."""
    mx = pts[:, 0].astype(np.float32).reshape(-1, 1)
    my = pts[:, 1].astype(np.float32).reshape(-1, 1)
    return cv2.remap(fmap, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0).ravel()


def _fit_line(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Total-least-squares line: (point, unit direction, residuals)."""
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c)
    d = vt[0]
    n = np.array([-d[1], d[0]])
    return c, d, (pts - c) @ n


def _side_points(fmap, a, b, search, stations=40, span=0.7):
    """Wall/edge positions found across the side a->b (N, 2), and the response's width at half maximum (N,)."""
    d = b - a
    length = float(np.hypot(*d))
    u = d / length
    n = np.array([-u[1], u[0]])
    ts = np.linspace(0.5 - span / 2, 0.5 + span / 2, stations)
    step = 0.5
    offs = np.arange(-search, search + step / 2, step)
    out, widths = [], []
    for t in ts:
        base = a + t * d
        prof = _sample(fmap, base + offs[:, None] * n)
        k = int(np.argmax(prof))
        if prof[k] <= 0:
            continue
        off = offs[k]
        if 0 < k < len(offs) - 1:  # parabolic sub-pixel peak
            y0, y1, y2 = prof[k - 1], prof[k], prof[k + 1]
            den = y0 - 2 * y1 + y2
            if den < 0:
                off += 0.5 * step * (y0 - y2) / den
        out.append(base + off * n)
        # full width at half maximum: walk out from the peak until the response halves
        half = 0.5 * prof[k]
        lo, hi = k, k
        while lo > 0 and prof[lo] > half:
            lo -= 1
        while hi < len(prof) - 1 and prof[hi] > half:
            hi += 1
        widths.append((hi - lo) * step)
    return np.asarray(out), np.asarray(widths)


def _robust_line(pts: np.ndarray):
    """Line fit with one round of outlier rejection: (point, direction, inlier residuals, inlier mask)."""
    c, d, r = _fit_line(pts)
    mad = np.median(np.abs(r - np.median(r))) * 1.4826 + 1e-6
    keep = np.abs(r) <= max(3 * mad, 0.75)
    if keep.sum() < 5:
        return c, d, r, np.ones(len(pts), bool)
    c, d, r = _fit_line(pts[keep])
    return c, d, r, keep


def _intersect(c1, d1, c2, d2):
    A = np.array([d1, -d2]).T
    if abs(np.linalg.det(A)) < 1e-9:
        return 0.5 * (c1 + c2)
    s, _ = np.linalg.solve(A, c2 - c1)
    return c1 + s * d1


def refine_hexagon(image: np.ndarray, rough: np.ndarray, kind: str, scale: float = 2.0,
                   search: float | None = None, bootstrap: int = 200, seed: int = 0) -> Refined:
    """Refine six roughly marked corners (any order around the hexagon) onto the walls/edges."""
    rough = np.asarray(rough, dtype=float)
    fmap = feature_map(image, kind, scale)
    sides = [(rough[i], rough[(i + 1) % 6]) for i in range(6)]
    if search is None:
        search = 0.15 * np.mean([np.hypot(*(b - a)) for a, b in sides])
    found, lines, rms, frac, widths = [], [], np.zeros(6), np.zeros(6), []
    for i, (a, b) in enumerate(sides):
        pts, w = _side_points(fmap, a, b, search)
        if len(pts) < 6:
            raise ValueError(f"side {i + 1}: wall/edge not found; mark the corners more closely or change 'kind'")
        c, d, r, keep = _robust_line(pts)
        found.append(pts[keep])
        widths.append(w[keep])
        lines.append((c, d))
        rms[i] = float(np.sqrt(np.mean(r ** 2)))
        frac[i] = float(keep.mean())
    corners = np.array([_intersect(*lines[(i - 1) % 6], *lines[i]) for i in range(6)])

    # Bootstrap: resample each side's wall points, refit, re-intersect -> spread of each corner.
    rng = np.random.default_rng(seed)
    boots = np.zeros((bootstrap, 6, 2))
    for k in range(bootstrap):
        bl = []
        for pts in found:
            c, d, _ = _fit_line(pts[rng.integers(0, len(pts), len(pts))])
            bl.append((c, d))
        boots[k] = [_intersect(*bl[(i - 1) % 6], *bl[i]) for i in range(6)]
    sigma = np.sqrt(np.mean(np.sum((boots - boots.mean(axis=0)) ** 2, axis=2), axis=0))
    out = Refined(points=corners, corner_sigma=sigma, side_rms=rms, inlier_frac=frac,
                  shift=np.hypot(*(corners - rough).T), wall_width_px=float(np.median(np.concatenate(widths))),
                  radius_px=float(np.hypot(*(corners - corners.mean(axis=0)).T).mean()), kind=kind, scale=scale)
    out.qc_reasons = quality(out)
    return out


def refine_auto(image: np.ndarray, rough: np.ndarray, kinds=FEATURES, scales=(1.0, 2.0, 4.0),
                bootstrap: int = 100) -> Refined:
    """Try every wall type and scale; return the best trusted result.

    "Best" = lowest corner uncertainty + SHIFT_PENALTY * mean movement from the clicks, among results that pass
    the quality gate. The movement term matters: a different, sharper line nearby (e.g. one flank of a thick wall)
    can give a tiny uncertainty while sitting well away from where the user clicked. If none pass, the best
    failing one is returned (check `qc_pass`). Raises ValueError only if no combination found walls.
    """
    results = []
    for kind in kinds:
        for scale in scales:
            try:
                results.append(refine_hexagon(image, rough, kind, scale=scale, bootstrap=bootstrap))
            except ValueError:
                continue
    if not results:
        raise ValueError("no wall or edge found along the marked sides; mark the corners more closely")
    passing = [r for r in results if r.qc_pass]
    pool = passing or results
    return min(pool, key=lambda r: r.sigma_px + SHIFT_PENALTY * float(np.mean(r.shift)))
