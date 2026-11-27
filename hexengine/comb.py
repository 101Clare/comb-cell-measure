"""Honeycomb cell measurement in millimetres from one photo with a ruler on the comb.

This is the pilot pipeline (experiments/honeycomb_pilot.py, results_v4) as one call, for the beekeeper app:

1. Scale: two points on the ruler and the distance between them (mm) -> px per mm. If a tick strip is given, the
   ruler's millimetre-tick period is measured along it as an independent cross-check.
2. Lattice: the comb's repeat vectors from the autocorrelation of the comb region (lattice.lattice_vectors). Its six
   neighbour spacings must agree within 15 %, or there is no clear hexagonal comb in the box.
3. Cells: an anchor is found by sliding a 3 x 3 patch of lattice cells over one cell's worth of positions and
   keeping the best fit to the walls. Every lattice cell inside the box is then snapped to its walls
   (refine.refine_auto) and must pass the engine's quality gate.
4. Tilt: a phone held at an angle squashes the comb along one direction, so cells read too narrow that way (10
   degrees costs about 1.5 %). Real worker comb is close to a regular hexagonal lattice, so the centres of the
   snapped cells are fitted with a lattice, and the stretch that makes it regular gives the tilt. Above
   TILT_MIN_DEG, and when the squash stands 2.5 standard errors clear of its bootstrap spread, the widths and the ruler scale are measured after that stretch (the ruler lies in the same plane);
   above TILT_WARN_DEG the user is asked to retake the photo straight-on.
5. Per cell: flat-to-flat width (wall centre to wall centre, the quantity a "10 cells = 48 mm" count measures) and
   size class. Per comb: median and quartiles over the passed cells, plus the lattice spacing over the whole box
   as a second, independent estimate.

Accuracy so far: on two controls with a printed ground truth, both estimates agreed within 0.06 mm, from small web
images. Single cells scatter by about +-0.1 mm at 15-25 px cell radius; the medians are the numbers to use.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import lattice, morphometry, refine, render

LATTICE_SPREAD_MAX = 0.15
TILT_MIN_DEG = 3.0       # below this the change is under 0.15 % (and near the fit's noise, ~1 degree): no correction
TILT_WARN_DEG = 20.0     # above this perspective is no longer a simple stretch: ask for a straight-on photo
MIN_CELLS_FOR_TILT = 8
SCALES = (1.0, 1.5, 2.5)


@dataclass
class CellRow:
    width_mm: float
    widths_mm: tuple[float, float, float]
    anisotropy_mm: float
    size_class: str
    radius_px: float
    corner_sigma_px: float
    qc_pass: bool
    qc_reasons: str
    points: np.ndarray


@dataclass
class CombReport:
    px_per_mm: float
    tick_px_per_mm: float | None
    scale_disagreement_pct: float | None
    lattice_width_mm: float
    cells_tried: int
    cells_passed: int
    median_width_mm: float | None
    quartiles_mm: tuple[float, float] | None
    size_class: str | None
    class_counts: dict[str, int]
    tilt_deg: float | None = None            # estimated camera tilt from the cell lattice
    tilt_corrected: bool = False
    median_uncorrected_mm: float | None = None
    cells: list[CellRow] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    overlay: np.ndarray | None = None


def _edge_score(fmap, corners_list) -> float:
    vals = []
    for c in corners_list:
        for k in range(6):
            a, b = c[k], c[(k + 1) % 6]
            ts = np.linspace(0.15, 0.85, 12)[:, None]
            pts = (a + ts * (b - a)).astype(np.float32)
            vals.append(cv2.remap(fmap, pts[:, :1], pts[:, 1:], cv2.INTER_LINEAR).ravel())
    return float(np.mean(np.concatenate(vals))) if vals else 0.0


def fit_lattice(centres: np.ndarray, basis: np.ndarray, anchor, n_boot: int = 0,
                seed: int = 0) -> np.ndarray | None | tuple[np.ndarray, list[np.ndarray]]:
    """Least-squares lattice basis (2, 2) through cell centres (n, 2), indexed by rounding against `basis`.

    With n_boot > 0, also returns n_boot bases refitted on resampled cells (for the tilt's uncertainty)."""
    centres = np.asarray(centres, float)
    ij = np.round((centres - np.asarray(anchor, float)) @ np.linalg.inv(basis)).astype(int)
    if len(centres) < MIN_CELLS_FOR_TILT or np.ptp(ij[:, 0]) < 2 or np.ptp(ij[:, 1]) < 2:
        return None
    X = np.column_stack([np.ones(len(ij)), ij])
    keep = np.ones(len(X), bool)
    for _ in range(4):                                      # refit without cells that sit off the lattice
        coef, *_ = np.linalg.lstsq(X[keep], centres[keep], rcond=None)   # rows: origin, a1, a2
        res = np.hypot(*(centres - X @ coef).T)
        new = res <= max(0.5, 3.0 * np.median(res[keep]))
        if new.sum() < MIN_CELLS_FOR_TILT or (new == keep).all():
            break
        keep = new
    if not n_boot:
        return coef[1:]
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(keep)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(idx, len(idx))
        if np.ptp(ij[pick, 0]) < 2 or np.ptp(ij[pick, 1]) < 2:
            continue
        b, *_ = np.linalg.lstsq(X[pick], centres[pick], rcond=None)
        boots.append(b[1:])
    return coef[1:], boots


def _squash(basis: np.ndarray) -> np.ndarray:
    """Squash as a headless 2-vector (1 - cos tilt) * (cos 2phi, sin 2phi), phi the squash axis: averages cleanly."""
    S, tilt = regularising_stretch(basis)
    w, v = np.linalg.eigh(S)
    phi = np.arctan2(v[1, 1], v[0, 1])                      # axis of the largest stretch
    m = 1 - np.cos(np.radians(tilt))
    return m * np.array([np.cos(2 * phi), np.sin(2 * phi)])


def tilt_is_significant(fitted: np.ndarray, boots: list[np.ndarray], z: float = 2.5) -> bool:
    """True when the squash stands clear of its bootstrap spread (z standard errors along its own direction)."""
    q = _squash(fitted)
    if len(boots) < 20 or not np.any(q):
        return False
    d = q / np.hypot(*q)
    proj = np.array([_squash(b) @ d for b in boots])
    return bool(np.hypot(*q) > z * proj.std())


def regularising_stretch(basis: np.ndarray) -> tuple[np.ndarray, float]:
    """Symmetric 2x2 stretch S (det 1) that turns the lattice into a regular hexagonal one, and the tilt (degrees)
    a flat comb would need to look this squashed (cos tilt = smallest / largest stretch)."""
    n = lattice.neighbour_vectors(basis)
    u1, u2 = n[0], n[1]                                     # adjacent by angle: about 60 degrees apart
    U = np.column_stack([u1, u2])
    sgn = 1.0 if u1[0] * u2[1] - u1[1] * u2[0] > 0 else -1.0
    T = np.column_stack([[1.0, 0.0], [0.5, sgn * np.sqrt(3) / 2]])
    A = T @ np.linalg.inv(U)
    W, sv, Vt = np.linalg.svd(A)
    S = Vt.T @ np.diag(sv / np.sqrt(sv[0] * sv[1])) @ Vt   # drop the rotation, keep area
    return S, float(np.degrees(np.arccos(min(1.0, sv[1] / sv[0]))))


def comb_lattice(img: np.ndarray, box) -> np.ndarray:
    """Lattice basis (2, 2) inside the box (x0, y0, x1, y1); ValueError if there is no clear hexagonal repeat."""
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    roi = np.ascontiguousarray(img[y0:y1, x0:x1])
    if min(roi.shape[:2]) < 40:
        raise ValueError("the comb area is too small: drag a larger box over the cells")
    basis = lattice.lattice_vectors(roi, min_period_px=8.0, max_period_px=max(roi.shape[:2]) / 2.2)
    d = np.hypot(*lattice.neighbour_vectors(basis).T)
    if (d.max() - d.min()) / d.mean() > LATTICE_SPREAD_MAX:
        raise ValueError("no clear hexagonal cell pattern in the box (neighbour spacings "
                         f"{np.round(d, 1).tolist()} px). Try a box over clean, straight-on cells.")
    return basis


def find_anchor(img: np.ndarray, basis: np.ndarray, box, wall_kind: str = "bright_ridge") -> np.ndarray:
    x0, y0, x1, y1 = box
    guess = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    fmap = refine.feature_map(img, wall_kind, 1.0)
    rep = float(min(np.hypot(*basis[0]), np.hypot(*basis[1])))
    best, best_s = guess, -1.0
    for dx in np.linspace(-0.6 * rep, 0.6 * rep, 25):
        for dy in np.linspace(-0.6 * rep, 0.6 * rep, 25):
            c = guess + [dx, dy]
            cells = [lattice.cell_corners(c + i * basis[0] + j * basis[1], basis) for i in (-1, 0, 1) for j in (-1, 0, 1)]
            s = _edge_score(fmap, cells)
            if s > best_s:
                best, best_s = c, s
    return best


def cells_in_box(anchor, basis, box, shape, max_cells: int) -> list[np.ndarray]:
    x0, y0, x1, y1 = box
    out = []
    for corners in lattice.lattice_cells(anchor, basis, shape, margin_frac=0.05, max_cells=2000):
        if (corners[:, 0] >= x0).all() and (corners[:, 0] <= x1).all() and \
           (corners[:, 1] >= y0).all() and (corners[:, 1] <= y1).all():
            out.append(corners)
    return out[:max_cells]


def measure_comb(img: np.ndarray, ruler_p1, ruler_p2, ruler_mm: float, box, tick_strip=None,
                 wall_kind: str = "bright_ridge", max_cells: int = 60, draw: bool = True) -> CombReport:
    """Measure the comb cells inside `box` (x0, y0, x1, y1, image px) of an RGB image.

    ruler_p1, ruler_p2: two points on the ruler (image px) `ruler_mm` apart. tick_strip: optional (p1, p2) line
    along the ruler's millimetre ticks for the cross-check. wall_kind: "bright_ridge" when the cell walls are
    lighter than the cells (most comb), "dark_ridge" when darker (backlit comb).
    """
    if ruler_mm <= 0:
        raise ValueError("the ruler distance must be positive")
    scale = morphometry.Scale(tuple(ruler_p1), tuple(ruler_p2), float(ruler_mm))
    if scale.length_px < 20:
        raise ValueError("the two ruler points are too close together: pick marks further apart")
    warnings = []
    ticks = None
    if tick_strip is not None:
        ticks = morphometry.ruler_tick_period(img, *tick_strip, expect_px_per_mm=scale.px_per_mm)
    dis = None if ticks is None else 100 * (ticks / scale.px_per_mm - 1)
    if dis is not None and abs(dis) > 3:
        warnings.append(f"the ruler's tick spacing disagrees with your two points by {dis:+.1f} %: check the "
                        "two marks and the distance")
    x0, y0, x1, y1 = (float(v) for v in box)
    box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    basis = comb_lattice(img, box)
    lattice_px = float(np.mean(np.hypot(*lattice.neighbour_vectors(basis).T)))
    anchor = find_anchor(img, basis, box, wall_kind)
    cands = cells_in_box(anchor, basis, box, img.shape, max_cells)
    shown = img.copy() if draw else None
    if draw:
        cv2.line(shown, tuple(int(v) for v in ruler_p1), tuple(int(v) for v in ruler_p2), (255, 0, 255), 2,
                 cv2.LINE_AA)
        cv2.rectangle(shown, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (60, 200, 60), 1)
    rows = []
    for rough in cands:
        try:
            r = refine.refine_auto(img, rough, kinds=(wall_kind, "edge"), scales=SCALES, bootstrap=100)
        except ValueError:
            continue
        m = morphometry.measure_cell(r.points, scale, r.wall_width_px)
        rows.append(CellRow(width_mm=float(m.width_mm), widths_mm=tuple(float(v) for v in m.widths_mm),
                            anisotropy_mm=float(m.anisotropy_mm), size_class=m.size_class,
                            radius_px=float(r.radius_px), corner_sigma_px=float(r.sigma_px), qc_pass=r.qc_pass,
                            qc_reasons="; ".join(r.qc_reasons), points=r.points))
        if draw:
            shown = render.draw_points(shown, r.points, labels=False,
                                       color=(0, 230, 255) if r.qc_pass else (255, 60, 60))
    ok = [c for c in rows if c.qc_pass]
    med_raw = float(np.median([c.width_mm for c in ok])) if ok else None
    tilt, corrected = None, False
    fit = fit_lattice(np.array([c.points.mean(axis=0) for c in ok]), basis, anchor, n_boot=200) if ok else None
    if fit is not None:
        fitted, boots = fit
        S, tilt = regularising_stretch(fitted)
        lattice_px = float(np.mean(np.hypot(*lattice.neighbour_vectors(fitted).T)))   # sub-pixel spacing
        if tilt >= TILT_MIN_DEG and tilt_is_significant(fitted, boots):
            corrected = True
            p1s, p2s = S @ np.asarray(ruler_p1, float), S @ np.asarray(ruler_p2, float)
            scale = morphometry.Scale(tuple(p1s), tuple(p2s), float(ruler_mm))
            for c in rows:
                m = morphometry.measure_cell(c.points @ S.T, scale)
                c.width_mm, c.widths_mm = float(m.width_mm), tuple(float(v) for v in m.widths_mm)
                c.anisotropy_mm, c.size_class = float(m.anisotropy_mm), m.size_class
            lattice_px = float(np.mean(np.hypot(*(lattice.neighbour_vectors(fitted) @ S.T).T)))
        if tilt > TILT_WARN_DEG:
            warnings.append(f"the photo looks tilted by about {tilt:.0f} degrees: the tilt is corrected, but for "
                            "the best accuracy hold the phone parallel to the comb")
    w = np.array([c.width_mm for c in ok])
    med = float(np.median(w)) if len(w) else None
    if len(ok) < 5:
        warnings.append(f"only {len(ok)} cells passed the quality check: use a sharper, straight-on photo "
                        "or a box over cleaner cells; the lattice spacing is the more reliable number here")
    if ok and np.median([c.radius_px for c in ok]) < 20:
        warnings.append("the cells are small in this photo (under 20 px radius): single-cell widths scatter by "
                        "about 0.1 mm; move closer or use the full-size photo")
    counts: dict[str, int] = {}
    for c in ok:
        counts[c.size_class] = counts.get(c.size_class, 0) + 1
    return CombReport(
        px_per_mm=scale.px_per_mm, tick_px_per_mm=ticks, scale_disagreement_pct=dis,
        lattice_width_mm=lattice_px * scale.mm_per_px, cells_tried=len(cands), cells_passed=len(ok),
        median_width_mm=med, tilt_deg=tilt, tilt_corrected=corrected, median_uncorrected_mm=med_raw,
        quartiles_mm=None if len(w) < 4 else tuple(float(q) for q in np.percentile(w, [25, 75])),
        size_class=None if med is None else morphometry.classify(med), class_counts=counts, cells=rows,
        warnings=warnings, overlay=shown)
