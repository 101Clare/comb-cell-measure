"""Residual distortion / symmetry deviation scores, plus a random-hexagon null baseline.

OWNER: Agent A (geometry).

Mapping 6 points onto a shape that strictly needs 3 (affine) or 4 (homography) leaves leftover degrees of
freedom; the mean vertex residual measures how much of the observed shape cannot be explained by that
transform. Lower = closer to a projected regular hexagon.

Scores (all in canonical units, circumradius = 1)
-------------------------------------------------
* ``<kind>_mean`` / ``<kind>_max``: mean / largest of the six per-vertex residuals
  r_i = |M(p_i) - V_i| for the fitted transform M of that kind (see transforms.py).
  - affine ~ 0      -> the hexagon is a parallel-projected regular hexagon (the headline score).
  - homography ~ 0  -> it is a perspective view of a flat regular hexagon.
  - similarity ~ 0  -> it is itself (almost) regular, before any skew.
* ``central_symmetry_error``: whether opposite sides are parallel and equal. Every parallel view of a cube
  is centrally symmetric, even when it is not an affine image of a *regular* hexagon, so this score keeps
  genuinely cube-like shapes from being penalised by the affine residual alone. Formula in its docstring.

Null baseline
-------------
A "low" residual only means something next to what chance gives. ``random_hexagons`` draws 1,000 random
hexagons (fixed seed, so the numbers are reproducible) and ``null_distribution`` scores them the same way.
``null_percentile[metric]`` is the percentage of those random hexagons with a LOWER (better) value than the
observed one: near 0 % means the observed hexagon beats almost every random one.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import canonical
from .transforms import _fit_affine, _fit_similarity, apply_transform, fit_hexagon, order_points
from .types import HexFit, Scores, TransformKind

# Default verdict thresholds on the affine mean residual (canonical units). Tunable from the UI.
DEFAULT_THRESHOLDS = {"strong": 0.03, "weak": 0.08}

# The metrics that get a null baseline / percentile.
NULL_METRICS = ("affine_mean", "homography_mean", "similarity_mean", "central_symmetry_error")


def vertex_residuals(fit: HexFit, kind: str) -> np.ndarray:
    """Per-vertex Euclidean distance (6,) between transformed image point i and canonical vertex V_i."""
    mapped = apply_transform(fit.matrix(kind), fit.image_points)
    return np.linalg.norm(mapped - canonical.VERTICES, axis=1)


def central_symmetry_error(points: np.ndarray) -> float:
    """0 when p_i + p_{i+3} is the same for all i (opposite sides parallel and equal).
    Computed AFTER similarity normalisation so it is in canonical units and scale-invariant.

    Exact formula. The points are first put in the standard cyclic order (order_points) and mapped into
    canonical units with the least-squares similarity fit, giving q_0..q_5. For the three opposite pairs
    (k = 0, 1, 2) take the midpoint of the long diagonal

        m_k = (q_k + q_{k+3}) / 2,

    and their common mean  m = (m_0 + m_1 + m_2) / 3. The error is the mean distance

        error = (1/3) * sum_k |m_k - m|.

    Why this works: p_i + p_{i+3} equal for all i  <=>  all three long diagonals bisect each other at one
    point  <=>  p_{i+1} - p_i = -(p_{i+4} - p_{i+3}), i.e. each side is parallel and equal to its opposite
    side. Relabelling or reversing the vertices only permutes the pairs, so the score does not depend on
    the click order.
    """
    ordered, _ = order_points(points)
    q = apply_transform(_fit_similarity(ordered, canonical.VERTICES), ordered)
    mids = 0.5 * (q[:3] + q[3:])
    return float(np.mean(np.linalg.norm(mids - mids.mean(axis=0), axis=1)))


def random_hexagons(n: int = 1000, seed: int = 0) -> np.ndarray:
    """(n, 6, 2) random convex-ish hexagons for the null baseline: six sorted random angles with
    random radii in [0.5, 1.5] around the origin (documented in the docstring for reproducibility).

    Recipe (numpy.random.default_rng(seed)): for each hexagon draw six angles uniform in [0, 2*pi) and sort
    them ascending (which, with y down, runs clockwise on screen), and six radii uniform in [0.5, 1.5];
    vertex j = r_j * (cos a_j, sin a_j). The shapes are star-shaped around the origin but not necessarily
    convex; they span everything from nearly regular to badly skewed.
    """
    rng = np.random.default_rng(seed)
    angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, size=(n, 6)), axis=1)
    radii = rng.uniform(0.5, 1.5, size=(n, 6))
    return np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=-1)


@lru_cache(maxsize=8)
def _null_distribution_cached(n: int, seed: int) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {m: [] for m in NULL_METRICS}
    for hexagon in random_hexagons(n, seed):
        try:
            fit = fit_hexagon(hexagon)
        except ValueError:  # (practically impossible) degenerate draw: skip it
            continue
        for kind in TransformKind:
            values[f"{kind.value}_mean"].append(float(np.mean(vertex_residuals(fit, kind.value))))
        values["central_symmetry_error"].append(central_symmetry_error(fit.image_points))
    out = {}
    for m, v in values.items():
        arr = np.asarray(v, dtype=np.float64)
        arr.setflags(write=False)  # shared via the cache: make accidental mutation an error
        out[m] = arr
    return out


def null_distribution(n: int = 1000, seed: int = 0) -> dict[str, np.ndarray]:
    """Scores for random hexagons: keys 'affine_mean', 'homography_mean', 'similarity_mean',
    'central_symmetry_error'. Cache the result (functools.lru_cache) — it is reused for every image.

    The arrays are cached and read-only; the returned dict itself is a fresh copy each call.
    """
    return dict(_null_distribution_cached(int(n), int(seed)))


def null_percentile(value: float, null_values: np.ndarray) -> float:
    """Percentage (0..100) of null values strictly LOWER than `value`."""
    null_values = np.asarray(null_values)
    if null_values.size == 0:
        return float("nan")
    return float(100.0 * np.mean(null_values < value))


# ---------------------------------------------------------------------------------------------------------------------
# Noise floor and the calibrated verdict
# ---------------------------------------------------------------------------------------------------------------------

# Assumed corner error for hand clicks that were not snapped to the walls (px). Refined corners bring their own.
DEFAULT_CLICK_SIGMA_PX = 1.5
# Smallest corner uncertainty used for a noise floor (px). refine's bootstrap measures only the random scatter of
# the wall points; it cannot see systematic error (JPEG blocks, anti-aliasing, a slightly bowed wall), and on clean
# patterns it reports ~0.05 px. Sub-pixel edge positions in real photos are rarely better than ~0.25 px, and with
# the raw bootstrap value even the tumbling-block control (residual 0.002) fell outside its floor.
MIN_CORNER_SIGMA_PX = 0.25
# A hexagon whose residual is no better than this percentage of random hexagons is "random-like".
RANDOM_LIKE_PCT = 5.0
VERDICTS = ("regular", "irregular", "random-like")


@lru_cache(maxsize=1024)
def _noise_floor_cached(radius_px: float, sigma_px: float, kind: str, quantile: float, n: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    v = canonical.VERTICES * radius_px
    vals = np.empty(n)
    for k in range(n):
        q, _ = order_points(v + rng.normal(0.0, max(sigma_px, 1e-6), v.shape))
        if kind == "homography":
            m = fit_hexagon(q).homography          # slower (non-linear refinement), so only when asked for
        elif kind == "similarity":
            m = _fit_similarity(q, canonical.VERTICES)
        else:
            m = _fit_affine(q, canonical.VERTICES)
        vals[k] = np.mean(np.linalg.norm(apply_transform(m, q) - canonical.VERTICES, axis=1))
    return float(np.quantile(vals, quantile))


def noise_floor(radius_px: float, sigma_px: float, kind: TransformKind | str = TransformKind.AFFINE,
                quantile: float = 0.95, n: int = 400, seed: int = 0) -> float:
    """Mean vertex residual that a PERFECT regular hexagon of this size would show if each corner were off by
    Gaussian noise of sigma_px (Monte Carlo, the `quantile` of n trials, canonical units).

    A measured residual at or below this cannot be told apart from a perfect hexagon: the deviation is within
    marking error. Only the excess above it is real shape information. Cached on rounded inputs.
    """
    return _noise_floor_cached(round(float(radius_px) * 2) / 2, round(float(sigma_px), 2),
                               TransformKind(kind).value, float(quantile), int(n), int(seed))


def shape_verdict(residual: float, floor: float, null_pct: float | None = None) -> str:
    """'regular': within the noise floor (indistinguishable from a perfect projected hexagon);
    'random-like': no better than RANDOM_LIKE_PCT % of random hexagons; otherwise 'irregular' (a real hexagon,
    measurably distorted beyond marking error)."""
    if residual <= floor:
        return "regular"
    if null_pct is not None and np.isfinite(null_pct) and null_pct >= RANDOM_LIKE_PCT:
        return "random-like"
    return "irregular"


def hexagon_radius_px(points: np.ndarray) -> float:
    """Mean distance of the six corners from their centroid (image px)."""
    p = np.asarray(points, float)
    return float(np.hypot(*(p - p.mean(axis=0)).T).mean())


def verdict(affine_mean: float, thresholds: dict[str, float] | None = None) -> str:
    """The original fixed-threshold verdict, kept for comparison (Scores.threshold_verdict):
    'strong' if affine_mean < thresholds['strong'], 'weak' if < thresholds['weak'], else 'noise'."""
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    if affine_mean < th["strong"]:
        return "strong"
    if affine_mean < th["weak"]:
        return "weak"
    return "noise"


def compute_scores(fit: HexFit, thresholds: dict[str, float] | None = None, with_null: bool = True,
                   verdict_kind: TransformKind | str = TransformKind.AFFINE,
                   corner_sigma_px: float | None = None) -> Scores:
    """All scores for one fit, the null percentile for each headline metric, the cell's noise floor and the
    calibrated verdict (see shape_verdict), judged on the `verdict_kind` mean residual.

    corner_sigma_px: the corner uncertainty in image px (from refine.refine_hexagon), never taken below
    MIN_CORNER_SIGMA_PX. If None, hand clicks are assumed to be off by DEFAULT_CLICK_SIGMA_PX and
    Scores.sigma_source says "assumed".

    Why verdict_kind: a photo of a flat regular hexagon taken at an angle is an exact homography but not an exact
    affine image, so judging it by the affine residual alone would call a perfect perspective view distorted. The
    UI passes the transform selected for the warp.

    per_vertex maps 'affine' / 'homography' / 'similarity' to the six residuals (V0..V5 order).
    null_percentile (only if with_null) maps each of NULL_METRICS to a percentage, see null_percentile().
    threshold_verdict keeps the original fixed-threshold verdict so raw numbers can still be compared.
    """
    per_vertex = {kind.value: vertex_residuals(fit, kind.value) for kind in TransformKind}
    stats = {}
    for kind, r in per_vertex.items():
        stats[f"{kind}_mean"] = float(np.mean(r))
        stats[f"{kind}_max"] = float(np.max(r))
    cse = central_symmetry_error(fit.image_points)

    percentiles: dict[str, float] = {}
    if with_null:
        null = null_distribution()
        observed = {**stats, "central_symmetry_error": cse}
        percentiles = {m: null_percentile(observed[m], null[m]) for m in NULL_METRICS}

    basis = TransformKind(verdict_kind).value
    residual = stats[f"{basis}_mean"]
    radius = hexagon_radius_px(fit.image_points)
    sigma = DEFAULT_CLICK_SIGMA_PX if corner_sigma_px is None else max(float(corner_sigma_px), MIN_CORNER_SIGMA_PX)
    floor = noise_floor(radius, sigma, basis)
    return Scores(
        affine_mean=stats["affine_mean"],
        affine_max=stats["affine_max"],
        homography_mean=stats["homography_mean"],
        homography_max=stats["homography_max"],
        similarity_mean=stats["similarity_mean"],
        similarity_max=stats["similarity_max"],
        central_symmetry_error=cse,
        per_vertex={k: [float(x) for x in v] for k, v in per_vertex.items()},
        null_percentile=percentiles,
        verdict=shape_verdict(residual, floor, percentiles.get(f"{basis}_mean")),
        verdict_basis=basis,
        noise_floor=floor,
        excess=residual - floor,
        radius_px=radius,
        corner_sigma_px=sigma,
        sigma_source="assumed" if corner_sigma_px is None else "refined",
        threshold_verdict=verdict(residual, thresholds),
    )
