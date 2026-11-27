"""Tests for hexengine.metrics (residuals, central symmetry, null baseline, scores)."""

import numpy as np
import pytest

from hexengine import canonical
from hexengine.metrics import (DEFAULT_THRESHOLDS, central_symmetry_error, compute_scores, null_distribution,
                               random_hexagons, vertex_residuals)
from hexengine.transforms import apply_transform, fit_hexagon
from hexengine.types import Scores

V = canonical.VERTICES


def sim(scale, angle, tx, ty):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[scale * c, -scale * s, tx], [scale * s, scale * c, ty], [0, 0, 1.0]])


IRREGULAR = np.array([[300, 100], [420, 160], [460, 330], [310, 420], [200, 300], [180, 170]], float)


def test_vertex_residuals_shape_and_zero():
    fit = fit_hexagon(apply_transform(sim(90, 1.0, 250, 250), V))
    for kind in ("affine", "homography", "similarity"):
        r = vertex_residuals(fit, kind)
        assert r.shape == (6,)
        assert r.max() < 1e-9


def test_central_symmetry_zero_for_affine_images():
    A = np.array([[120.0, 70.0, 400.0], [-20.0, 90.0, 300.0], [0, 0, 1]])
    assert central_symmetry_error(apply_transform(A, V)) < 1e-9
    # Centrally symmetric but NOT an affine image of a regular hexagon: still zero symmetry error.
    p = np.array([[0, -1.0], [1.3, -0.2], [0.7, 0.9]])
    hexagon = np.vstack([p, -p]) * 50 + 200
    assert central_symmetry_error(hexagon) < 1e-9
    assert vertex_residuals(fit_hexagon(hexagon), "affine").mean() > 0.01


def test_central_symmetry_formula_and_invariance():
    pts = V.copy()
    pts[0] += [0.0, -0.3]           # pull V0 up: midpoint of diagonal 0 moves by (0, -0.15)
    cse = central_symmetry_error(pts)
    # independent recomputation of the documented formula
    from hexengine.transforms import _fit_similarity
    q = apply_transform(_fit_similarity(pts, V), pts)
    mids = 0.5 * (q[:3] + q[3:])
    expected = np.mean(np.linalg.norm(mids - mids.mean(axis=0), axis=1))
    assert cse == pytest.approx(expected)
    assert cse > 0.01
    # independent of click order / winding and of image scale/position/rotation
    assert central_symmetry_error(pts[::-1]) == pytest.approx(cse)
    assert central_symmetry_error(np.roll(pts, 2, axis=0)) == pytest.approx(cse)
    assert central_symmetry_error(apply_transform(sim(300, 0.5, 40, 70), pts)) == pytest.approx(cse)


def test_perspective_breaks_central_symmetry():
    H = np.array([[200.0, 30.0, 400.0], [10.0, 180.0, 300.0], [0.25, 0.15, 1.0]])
    assert central_symmetry_error(apply_transform(H, V)) > 0.01


def test_verdict_follows_selected_transform():
    # A perspective view of a regular hexagon: exact for a homography, not for an affine map.
    H = np.array([[200.0, 30.0, 400.0], [10.0, 180.0, 300.0], [0.25, 0.15, 1.0]])
    fit = fit_hexagon(apply_transform(H, V))
    by_affine = compute_scores(fit, with_null=False)
    by_homography = compute_scores(fit, with_null=False, verdict_kind="homography")
    assert by_affine.verdict_basis == "affine" and by_affine.threshold_verdict != "strong"
    assert by_homography.verdict_basis == "homography" and by_homography.threshold_verdict == "strong"


def test_random_hexagons_reproducible():
    a = random_hexagons(50, seed=1)
    b = random_hexagons(50, seed=1)
    c = random_hexagons(50, seed=2)
    assert a.shape == (50, 6, 2)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)
    r = np.linalg.norm(a, axis=-1)
    assert r.min() >= 0.5 and r.max() <= 1.5


def test_null_distribution_cached_and_keys():
    d1 = null_distribution(200, seed=5)
    d2 = null_distribution(200, seed=5)
    assert set(d1) == {"affine_mean", "homography_mean", "similarity_mean", "central_symmetry_error"}
    for k in d1:
        assert d1[k] is d2[k]                # same cached arrays
        assert len(d1[k]) == 200
        assert not d1[k].flags.writeable
    # homography (8 dof) can always fit at least as well as affine (6 dof) on average
    assert d1["homography_mean"].mean() < d1["affine_mean"].mean() < d1["similarity_mean"].mean()


def test_compute_scores_regular():
    fit = fit_hexagon(apply_transform(sim(90, 1.0, 250, 250), V))
    s = compute_scores(fit)
    assert isinstance(s, Scores)
    assert s.affine_mean < 1e-9 and s.homography_max < 1e-9 and s.similarity_max < 1e-9
    assert s.central_symmetry_error < 1e-9
    assert set(s.per_vertex) == {"affine", "homography", "similarity"}
    assert all(len(v) == 6 for v in s.per_vertex.values())
    assert s.verdict == "regular" and s.threshold_verdict == "strong"
    # a perfect hexagon beats every random hexagon
    for m in ("affine_mean", "homography_mean", "similarity_mean", "central_symmetry_error"):
        assert s.null_percentile[m] == 0.0


def test_compute_scores_irregular_and_percentile_sanity():
    fit = fit_hexagon(IRREGULAR)
    s = compute_scores(fit)
    assert s.affine_mean > 0.03
    assert s.affine_max >= s.affine_mean
    for m, pct in s.null_percentile.items():
        assert 0.0 <= pct <= 100.0
    # A random hexagon drawn from the same distribution as the null should land mid-range on average.
    pcts = [compute_scores(fit_hexagon(h)).null_percentile["affine_mean"] for h in random_hexagons(40, seed=99)]
    assert 25 < np.mean(pcts) < 75


def test_compute_scores_without_null_and_thresholds():
    fit = fit_hexagon(IRREGULAR)
    s = compute_scores(fit, with_null=False)
    assert s.null_percentile == {}
    a = s.affine_mean
    assert compute_scores(fit, thresholds={"strong": a + 1, "weak": a + 2}, with_null=False).threshold_verdict == "strong"
    assert compute_scores(fit, thresholds={"strong": a / 2, "weak": a + 1}, with_null=False).threshold_verdict == "weak"
    assert compute_scores(fit, thresholds={"strong": a / 3, "weak": a / 2}, with_null=False).threshold_verdict == "noise"
    assert DEFAULT_THRESHOLDS == {"strong": 0.03, "weak": 0.08}


def test_noise_floor_scales_with_error_and_size():
    small_err = noise_floor_fn(60, 0.5)
    big_err = noise_floor_fn(60, 2.0)
    big_cell = noise_floor_fn(240, 2.0)
    assert small_err < big_err and big_cell < big_err
    assert big_err == pytest.approx(4 * big_cell, rel=0.25)      # floor ~ sigma / radius


def test_calibrated_verdict_regular_irregular_random():
    from hexengine import metrics
    perfect = fit_hexagon(V * 80.0 + 300.0)
    s = compute_scores(perfect, corner_sigma_px=0.5)
    assert s.verdict == "regular" and s.sigma_source == "refined" and s.excess <= 0
    bent = V * 80.0 + 300.0
    bent[0] += [0, -12.0]                                           # one corner pulled out by 0.15 R
    s2 = compute_scores(fit_hexagon(bent), corner_sigma_px=0.5)
    assert s2.verdict == "irregular" and s2.excess > 0
    assert s2.affine_mean > 0 and s2.threshold_verdict in ("strong", "weak", "noise")   # raw values kept
    assert metrics.shape_verdict(0.3, 0.01, null_pct=40.0) == "random-like"
    s3 = compute_scores(fit_hexagon(bent))                          # no refinement: assumed click error
    assert s3.sigma_source == "assumed" and s3.corner_sigma_px == metrics.DEFAULT_CLICK_SIGMA_PX


def noise_floor_fn(radius, sigma):
    from hexengine.metrics import noise_floor
    return noise_floor(radius, sigma, n=300)
