"""Tests for hexengine.transforms (ordering, fits, point mapping, warping)."""

import itertools

import cv2
import numpy as np
import pytest

from hexengine import canonical
from hexengine.metrics import vertex_residuals
from hexengine.transforms import (apply_transform, fit_hexagon, local_direction_to, order_points,
                                  signed_area, warp_to_canonical)
from hexengine.types import TransformKind

V = canonical.VERTICES
KINDS = [k.value for k in TransformKind]


def similarity_matrix(scale, angle, tx, ty):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[scale * c, -scale * s, tx], [scale * s, scale * c, ty], [0, 0, 1.0]])


def click_variants(points):
    """All 12 ways of clicking a hexagon: 6 start vertices x 2 windings."""
    for start, reverse in itertools.product(range(6), (False, True)):
        idx = np.roll(np.arange(6), -start)
        if reverse:
            idx = idx[::-1]
        yield points[idx], idx


# ---------------------------------------------------------------------------- ordering

def test_canonical_is_clockwise_positive_area():
    assert signed_area(V) > 0


def test_order_points_identity_on_canonical():
    ordered, order = order_points(V)
    np.testing.assert_array_equal(order, np.arange(6))
    np.testing.assert_allclose(ordered, V)


@pytest.mark.parametrize("start,reverse", list(itertools.product(range(6), (False, True))))
def test_order_points_topmost_any_click_order(start, reverse):
    pts = apply_transform(similarity_matrix(100, 0.1, 300, 200), V)  # small tilt, V0 still topmost
    idx = np.roll(np.arange(6), -start)
    if reverse:
        idx = idx[::-1]
    clicks = pts[idx]
    ordered, order = order_points(clicks)
    np.testing.assert_allclose(ordered, pts, atol=1e-12)
    np.testing.assert_allclose(clicks[order], ordered)
    assert signed_area(ordered) > 0


def test_order_points_rotation_offset():
    base, base_order = order_points(V)
    for offset in range(-6, 13):
        ordered, order = order_points(V, rotation_offset=offset)
        k = offset % 6
        # index (0 + offset) % 6 of the topmost-first list becomes V0
        np.testing.assert_allclose(ordered[0], base[k])
        np.testing.assert_allclose(ordered, np.roll(base, -k, axis=0))
        np.testing.assert_array_equal(order, np.roll(base_order, -k))


def test_order_points_flat_top_tie_goes_left():
    flat = apply_transform(similarity_matrix(1, np.deg2rad(30), 0, 0), V)  # vertices at -60 and -120 deg
    ordered, _ = order_points(flat)
    assert ordered[0, 1] < 0 and ordered[0, 0] < 0       # upper-left of the tied pair
    assert ordered[1, 0] > 0 and ordered[1, 1] < 0       # then clockwise to upper-right


def test_order_points_errors():
    with pytest.raises(ValueError):
        order_points(V[:5])
    with pytest.raises(ValueError):
        order_points(np.vstack([V, V[:1]]))
    with pytest.raises(ValueError):
        order_points(np.column_stack([np.arange(6.0), 2 * np.arange(6.0)]))  # collinear
    with pytest.raises(ValueError):
        order_points(np.ones((6, 2)))


# ---------------------------------------------------------------------------- fits

@pytest.mark.parametrize("scale,angle,tx,ty", [
    (1.0, 0.0, 0.0, 0.0),
    (150.0, 0.7, 640.0, 360.0),
    (37.5, -2.1, 12.0, 900.0),
    (420.0, np.pi, 1000.0, 50.0),
])
def test_regular_hexagon_all_residuals_zero(scale, angle, tx, ty):
    from hexengine.metrics import central_symmetry_error

    pts = apply_transform(similarity_matrix(scale, angle, tx, ty), V)
    for clicks, _ in click_variants(pts):
        for offset in (0, 3, 5):
            fit = fit_hexagon(clicks, rotation_offset=offset)
            for kind in KINDS:
                assert vertex_residuals(fit, kind).max() < 1e-9, kind
            assert central_symmetry_error(clicks) < 1e-9


def test_fit_matrices_are_3x3_and_map_image_to_canonical():
    pts = apply_transform(similarity_matrix(80, 0.3, 200, 150), V)
    fit = fit_hexagon(pts)
    for kind in KINDS:
        M = fit.matrix(kind)
        assert M.shape == (3, 3) and M.dtype == np.float64
        np.testing.assert_allclose(apply_transform(M, fit.image_points), V, atol=1e-9)
    assert fit.rotation_offset == 0
    np.testing.assert_allclose(pts[fit.order], fit.image_points)


def test_affine_sheared_hexagon():
    A = np.array([[120.0, 70.0, 400.0], [-20.0, 90.0, 300.0], [0, 0, 1]])  # shear + non-uniform scale
    pts = apply_transform(A, V)
    for clicks, _ in click_variants(pts):
        fit = fit_hexagon(clicks)
        assert vertex_residuals(fit, "affine").max() < 1e-9
        assert vertex_residuals(fit, "homography").max() < 1e-9
        assert vertex_residuals(fit, "similarity").mean() > 0.05


def test_perspective_hexagon():
    H = np.array([[200.0, 30.0, 400.0], [10.0, 180.0, 300.0], [0.25, 0.15, 1.0]])
    pts = apply_transform(H, V)
    for clicks, _ in click_variants(pts):
        fit = fit_hexagon(clicks)
        assert vertex_residuals(fit, "homography").max() < 1e-8
        assert vertex_residuals(fit, "affine").mean() > 0.01


def test_homography_refinement_never_worse_than_affine():
    rng = np.random.default_rng(3)
    for _ in range(20):
        pts = V * rng.uniform(0.6, 1.4, size=(6, 1)) * 100 + 300
        fit = fit_hexagon(pts)
        # homography contains affine, so its least-squares optimum cannot be worse
        assert (np.sum(vertex_residuals(fit, "homography") ** 2)
                <= np.sum(vertex_residuals(fit, "affine") ** 2) + 1e-9)


def test_irregular_hexagon_has_clear_affine_residual():
    pts = np.array([[300, 100], [420, 160], [460, 330], [310, 420], [200, 300], [180, 170]], float)
    fit = fit_hexagon(pts)
    assert vertex_residuals(fit, "affine").mean() > 0.03


# ---------------------------------------------------------------------------- point mapping

def test_apply_transform_single_and_batch():
    M = np.array([[2.0, 0, 1], [0, 3.0, -1], [0, 0, 1]])
    np.testing.assert_allclose(apply_transform(M, np.array([1.0, 1.0])), [3.0, 2.0])
    np.testing.assert_allclose(apply_transform(M, np.array([[1.0, 1.0], [0, 0]])), [[3.0, 2.0], [1.0, -1.0]])
    H = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 2.0]])
    np.testing.assert_allclose(apply_transform(H, np.array([[4.0, 6.0]])), [[2.0, 3.0]])


def test_local_direction_to():
    R = similarity_matrix(2.0, np.deg2rad(30), 5, 5)
    out = local_direction_to(R, np.array([1.0, 2.0]), 0.0)
    assert out == pytest.approx(np.deg2rad(30))
    # orientation result folds into [0, pi)
    out = local_direction_to(R, np.array([1.0, 2.0]), np.deg2rad(170))
    assert out == pytest.approx(np.deg2rad(20))
    assert 0 <= local_direction_to(np.eye(3), np.zeros(2), np.pi) < np.pi
    # a tiny negative angle must not round up to exactly pi (np.mod(-1e-20, pi) == pi)
    assert local_direction_to(np.eye(3), np.zeros(2), -1e-20) == 0.0
    # shear: horizontal stays horizontal, vertical tilts
    S = np.array([[1.0, 1.0, 0], [0, 1.0, 0], [0, 0, 1]])
    assert local_direction_to(S, np.zeros(2), np.pi / 2) == pytest.approx(np.pi / 4)


# ---------------------------------------------------------------------------- warping

FACE_COLOURS = [(220, 40, 40), (40, 200, 40), (40, 40, 220)]
# Three rhombi (centre + V_a, V_b, V_c) tiling the hexagon, with a distinct colour each.
FACES = [(5, 0, 1), (1, 2, 3), (3, 4, 5)]


def draw_cube_image(M_canon_to_img, shape=(600, 800)):
    img = np.full(shape + (3,), 255, np.uint8)
    for (a, b, c), colour in zip(FACES, FACE_COLOURS):
        poly = np.vstack([np.zeros(2), V[a], V[b], V[c]])
        poly_img = apply_transform(M_canon_to_img, poly)
        cv2.fillPoly(img, [np.round(poly_img).astype(np.int32)], colour)
    return img


@pytest.mark.parametrize("M,kind", [
    (np.array([[150.0, 60.0, 400.0], [-30.0, 120.0, 300.0], [0, 0, 1]]), "affine"),
    (similarity_matrix(140, 0.4, 400, 300), "similarity"),
    (np.array([[160.0, 20.0, 400.0], [10.0, 150.0, 300.0], [0.2, 0.1, 1.0]]), "homography"),
])
def test_warp_to_canonical_maps_faces(M, kind):
    img = draw_cube_image(M)
    pts = apply_transform(M, V)[::-1]     # clicked anticlockwise, still fine
    fit = fit_hexagon(pts)
    size = 256
    out = warp_to_canonical(img, fit, kind=kind, size=size)
    assert out.shape == (size, size, 3) and out.dtype == np.uint8

    for (a, b, c), colour in zip(FACES, FACE_COLOURS):
        centre = (V[a] + V[b] + V[c]) / 4.0          # rhombus centroid (incl. the origin corner)
        x, y = np.round(canonical.to_frame(centre, size)).astype(int)
        np.testing.assert_allclose(out[y, x], colour, atol=3)
    # outside the hexagon is masked grey
    np.testing.assert_array_equal(out[0, 0], (30, 30, 30))
    np.testing.assert_array_equal(out[size // 2, 2], (30, 30, 30))


def test_warp_without_mask_and_default_size():
    M = similarity_matrix(140, 0.0, 400, 300)
    img = draw_cube_image(M)
    fit = fit_hexagon(apply_transform(M, V))
    out = warp_to_canonical(img, fit, mask_outside=False)
    assert out.shape == (canonical.FRAME_SIZE, canonical.FRAME_SIZE, 3)
    # the frame corner lies inside the source image (white background), so it is not masked
    np.testing.assert_array_equal(out[0, 0], (255, 255, 255))
