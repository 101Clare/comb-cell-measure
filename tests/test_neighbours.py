"""Tests for hexengine.neighbours (shared edges and cross-edge continuity)."""

import numpy as np
import pytest

from hexengine import cube, neighbours, transforms
from hexengine.types import CubeState, FaceResult

R = 80.0


def hexagon(centre, radius=R, rot_deg=0.0) -> np.ndarray:
    """Pointy-top regular hexagon in image px (y down), vertex k at -90 + 60k (+ rot) degrees."""
    ang = np.deg2rad(-90.0 + 60.0 * np.arange(6) + rot_deg)
    return np.asarray(centre, float) + radius * np.stack([np.cos(ang), np.sin(ang)], axis=1)


def pair(rot_deg=0.0):
    """Centre cell and its right-hand neighbour, sharing the edge V1-V2 (cell 0) = V5-V4 (cell 1)."""
    c0 = np.array([300.0, 300.0])
    step = np.sqrt(3) * R * np.array([np.cos(np.deg2rad(rot_deg)), np.sin(np.deg2rad(rot_deg))])
    return [hexagon(c0, rot_deg=rot_deg), hexagon(c0 + step, rot_deg=rot_deg)]


def results_with(theta: float, ternary: int, state=CubeState.CONVEX):
    return [FaceResult(face=f, dominant_theta=theta, coherence=0.9, ternary=ternary,
                       radial_alignment=1.0, tangential_alignment=0.0) for f in cube.faces(state)]


def test_snap_points_merges_across_cells_only():
    a, b = pair()
    rng = np.random.default_rng(1)
    b_noisy = b + rng.uniform(-3, 3, b.shape)
    a_in, b_in = a.copy(), b_noisy.copy()
    sa, sb = neighbours.snap_points([a, b_noisy], tol_px=8.0)
    np.testing.assert_array_equal(a, a_in)
    np.testing.assert_array_equal(b_noisy, b_in)
    np.testing.assert_allclose(sa[1], sb[5])
    np.testing.assert_allclose(sa[2], sb[4])
    np.testing.assert_allclose(sa[1], (a[1] + b_noisy[5]) / 2)
    # unshared vertices untouched
    np.testing.assert_array_equal(sa[0], a[0])
    np.testing.assert_array_equal(sb[0], b_noisy[0])


def test_find_shared_edges_single_edge():
    fits = [transforms.fit_hexagon(p) for p in pair()]
    edges = neighbours.find_shared_edges(fits)
    assert len(edges) == 1
    e = edges[0]
    assert (e.cell_a, e.cell_b) == (0, 1)
    assert e.edge_a == (1, 2)
    assert e.edge_b == (4, 5)
    np.testing.assert_allclose(e.image_segment, fits[0].image_points[[1, 2]], atol=1e-9)


def test_find_shared_edges_none_when_apart():
    a, b = pair()
    fits = [transforms.fit_hexagon(a), transforms.fit_hexagon(b + [40.0, 0.0])]
    assert neighbours.find_shared_edges(fits) == []


@pytest.mark.parametrize("rot_deg", [0.0, 12.0])
def test_continuity_same_orientation_scores_one(rot_deg):
    fits = [transforms.fit_hexagon(p) for p in pair(rot_deg)]
    edges = neighbours.find_shared_edges(fits)
    assert len(edges) == 1
    theta = 0.3
    fr = [results_with(theta, 1), results_with(theta, 1)]
    out = neighbours.edge_continuity(fits, fr, CubeState.CONVEX, edges, kind="similarity")
    assert len(out) == 1
    c = out[0]
    assert c.score == pytest.approx(1.0, abs=1e-6)
    assert c.delta_deg == pytest.approx(0.0, abs=1e-3)
    # canonical -> image adds the cluster rotation
    expected = (theta + np.deg2rad(rot_deg)) % np.pi
    assert np.degrees(abs(c.theta_a_image - expected)) < 0.1
    assert c.relation == "same"


def test_continuity_perpendicular_scores_low_and_flipped():
    fits = [transforms.fit_hexagon(p) for p in pair()]
    edges = neighbours.find_shared_edges(fits)
    fr = [results_with(0.3, 1), results_with(0.3 + np.pi / 2, -1)]
    out = neighbours.edge_continuity(fits, fr, CubeState.CONVEX, edges, kind="affine")
    c = out[0]
    assert c.score < 1e-3
    assert c.delta_deg == pytest.approx(90.0, abs=0.1)
    assert c.relation == "flipped"


def test_continuity_uses_adjacent_face():
    # Give only the edge-adjacent face in each cell a distinctive orientation; others are perpendicular.
    state = CubeState.CONVEX
    fits = [transforms.fit_hexagon(p) for p in pair()]
    e = neighbours.find_shared_edges(fits)[0]
    fr = []
    for cell, edge in ((0, e.edge_a), (1, e.edge_b)):
        k = cube.face_for_outer_edge(state, edge[0])
        res = results_with(1.0 + np.pi / 2, 0, state)
        res[k] = FaceResult(face=res[k].face, dominant_theta=1.0, coherence=0.9, ternary=1,
                            radial_alignment=1.0, tangential_alignment=0.0)
        fr.append(res)
    c = neighbours.edge_continuity(fits, fr, state, [e])[0]
    assert c.score == pytest.approx(1.0, abs=1e-6)
    assert c.relation == "same"


def test_cluster_summary():
    fits = [transforms.fit_hexagon(p) for p in pair()]
    edges = neighbours.find_shared_edges(fits)
    a = neighbours.edge_continuity(fits, [results_with(0.3, 1), results_with(0.3, 1)], CubeState.CONVEX, edges)
    b = neighbours.edge_continuity(fits, [results_with(0.3, 1), results_with(0.3 + np.pi / 2, 0)],
                                   CubeState.CONVEX, edges)
    s = neighbours.cluster_summary(a + b)
    assert s["n_edges"] == 2 and s["n_same"] == 1 and s["n_unrelated"] == 1 and s["n_flipped"] == 0
    assert s["mean_score"] == pytest.approx(0.5, abs=1e-3)
    assert neighbours.cluster_summary([])["n_edges"] == 0
