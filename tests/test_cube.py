import numpy as np
import pytest

from hexengine import canonical, cube
from hexengine.types import CubeState

STATES = [CubeState.CONVEX, CubeState.CONCAVE]


def test_projection_shape_and_silhouette_corners():
    P = cube.isometric_projection()
    assert P.shape == (2, 3)
    np.testing.assert_allclose(cube.project(cube.silhouette_corners_3d()), canonical.VERTICES, atol=1e-9)


def test_diagonal_corners_project_to_origin():
    np.testing.assert_allclose(cube.project(np.array([[1, 1, 1], [-1, -1, -1]])), 0.0, atol=1e-9)


def test_all_corners_accounted_for():
    corners = cube.cube_corners_3d()
    assert corners.shape == (8, 3)
    proj = cube.project(corners)
    radii = np.linalg.norm(proj, axis=1)
    assert np.sum(radii < 1e-9) == 2
    assert np.allclose(radii[radii > 1e-9], 1.0)


def test_spokes_and_near_corner():
    assert cube.spoke_ids(CubeState.CONVEX) == (1, 3, 5)
    assert cube.spoke_ids(CubeState.CONCAVE) == (0, 2, 4)
    for state in STATES:
        nc = cube.near_corner(state)
        # the three cube edges leaving the near corner project onto the state's spokes
        ends = [nc.copy() for _ in range(3)]
        for i in range(3):
            ends[i][i] *= -1
        proj = cube.project(np.array(ends))
        targets = canonical.VERTICES[list(cube.spoke_ids(state))]
        for p in proj:
            assert np.min(np.linalg.norm(targets - p, axis=1)) < 1e-9


EXPECTED = {
    CubeState.CONVEX: [("TOP", (5, 0, 1)), ("RIGHT", (1, 2, 3)), ("LEFT", (3, 4, 5))],
    CubeState.CONCAVE: [("UPPER_RIGHT", (0, 1, 2)), ("BOTTOM", (2, 3, 4)), ("UPPER_LEFT", (4, 5, 0))],
}


@pytest.mark.parametrize("state", STATES)
def test_faces_match_contract(state):
    fs = cube.faces(state)
    assert [(f.name, f.outer_vertex_ids) for f in fs] == EXPECTED[state]
    for f in fs:
        ids = f.outer_vertex_ids
        np.testing.assert_allclose(f.polygon[0], 0.0, atol=1e-12)
        np.testing.assert_allclose(f.polygon[1:], canonical.VERTICES[list(ids)], atol=1e-12)
        assert f.axis_ids == (ids[0], ids[2])
        assert set(f.axis_ids) <= set(cube.spoke_ids(state))
        mid = canonical.VERTICES[ids[1]]
        assert 0 <= f.radial_theta < np.pi and 0 <= f.tangential_theta < np.pi
        assert np.isclose(f.radial_theta, np.arctan2(mid[1], mid[0]) % np.pi)
        d = abs(f.radial_theta - f.tangential_theta)
        assert np.isclose(d, np.pi / 2)


def test_top_face_radial_is_vertical():
    top = cube.faces(CubeState.CONVEX)[0]
    assert np.isclose(top.radial_theta, np.pi / 2)
    assert np.isclose(top.tangential_theta, 0.0)


@pytest.mark.parametrize("state", STATES)
def test_3d_faces_project_to_contract_rhombi(state):
    """The 3D faces meeting at near_corner(state) project exactly onto faces(state) polygons."""
    nc = cube.near_corner(state)
    for f, f3 in zip(cube.faces(state), cube.faces_3d(state)):
        np.testing.assert_allclose(f3[0], nc)
        # all four corners share one coordinate with the near corner -> genuinely one cube face
        common = np.all(np.isclose(f3, f3[0]), axis=0)
        assert common.sum() == 1
        assert np.all(cube.is_visible_3d(f3, state))
        np.testing.assert_allclose(cube.project(f3), f.polygon, atol=1e-9)
    # the three faces are the three distinct faces through the near corner
    axes = {int(np.flatnonzero(np.all(np.isclose(f3, f3[0]), axis=0))[0]) for f3 in cube.faces_3d(state)}
    assert axes == {0, 1, 2}


@pytest.mark.parametrize("state", STATES)
def test_face_for_outer_edge(state):
    counts = [0, 0, 0]
    for i in range(6):
        idx = cube.face_for_outer_edge(state, i)
        f = cube.faces(state)[idx]
        assert {i, (i + 1) % 6} <= set(f.outer_vertex_ids)
        counts[idx] += 1
    assert counts == [2, 2, 2]


@pytest.mark.parametrize("state", STATES)
def test_edges(state):
    vis = cube.visible_edges(state)
    hid = cube.hidden_edges(state)
    assert len(vis) == 9 and len(hid) == 3
    for seg in vis + hid:
        assert seg.shape == (2, 2)
    hidden_tips = {int(np.argmin(np.linalg.norm(canonical.VERTICES - s[1], axis=1))) for s in hid}
    assert hidden_tips == set(range(6)) - set(cube.spoke_ids(state))


def test_is_visible_3d():
    pts = np.array([[1, 0, 0], [-1, 0, 0], [0.3, 1, -0.2], [0, 0, -1], [1, -1, 0]])
    np.testing.assert_array_equal(cube.is_visible_3d(pts, CubeState.CONVEX), [True, False, True, False, True])
    np.testing.assert_array_equal(cube.is_visible_3d(pts, CubeState.CONCAVE), [False, True, False, True, True])
