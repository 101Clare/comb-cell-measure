import numpy as np
import pytest

from hexengine import canonical, cube, manifolds
from hexengine.types import CubeState

STATES = [CubeState.CONVEX, CubeState.CONCAVE]


def _inside_hexagon(points, tol=1e-9):
    """Point-in-hexagon: inside every edge half-plane of the (clockwise, y-down) canonical hexagon."""
    v = canonical.VERTICES
    for k in range(6):
        a, b = v[k], v[(k + 1) % 6]
        e = b - a
        cross = e[0] * (points[:, 1] - a[1]) - e[1] * (points[:, 0] - a[0])
        if np.any(cross < -tol):   # clockwise on screen in y-down -> interior has cross >= 0
            return False
    return True


def _signed_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def _all_points(polys):
    return np.vstack([p.points for p in polys])


def test_inside_helper_sanity():
    assert _inside_hexagon(canonical.VERTICES)
    assert not _inside_hexagon(np.array([[0.0, -1.01]]))


@pytest.mark.parametrize("state", STATES)
@pytest.mark.parametrize("name", manifolds.TEMPLATE_NAMES)
def test_templates_nonempty_and_inside(name, state):
    polys = manifolds.template(name, state)
    assert len(polys) >= 1
    for p in polys:
        assert p.points.ndim == 2 and p.points.shape[1] == 2 and len(p.points) >= 2
        assert p.name == name
        assert _inside_hexagon(p.points)


@pytest.mark.parametrize("state", STATES)
def test_around_cube_on_midpoint_hexagon(state):
    polys = manifolds.template(manifolds.AROUND_CUBE, state)
    mids = canonical.edge_midpoints()
    pts = _all_points(polys)
    # every sample lies on one of the six sides of the edge-midpoint hexagon
    for q in pts:
        ok = False
        for k in range(6):
            a, b = mids[k], mids[(k + 1) % 6]
            t = np.clip(np.dot(q - a, b - a) / np.dot(b - a, b - a), 0, 1)
            if np.linalg.norm(a + t * (b - a) - q) < 1e-9:
                ok = True
                break
        assert ok
    # and it passes through every midpoint
    for m in mids:
        assert np.min(np.linalg.norm(pts - m, axis=1)) < 1e-9
    # alternating faces -> 6 runs, 3 solid and 3 dashed
    assert len(polys) == 6
    assert sum(p.hidden for p in polys) == 3


def test_around_vertex_visibility_and_winding():
    for state in STATES:
        near = manifolds.template(manifolds.AROUND_VERTEX, state)
        far = manifolds.template(manifolds.REVERSE_AROUND_VERTEX, state)
        convex = state is CubeState.CONVEX
        # around (1,1,1): solid when convex, dashed when concave; the reverse loop the other way round
        assert len(near) == 1 and near[0].hidden is (not convex) and near[0].closed and near[0].arrows
        assert len(far) == 1 and far[0].hidden is convex and far[0].closed and far[0].arrows
        assert _signed_area(near[0].points) > 0     # clockwise on screen (y down)
        assert _signed_area(far[0].points) < 0      # counter-clockwise


def test_necker_toggle_changes_split():
    for name in manifolds.TEMPLATE_NAMES:
        a = manifolds.template(name, CubeState.CONVEX)
        b = manifolds.template(name, CubeState.CONCAVE)
        vis_a = np.vstack([p.points for p in a if not p.hidden]) if any(not p.hidden for p in a) else np.empty((0, 2))
        vis_b = np.vstack([p.points for p in b if not p.hidden]) if any(not p.hidden for p in b) else np.empty((0, 2))
        same = vis_a.shape == vis_b.shape and np.allclose(vis_a, vis_b)
        assert not same, name


def test_across_six_faces_visits_all_centres():
    wp, closed, _ = manifolds._across_six_faces_waypoints(CubeState.CONVEX)
    assert closed
    for axis in range(3):
        for s in (-1.0, 1.0):
            c = np.zeros(3)
            c[axis] = s
            assert np.min(np.linalg.norm(wp - c, axis=1)) < 1e-12
    # every waypoint is on the cube surface
    assert np.all(np.isclose(np.max(np.abs(wp), axis=1), 1.0))
    polys = manifolds.template(manifolds.ACROSS_SIX_FACES, CubeState.CONVEX)
    assert any(p.hidden for p in polys) and any(not p.hidden for p in polys)


def test_dense_paths_stay_on_surface():
    for name, fn in manifolds._DEFINITIONS.items():
        for state in STATES:
            wp, closed, _ = fn(state)
            dense = manifolds._densify(np.asarray(wp, float), closed)
            assert np.all(np.isclose(np.max(np.abs(dense), axis=1), 1.0)), name
            # projected path length is consistent with the template output
            assert len(dense) >= manifolds.SAMPLES_PER_SEGMENT


def test_all_templates_keys():
    d = manifolds.all_templates(CubeState.CONVEX)
    assert list(d) == manifolds.TEMPLATE_NAMES
