"""The four canonical cube geodesic / manifold path templates, built in 3D and projected isometrically.

OWNER: Agent B (cube & manifolds).

How a template is made
----------------------
1. A template is a list of 3D "waypoints" on the surface of the cube [-1, 1]^3. Consecutive waypoints always
   lie on a common face, so the straight segment between them stays on the cube surface.
2. Every segment is subdivided densely (SAMPLES_PER_SEGMENT points), so each little piece lies on one face.
3. The 3D points are projected to the canonical hexagon with cube.project (see cube.py for the projection).
4. Each little piece is classified visible / hidden with cube.is_visible_3d for the chosen Necker state, and
   consecutive pieces with the same flag are joined into one Polyline. Hidden runs come out as
   Polyline(hidden=True) and are drawn dashed.

The 3D paths themselves do NOT depend on the Necker state: they are fixed curves on a fixed cube. Flipping the
state only swaps which three faces are visible (cube.near_corner), so the same path is cut into different
solid / dashed runs: that is how the toggle changes the overlay. ("Near"/"far" corner in the definitions
below means near/far in the default CONVEX reading, i.e. the corners (1,1,1) / (-1,-1,-1).)

Working definitions (the spec names these but does not define them — each lives in its own function below so
Clare and Ben can edit it):
1. AROUND_CUBE        — the 6-edge-midpoint "belt" (equator) perpendicular to the viewing diagonal
                         (1,1,1); a closed skew hexagon through the midpoints of the six edges not touching
                         the two diagonal corners. Projects onto the edge-midpoint hexagon.
2. AROUND_VERTEX      — closed loop around the NEAR corner (1,1,1) through its three faces: face centre ->
                         shared edge midpoint -> next face centre ... (arrows, clockwise on screen).
                         Solid when CONVEX, dashed when CONCAVE.
3. REVERSE_AROUND_VERTEX — the same loop around the FAR corner (-1,-1,-1), traversed counter-clockwise
                         (arrows). Dashed when CONVEX, solid when CONCAVE.
4. ACROSS_SIX_FACES   — a continuous path visiting all six face centres, each step crossing a shared edge at
                         its midpoint (a Hamiltonian cycle of the face-adjacency graph), closed.
"""

from __future__ import annotations

import numpy as np

from . import cube
from .types import CubeState, Polyline

AROUND_CUBE = "Around the cube"
AROUND_VERTEX = "Around a vertex"
REVERSE_AROUND_VERTEX = "Reverse around a vertex"
ACROSS_SIX_FACES = "Across six faces"
TEMPLATE_NAMES = [AROUND_CUBE, AROUND_VERTEX, REVERSE_AROUND_VERTEX, ACROSS_SIX_FACES]

SAMPLES_PER_SEGMENT = 20   # sub-steps per straight surface segment


# ---------------------------------------------------------------------------------------------------------
# Small 3D helpers
# ---------------------------------------------------------------------------------------------------------

def _axis(i: int, sign: float = 1.0) -> np.ndarray:
    """sign * unit vector along axis i (0 = x, 1 = y, 2 = z). A face is named by (axis, sign): the face
    x = +1 is (0, +1); its centre is exactly _axis(0, +1)."""
    v = np.zeros(3)
    v[i] = sign
    return v


def _signed_area_2d(points: np.ndarray) -> float:
    """Shoelace sum in y-down canonical coords. POSITIVE = clockwise on screen (like V0..V5)."""
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _orient_on_screen(waypoints: np.ndarray, clockwise: bool) -> np.ndarray:
    """Reverse a closed waypoint loop if needed so its projection winds the requested way on screen."""
    area = _signed_area_2d(cube.project(waypoints))
    if (area > 0) != clockwise:
        waypoints = waypoints[::-1].copy()
    return waypoints


# ---------------------------------------------------------------------------------------------------------
# Template definitions (3D waypoints).  EDIT THESE to change what the overlays mean.
# Each returns (waypoints (M, 3), closed, arrows). For closed paths the first waypoint is NOT repeated.
# ---------------------------------------------------------------------------------------------------------

def _around_cube_waypoints(state: CubeState):
    """The belt: midpoints of the six silhouette edges, in order V0V1, V1V2, ..., V5V0.

    These six cube edges are exactly those that touch neither (1,1,1) nor (-1,-1,-1); their midpoints lie on
    the plane x + y + z = 0 (perpendicular to the viewing diagonal). Consecutive midpoints share a face, and
    the faces alternate between the two Necker face-sets, so the belt is half solid / half dashed in either
    state. By linearity it projects exactly onto canonical.edge_midpoints().
    """
    corners = cube.silhouette_corners_3d()
    mids = 0.5 * (corners + np.roll(corners, -1, axis=0))
    return mids, True, False


def _loop_around_corner(corner: np.ndarray) -> np.ndarray:
    """Loop through the three faces meeting at `corner`:
    centre(face x) -> midpoint(edge x|y) -> centre(face y) -> midpoint(edge y|z) -> centre(face z) -> midpoint(edge z|x).

    The face of axis i through the corner is coordinate i = corner[i]; its centre is corner[i] * e_i. The edge
    shared by faces i and j is the cube edge leaving the corner along the third axis; its midpoint is
    corner[i] e_i + corner[j] e_j.
    """
    pts = []
    for i in range(3):
        j = (i + 1) % 3
        pts.append(_axis(i, corner[i]))                           # face centre
        pts.append(_axis(i, corner[i]) + _axis(j, corner[j]))     # shared-edge midpoint
    return np.array(pts)


# "The vertex" of the vertex templates is a FIXED 3D corner: the one that is near in the default CONVEX reading.
# (If we followed whichever corner is near in the current state, the toggle could never change these overlays:
# the loops around (1,1,1) and (-1,-1,-1) project onto the very same 2D hexagon. Fixing the 3D corner means
# the Necker toggle turns "around a vertex" from solid to dashed and "reverse around" from dashed to solid.)
THE_VERTEX = cube.near_corner(CubeState.CONVEX)          # (1, 1, 1)


def _around_vertex_waypoints(state: CubeState):
    """Closed loop around THE_VERTEX (near corner when CONVEX), clockwise on screen, with arrows."""
    loop = _loop_around_corner(THE_VERTEX)
    return _orient_on_screen(loop, clockwise=True), True, True


def _reverse_around_vertex_waypoints(state: CubeState):
    """Same loop around the opposite corner (far corner when CONVEX), counter-clockwise on screen, with arrows."""
    loop = _loop_around_corner(-THE_VERTEX)
    return _orient_on_screen(loop, clockwise=False), True, True


# Order in which the six faces are visited, as (axis, sign). Consecutive faces (and last -> first) must be
# adjacent, i.e. never opposite. This order visits +x, +y, +z then -x, -y, -z, so in either Necker state
# one half of the path is solid and the other half dashed.
SIX_FACE_ORDER = [(0, 1.0), (1, 1.0), (2, 1.0), (0, -1.0), (1, -1.0), (2, -1.0)]


def _across_six_faces_waypoints(state: CubeState):
    """Closed Hamiltonian cycle through all six face centres, crossing each shared edge at its midpoint."""
    pts = []
    n = len(SIX_FACE_ORDER)
    for k in range(n):
        (ai, si), (aj, sj) = SIX_FACE_ORDER[k], SIX_FACE_ORDER[(k + 1) % n]
        assert ai != aj, "consecutive faces in SIX_FACE_ORDER must be adjacent (not opposite)"
        pts.append(_axis(ai, si))                    # face centre
        pts.append(_axis(ai, si) + _axis(aj, sj))    # midpoint of the shared edge
    return np.array(pts), True, False


_DEFINITIONS = {
    AROUND_CUBE: _around_cube_waypoints,
    AROUND_VERTEX: _around_vertex_waypoints,
    REVERSE_AROUND_VERTEX: _reverse_around_vertex_waypoints,
    ACROSS_SIX_FACES: _across_six_faces_waypoints,
}


# ---------------------------------------------------------------------------------------------------------
# Sampling, projection and visible / hidden splitting
# ---------------------------------------------------------------------------------------------------------

def _densify(waypoints: np.ndarray, closed: bool, n: int = SAMPLES_PER_SEGMENT) -> np.ndarray:
    """Subdivide each straight segment into n steps. Returns the points including both ends (for a closed
    path the start point is repeated at the end)."""
    pts = np.vstack([waypoints, waypoints[:1]]) if closed else waypoints
    t = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
    dense = [a + t * (b - a) for a, b in zip(pts[:-1], pts[1:])]
    return np.vstack(dense + [pts[-1:]])


def _split_runs(dense_3d: np.ndarray, state: CubeState) -> list[tuple[bool, np.ndarray]]:
    """Cut the dense path into runs of equal visibility. A piece (between two consecutive samples) is judged
    by its midpoint, which lies inside a single face; runs share their boundary sample so they join up."""
    piece_mid = 0.5 * (dense_3d[:-1] + dense_3d[1:])
    vis = cube.is_visible_3d(piece_mid, state)
    runs = []
    start = 0
    for k in range(1, len(vis) + 1):
        if k == len(vis) or vis[k] != vis[start]:
            runs.append((bool(vis[start]), dense_3d[start:k + 1]))
            start = k
    return runs


def template(name: str, state: CubeState) -> list[Polyline]:
    """Polylines (canonical units) for one named template in the given Necker state."""
    if name not in _DEFINITIONS:
        raise KeyError(f"unknown template {name!r}; choose from {TEMPLATE_NAMES}")
    state = CubeState(state)
    waypoints, closed, arrows = _DEFINITIONS[name](state)
    dense = _densify(np.asarray(waypoints, dtype=float), closed)
    runs = _split_runs(dense, state)

    # A closed loop that starts and ends in the same kind of run: glue the last run onto the first.
    if closed and len(runs) > 1 and runs[0][0] == runs[-1][0]:
        runs[0] = (runs[0][0], np.vstack([runs[-1][1], runs[0][1][1:]]))
        runs.pop()

    whole_loop = closed and len(runs) == 1
    out = []
    for visible, pts in runs:
        pts2d = cube.project(pts)
        if whole_loop:
            pts2d = pts2d[:-1]   # closed=True: the renderer joins last -> first itself
        out.append(Polyline(name=name, points=pts2d, hidden=not visible, closed=whole_loop, arrows=arrows))
    return out


def all_templates(state: CubeState) -> dict[str, list[Polyline]]:
    return {name: template(name, state) for name in TEMPLATE_NAMES}
