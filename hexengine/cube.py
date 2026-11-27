"""Isometric cube model and the two Necker (dual) interpretations.

OWNER: Agent B (cube & manifolds). See CONTRACTS.md "Necker / cube states".

The isometric projection
------------------------
We model the cube as [-1, 1]^3 and look at it straight down its body diagonal (1, 1, 1). A parallel
(orthographic) projection along that diagonal throws away the depth coordinate, so the two diagonal
corners (1, 1, 1) and (-1, -1, -1) both land on the centre of the picture, and the other six corners form a
perfect regular hexagon: the cube's silhouette. We scale the projection so that this hexagon is exactly the
canonical hexagon of canonical.py (circumradius 1, y DOWN, V0 at the top, clockwise on screen).

The projection is one fixed 2x3 matrix P (see isometric_projection). Each 3D axis becomes one 2D direction:

    x axis (1,0,0) -> ( sqrt(3)/4,  1/4)   down-right
    y axis (0,1,0) -> (-sqrt(3)/4,  1/4)   down-left
    z axis (0,0,1) -> ( 0,         -1/2)   straight up  (y is down, so -1/2 means "up")

and the three directions sum to zero, which is exactly why the (1,1,1) diagonal collapses to the origin.

Corner table (3D corner -> canonical vertex):

    V0 top          <- (-1, -1,  1)
    V1 upper right  <- ( 1, -1,  1)
    V2 lower right  <- ( 1, -1, -1)
    V3 bottom       <- ( 1,  1, -1)
    V4 lower left   <- (-1,  1, -1)
    V5 upper left   <- (-1,  1,  1)
    centre (origin) <- ( 1,  1,  1)  and  (-1, -1, -1)

Opposite hexagon vertices (V_k and V_{k+3}) are opposite cube corners (negatives of each other).

The Necker duality
------------------
Because the projection forgets depth, a flat hexagon with an interior "Y" can be read two ways:

* CONVEX (vertex-out, sign +1). The centre is the corner (1, 1, 1) and it pokes TOWARDS the viewer. Its three
  edges go to (1,-1,1), (1,1,-1), (-1,1,1) = V1, V3, V5: the upright "Y". The visible faces are z=+1 (TOP),
  x=+1 (RIGHT) and y=+1 (LEFT).

* CONCAVE (vertex-in, sign -1). The centre is read as the corner (-1, -1, -1). Its three edges go to V0, V2, V4:
  the inverted "Y". The faces drawn solid are x=-1 (UPPER_LEFT), y=-1 (UPPER_RIGHT) and z=-1 (BOTTOM).
  Two equivalent ways to picture this, which give the SAME drawing:
    - looking into the inside corner of an open box from the (1,1,1) side: the centre is the far, interior
      corner and we see the inner surfaces of the three walls that meet there; or
    - depth reversed: the cube seen from the (-1,-1,-1) side, so (-1,-1,-1) is the nearest corner.
  In code we therefore say near_corner(CONCAVE) = (-1, -1, -1): "the corner whose three faces are drawn
  solid". Everything else (which faces/paths are visible) follows from that single choice.

The 2D face rhombi listed in CONTRACTS.md are exactly the projections of those 3D faces; the tests check this.
"""

from __future__ import annotations

import numpy as np

from . import canonical
from .types import CubeState, Face

# Tolerance for "this coordinate is exactly on the face plane" (points are on the cube surface).
_SURFACE_TOL = 1e-9

# Face names and their outer vertex ids (in contract order: the polygon is centre, then these three).
_FACE_TABLE: dict[CubeState, list[tuple[str, tuple[int, int, int]]]] = {
    CubeState.CONVEX: [
        ("TOP", (5, 0, 1)),
        ("RIGHT", (1, 2, 3)),
        ("LEFT", (3, 4, 5)),
    ],
    CubeState.CONCAVE: [
        ("UPPER_RIGHT", (0, 1, 2)),
        ("BOTTOM", (2, 3, 4)),
        ("UPPER_LEFT", (4, 5, 0)),
    ],
}


# ---------------------------------------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------------------------------------

def isometric_projection() -> np.ndarray:
    """2x3 matrix P projecting 3D cube coordinates (cube = [-1, 1]^3) to canonical units (y down).

    The six silhouette corners land exactly on canonical V0..V5 and both diagonal corners land on the origin:

        V0 <- (-1,-1, 1)   V1 <- ( 1,-1, 1)   V2 <- ( 1,-1,-1)
        V3 <- ( 1, 1,-1)   V4 <- (-1, 1,-1)   V5 <- (-1, 1, 1)
        origin <- (1,1,1) and (-1,-1,-1)

    Derivation: write P's columns as px, py, pz (the images of the unit axes). From
    P(1,-1,1)=V1, P(-1,1,1)=V5, P(1,1,-1)=V3 we get 2pz = V1+V5, 2px = V1+V3, 2py = V5+V3.
    The silhouette circumradius is then exactly 1 (it IS the canonical hexagon).
    """
    s = np.sqrt(3.0) / 4.0
    return np.array([[s, -s, 0.0],
                     [0.25, 0.25, -0.5]])


def project(points_3d: np.ndarray) -> np.ndarray:
    """(N, 3) -> (N, 2) canonical units using isometric_projection()."""
    pts = np.atleast_2d(np.asarray(points_3d, dtype=float))
    return pts @ isometric_projection().T


def cube_corners_3d() -> np.ndarray:
    """(8, 3) corners of [-1, 1]^3 (all sign combinations, x varying slowest)."""
    signs = (-1.0, 1.0)
    return np.array([[x, y, z] for x in signs for y in signs for z in signs])


def silhouette_corners_3d() -> np.ndarray:
    """(6, 3) 3D corners in canonical order: row k projects to canonical V_k (see the corner table)."""
    return np.array([
        [-1, -1, 1],   # V0
        [1, -1, 1],    # V1
        [1, -1, -1],   # V2
        [1, 1, -1],    # V3
        [-1, 1, -1],   # V4
        [-1, 1, 1],    # V5
    ], dtype=float)


# ---------------------------------------------------------------------------------------------------------
# Necker state
# ---------------------------------------------------------------------------------------------------------

def near_corner(state: CubeState) -> np.ndarray:
    """3D corner (3,) whose three faces are drawn solid in this state.

    CONVEX  -> (1, 1, 1): its edges project to the spokes centre->V1, V3, V5 (the corner pokes out at us).
    CONCAVE -> (-1, -1, -1): its edges project to centre->V0, V2, V4. Read as the interior (far) corner of an
               open box, or equivalently as the near corner with depth reversed; the drawing is identical.
    """
    return np.full(3, float(CubeState(state).sign))


def far_corner(state: CubeState) -> np.ndarray:
    """The diagonally opposite corner to near_corner(state) (its faces are the hidden ones)."""
    return -near_corner(state)


def spoke_ids(state: CubeState) -> tuple[int, int, int]:
    """(1, 3, 5) for CONVEX, (0, 2, 4) for CONCAVE."""
    return (1, 3, 5) if CubeState(state) is CubeState.CONVEX else (0, 2, 4)


# ---------------------------------------------------------------------------------------------------------
# Faces (2D rhombi) and the 3D faces they come from
# ---------------------------------------------------------------------------------------------------------

def _orientation(vec: np.ndarray) -> float:
    """Orientation of an undirected line in [0, pi) (atan2 in y-down coords)."""
    theta = float(np.arctan2(vec[1], vec[0])) % np.pi
    return 0.0 if np.isclose(theta, np.pi) else theta


def faces(state: CubeState) -> list[Face]:
    """The three visible faces as in CONTRACTS.md, with radial/tangential orientations filled in.

    polygon        = centre, then the three outer vertices (the middle one is opposite the centre)
    radial_theta   = orientation of the long diagonal centre -> middle outer vertex
    tangential     = the perpendicular (short diagonal, wrapping around the centre vertex)
    axis_ids       = the two spoke vertex ids bounding the face (first and last outer vertex)
    """
    out = []
    for name, ids in _FACE_TABLE[CubeState(state)]:
        polygon = np.vstack([canonical.CENTRE, canonical.VERTICES[list(ids)]])
        radial = _orientation(canonical.VERTICES[ids[1]] - canonical.CENTRE)
        tangential = (radial + np.pi / 2.0) % np.pi
        out.append(Face(name=name, polygon=polygon, outer_vertex_ids=ids,
                        radial_theta=radial, tangential_theta=float(tangential),
                        axis_ids=(ids[0], ids[2])))
    return out


def faces_3d(state: CubeState) -> list[np.ndarray]:
    """The 3D faces matching faces(state), same order: each (4, 3) = near corner, then the 3D corners of
    the three outer vertices. project(faces_3d(s)[i]) equals faces(s)[i].polygon."""
    corners = silhouette_corners_3d()
    nc = near_corner(state)
    return [np.vstack([nc, corners[list(ids)]]) for _, ids in _FACE_TABLE[CubeState(state)]]


def face_for_outer_edge(state: CubeState, i: int) -> int:
    """Index into faces(state) of the face containing outer edge (V_i, V_{i+1 mod 6})."""
    i %= 6
    edge = {i, (i + 1) % 6}
    for idx, (_, ids) in enumerate(_FACE_TABLE[CubeState(state)]):
        if edge <= set(ids) and abs(ids.index(i) - ids.index((i + 1) % 6)) == 1:
            return idx
    raise ValueError(f"no face contains outer edge {i}")  # cannot happen for a valid state


# ---------------------------------------------------------------------------------------------------------
# Edges for drawing
# ---------------------------------------------------------------------------------------------------------

def visible_edges(state: CubeState) -> list[np.ndarray]:
    """All visible cube edges as (2, 2) canonical segments (outline + 3 spokes)."""
    v = canonical.VERTICES
    outline = [np.stack([v[k], v[(k + 1) % 6]]) for k in range(6)]
    spokes = [np.stack([canonical.CENTRE, v[k]]) for k in spoke_ids(state)]
    return outline + spokes


def hidden_edges(state: CubeState) -> list[np.ndarray]:
    """The three hidden edges (the other spoke set), for dashed drawing."""
    other = CubeState.CONCAVE if CubeState(state) is CubeState.CONVEX else CubeState.CONVEX
    return [np.stack([canonical.CENTRE, canonical.VERTICES[k]]) for k in spoke_ids(other)]


# ---------------------------------------------------------------------------------------------------------
# Visibility of 3D surface points
# ---------------------------------------------------------------------------------------------------------

def is_visible_3d(points_3d: np.ndarray, state: CubeState) -> np.ndarray:
    """Boolean (N,) — True if a point on the cube SURFACE lies on a face visible in this state
    (i.e. on one of the three faces meeting at near_corner(state)).

    A point is on the face "x = n_x" if its x coordinate equals the near corner's x, and so on. Points on an
    edge between a visible and a hidden face count as visible.
    """
    pts = np.atleast_2d(np.asarray(points_3d, dtype=float))
    nc = near_corner(state)
    return np.any(np.abs(pts - nc) <= _SURFACE_TOL, axis=1)
