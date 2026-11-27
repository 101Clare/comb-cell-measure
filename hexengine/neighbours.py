"""Multi-cell (1-ring) neighbour / continuity mode.

OWNER: Agent C (inference).
Cell 0 is the centre hexagon; cells 1..3 are neighbours, each with its own six clicked points and HexFit.

Plain-words guide (for Clare and Ben)
-------------------------------------
In a honeycomb-like tiling, two neighbouring hexagons share an edge. Each hexagon is normalised on its own,
so its faces and their dominant line orientations live in that cell's private canonical frame. To ask
"does the pattern carry on smoothly across the shared edge?" we:

1. find which edges are shared (two clicked vertices of cell a match two of cell b, in either order);
2. in each cell take the analysed region that touches that edge (result_for_outer_edge);
3. carry each face's dominant orientation back into the ORIGINAL photo's pixel space, using the inverse of
   that cell's fitted transform, evaluated at the face centroid (so it also works for homographies);
4. compare the two image-space orientations as undirected lines: delta in [0, 90] deg, score = |cos delta|
   (1 = the lines continue in the same direction, 0 = they cross at right angles).

`relation` compares the two faces' ternary values: 'same' (equal and nonzero), 'flipped' (opposite nonzero
values, i.e. the Necker reading reverses across the edge) or 'unrelated' (either is 0).
"""

from __future__ import annotations

import numpy as np

from . import transforms
from .types import EdgeContinuity, FaceResult, HexFit, SharedEdge, CubeState


def snap_points(cells: list[np.ndarray], tol_px: float = 12.0) -> list[np.ndarray]:
    """Merge clicked vertices from different cells that lie within tol_px of each other (replace each cluster
    by its mean) so shared edges coincide exactly. Returns new (6, 2) arrays, inputs untouched.

    Clusters are connected components of the "within tol_px and in different cells" relation (union-find),
    so a vertex shared by three cells snaps to a single point. Points of the same cell are never linked
    directly to each other.
    """
    arrays = [np.asarray(c, dtype=float).copy() for c in cells]
    flat = [(ci, vi) for ci, arr in enumerate(arrays) for vi in range(len(arr))]
    n = len(flat)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pts = np.array([arrays[ci][vi] for ci, vi in flat]) if n else np.zeros((0, 2))
    for i in range(n):
        for j in range(i + 1, n):
            if flat[i][0] == flat[j][0]:
                continue
            if np.hypot(*(pts[i] - pts[j])) <= tol_px:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    for members in groups.values():
        if len(members) < 2:
            continue
        mean = pts[members].mean(axis=0)
        for k in members:
            ci, vi = flat[k]
            arrays[ci][vi] = mean
    return arrays


def find_shared_edges(fits: list[HexFit], tol_px: float = 12.0) -> list[SharedEdge]:
    """Edges (V_i, V_{i+1}) of cell a and (V_j, V_{j+1}) of cell b whose endpoints match within tol_px
    (in either direction). Only pairs a < b.

    edge_a / edge_b are given in each cell's own (i, i+1 mod 6) order; image_segment is the mean position of
    each matched endpoint pair, in edge_a's order.
    """
    def close(p, q) -> bool:
        return float(np.hypot(*(np.asarray(p) - np.asarray(q)))) <= tol_px

    edges: list[SharedEdge] = []
    for a in range(len(fits)):
        pa = np.asarray(fits[a].image_points, dtype=float)
        for b in range(a + 1, len(fits)):
            pb = np.asarray(fits[b].image_points, dtype=float)
            for i in range(6):
                a0, a1 = pa[i], pa[(i + 1) % 6]
                for j in range(6):
                    b0, b1 = pb[j], pb[(j + 1) % 6]
                    if close(a0, b0) and close(a1, b1):
                        seg = np.stack([(a0 + b0) / 2, (a1 + b1) / 2])
                    elif close(a0, b1) and close(a1, b0):
                        seg = np.stack([(a0 + b1) / 2, (a1 + b0) / 2])
                    else:
                        continue
                    edges.append(SharedEdge(cell_a=a, cell_b=b, edge_a=(i, (i + 1) % 6),
                                            edge_b=(j, (j + 1) % 6), image_segment=seg))
    return edges


def _face_theta_in_image(fit: HexFit, result: FaceResult, kind: str) -> float:
    """Map a face's dominant orientation (canonical frame) to image pixel space at the face centroid.

    The canonical frame is a uniform scale + shift of canonical units, so orientations measured in frame
    pixels are the same angles in canonical units. fit.matrix(kind) maps image -> canonical, so we use its
    inverse (canonical -> image) and push the orientation through it locally (finite difference).
    """
    to_image = np.linalg.inv(fit.matrix(kind))
    centroid = np.asarray(result.face.polygon, dtype=float).mean(axis=0)   # rhombus: vertex mean = centroid
    return float(transforms.local_direction_to(to_image, centroid, result.dominant_theta))


def _relation(ta: int, tb: int) -> str:
    if ta != 0 and tb != 0:
        return "same" if ta == tb else "flipped"
    return "unrelated"


def result_for_outer_edge(results: list[FaceResult], i: int) -> FaceResult:
    """The analysed region that contains outer edge (V_i, V_{i+1}), whichever ternary convention produced it.

    A region's outer vertices (a, b, c) give it the outer edges (a, b) and (b, c).
    """
    j = (i + 1) % 6
    for r in results:
        a, b, c = r.face.outer_vertex_ids
        if {i, j} in ({a, b}, {b, c}):
            return r
    raise ValueError(f"no analysed region contains outer edge V{i}-V{j}")


def edge_continuity(fits: list[HexFit], face_results: list[list[FaceResult]], state: CubeState,
                    edges: list[SharedEdge], kind: str = "affine") -> list[EdgeContinuity]:
    """For each shared edge: take the region adjacent to that edge in each cell (result_for_outer_edge), map its
    dominant orientation from canonical units back to IMAGE space at the region centroid (inverse transform,
    transforms.local_direction_to), compare. relation: 'same' if ternaries equal and nonzero, 'flipped' if they are
    opposite nonzero values, else 'unrelated'. `state` is kept for the API; the regions carry their own geometry."""
    out: list[EdgeContinuity] = []
    for e in edges:
        ra = result_for_outer_edge(face_results[e.cell_a], e.edge_a[0])
        rb = result_for_outer_edge(face_results[e.cell_b], e.edge_b[0])
        ta = _face_theta_in_image(fits[e.cell_a], ra, kind)
        tb = _face_theta_in_image(fits[e.cell_b], rb, kind)
        d = abs(ta - tb) % np.pi
        d = min(d, np.pi - d)                       # undirected difference, [0, pi/2]
        out.append(EdgeContinuity(edge=e, theta_a_image=ta, theta_b_image=tb,
                                  delta_deg=float(np.degrees(d)), score=abs(float(np.cos(d))),
                                  ternary_a=int(ra.ternary), ternary_b=int(rb.ternary),
                                  relation=_relation(int(ra.ternary), int(rb.ternary))))
    return out


def cluster_summary(results: list[EdgeContinuity]) -> dict[str, float]:
    """{'mean_score', 'n_edges', 'n_same', 'n_flipped', 'n_unrelated'}. mean_score is 0.0 when there are no edges."""
    n = len(results)
    return {
        "mean_score": float(np.mean([r.score for r in results])) if n else 0.0,
        "n_edges": n,
        "n_same": sum(r.relation == "same" for r in results),
        "n_flipped": sum(r.relation == "flipped" for r in results),
        "n_unrelated": sum(r.relation == "unrelated" for r in results),
    }
