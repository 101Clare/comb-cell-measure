"""Canonical reference: a mathematically perfect regular hexagon.

Circumradius 1, centred at the origin, y axis pointing DOWN (image convention), vertex 0 at the top,
vertices running clockwise on screen. This is the silhouette of a unit cube viewed along its body diagonal.
"""

from __future__ import annotations

import numpy as np

FRAME_SIZE = 512                    # canonical raster is FRAME_SIZE x FRAME_SIZE pixels
FRAME_SCALE = 0.45 * FRAME_SIZE     # pixels per canonical unit
CENTRE = np.zeros(2)


def canonical_vertices() -> np.ndarray:
    """Return the six exact vertices, shape (6, 2), V_k at angle -90 deg + 60 deg * k (y down)."""
    angles = np.deg2rad(-90.0 + 60.0 * np.arange(6))
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


VERTICES = canonical_vertices()


def edge_midpoints() -> np.ndarray:
    """Midpoint of edge (V_k, V_{k+1}) for k = 0..5, shape (6, 2)."""
    return 0.5 * (VERTICES + np.roll(VERTICES, -1, axis=0))


def construction_lines() -> dict[str, list[np.ndarray]]:
    """Internal construction lines used by overlays, as lists of (2, 2) segments in canonical units.

    - spokes_odd:  centre -> V1, V3, V5   (convex-cube visible edges, the upright "Y")
    - spokes_even: centre -> V0, V2, V4   (concave-cube edges, the inverted "Y")
    - diagonals:   V_k -> V_{k+3}, the three long diagonals
    - mid_hexagon: the edge-midpoint hexagon (inner ring)
    """
    mids = edge_midpoints()
    return {
        "spokes_odd": [np.stack([CENTRE, VERTICES[k]]) for k in (1, 3, 5)],
        "spokes_even": [np.stack([CENTRE, VERTICES[k]]) for k in (0, 2, 4)],
        "diagonals": [np.stack([VERTICES[k], VERTICES[k + 3]]) for k in range(3)],
        "mid_hexagon": [np.stack([mids[k], mids[(k + 1) % 6]]) for k in range(6)],
    }


def canonical_to_frame_matrix(size: int = FRAME_SIZE) -> np.ndarray:
    """3x3 matrix mapping canonical units -> canonical-frame pixel coordinates."""
    scale = 0.45 * size
    return np.array([[scale, 0.0, size / 2.0],
                     [0.0, scale, size / 2.0],
                     [0.0, 0.0, 1.0]])


def to_frame(points: np.ndarray, size: int = FRAME_SIZE) -> np.ndarray:
    """Convert (N, 2) canonical-unit points to frame pixels."""
    pts = np.asarray(points, dtype=float)
    return pts * (0.45 * size) + size / 2.0


def from_frame(pixels: np.ndarray, size: int = FRAME_SIZE) -> np.ndarray:
    """Convert (N, 2) frame pixels back to canonical units."""
    px = np.asarray(pixels, dtype=float)
    return (px - size / 2.0) / (0.45 * size)


def hexagon_mask(size: int = FRAME_SIZE) -> np.ndarray:
    """Boolean (size, size) mask, True inside the canonical hexagon."""
    import cv2  # local import keeps this module importable without OpenCV for pure maths

    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(to_frame(VERTICES, size)).astype(np.int32)], 1)
    return mask.astype(bool)
