"""Automatic cell finding for wall-type images (honeycomb, foam, basalt joints, turtle seams).

Instead of hand-marking one hexagon, find every cell that walls separate, count its sides, and return the ones
with six sides as rough corners for refine.refine_hexagon.

How:
1. A ridge map (refine.feature_map) is bright on the walls' centrelines.
2. Cell interiors = connected regions well away from walls (ridge below a threshold, then eroded).
3. Watershed floods outward from each interior over the ridge map, so neighbouring cells meet exactly on the
   wall centrelines: each cell's region now runs from wall centre to wall centre, like a hand-marked cell.
4. Each region's outline is simplified (Douglas-Peucker) and its corners counted. Cells touching the image
   border are dropped (their outline is cut off by the frame, not by walls).

Polygon side counts are themselves a finding: what share of cells are hexagons at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .refine import feature_map


@dataclass
class Cell:
    label: int
    n_sides: int
    corners: np.ndarray          # (n_sides, 2) simplified outline corners, image px
    area: float
    centroid: np.ndarray         # (2,)
    touches_border: bool


def _simplify(contour: np.ndarray, frac: float) -> np.ndarray:
    peri = cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, frac * peri, True).reshape(-1, 2).astype(float)


def find_cells(image: np.ndarray, kind: str, scale: float, cell_px: float, wall_quantile: float = 0.75,
               simplify_frac: float = 0.035) -> tuple[list[Cell], np.ndarray]:
    """All cells in a wall-type image. Returns (cells, label image) — label 0 is unassigned/walls.

    scale    about half the wall thickness, px (sets the ridge filter)
    cell_px  typical cell diameter, px: seeds are cell interiors shrunk by a quarter of this, so texture inside
             a cell cannot split it into several seeds, and cells smaller than ~1/3 of it are ignored.
    """
    f = feature_map(image, kind, scale)
    f = cv2.GaussianBlur(f, (0, 0), scale)          # walls become smooth crests; fine texture fades
    f = f / (f.max() + 1e-9)
    interior = (f < np.quantile(f, wall_quantile)).astype(np.uint8)
    # One seed per cell: the deepest interior points (distance-to-wall maxima), at most one per half cell width.
    dist = cv2.distanceTransform(interior, cv2.DIST_L2, 5)
    win = max(3, int(round(0.5 * cell_px)) | 1)
    peaks = (dist == cv2.dilate(dist, np.ones((win, win), np.float32))) & (dist > 0.08 * cell_px)
    markers = np.zeros(f.shape, np.int32)
    for i, (y, x) in enumerate(zip(*np.nonzero(peaks)), start=1):
        cv2.circle(markers, (int(x), int(y)), max(1, int(dist[y, x] * 0.5)), i, -1)
    min_area = 0.1 * cell_px ** 2
    ws_in = cv2.cvtColor(np.clip(f * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    labels = cv2.watershed(ws_in, markers.copy())   # -1 on boundaries
    h, w = labels.shape
    cells: list[Cell] = []
    for lab in np.unique(labels):
        if lab <= 0:
            continue
        mask = (labels == lab).astype(np.uint8)
        # include the watershed boundary pixels around this cell so the outline sits on the wall centreline
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8)) & ((labels == lab) | (labels == -1)).astype(np.uint8)
        area = float(mask.sum())
        if area < min_area:
            continue
        ys, xs = np.nonzero(mask)
        touches = xs.min() <= 2 or ys.min() <= 2 or xs.max() >= w - 3 or ys.max() >= h - 3
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(contours, key=cv2.contourArea)
        corners = _simplify(cnt, simplify_frac)
        cells.append(Cell(label=int(lab), n_sides=len(corners), corners=corners, area=area,
                          centroid=np.array([xs.mean(), ys.mean()]), touches_border=bool(touches)))
    return cells, labels


def cells_from_seeds(image: np.ndarray, kind: str, scale: float, seeds: list[tuple[float, float]],
                     seed_radius: float = 6.0, simplify_frac: float = 0.035) -> tuple[list[Cell], np.ndarray]:
    """Like find_cells, but with one hand-placed seed point inside each cell (label = seed index + 1).

    A seed only needs to be somewhere inside its cell, so it is far easier to place by eye than a corner. The
    watershed then settles each boundary on the strongest wall crest between two seeds. Only cells whose
    neighbours are also seeded get a closed, trustworthy outline: unseeded neighbours leak into seeded ones,
    so seed a ring of cells around the ones you want to analyse.
    """
    f = cv2.GaussianBlur(feature_map(image, kind, scale), (0, 0), scale)
    f = f / (f.max() + 1e-9)
    markers = np.zeros(f.shape, np.int32)
    for i, (x, y) in enumerate(seeds, start=1):
        cv2.circle(markers, (int(round(x)), int(round(y))), int(seed_radius), i, -1)
    ws_in = cv2.cvtColor(np.clip(f * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    labels = cv2.watershed(ws_in, markers)
    h, w = labels.shape
    out = []
    for lab in range(1, len(seeds) + 1):
        region = labels == lab
        if not region.any():
            continue
        mask = (cv2.dilate(region.astype(np.uint8), np.ones((3, 3), np.uint8)) & (region | (labels == -1))).astype(np.uint8)
        ys, xs = np.nonzero(mask)
        touches = xs.min() <= 2 or ys.min() <= 2 or xs.max() >= w - 3 or ys.max() >= h - 3
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        corners = _simplify(max(contours, key=cv2.contourArea), simplify_frac)
        out.append(Cell(label=lab, n_sides=len(corners), corners=corners, area=float(mask.sum()),
                        centroid=np.array([xs.mean(), ys.mean()]), touches_border=bool(touches)))
    return out, labels


def adjacency(cells: list[Cell], labels: np.ndarray) -> set[tuple[int, int]]:
    """Pairs of cell labels that share a wall (both sides of a watershed boundary pixel)."""
    pairs = set()
    h, w = labels.shape
    ys, xs = np.nonzero(labels == -1)
    keep = {c.label for c in cells}
    for y, x in zip(ys, xs):
        nb = labels[max(0, y - 1):min(h, y + 2), max(0, x - 1):min(w, x + 2)].ravel()
        ls = sorted({int(v) for v in nb if v > 0 and int(v) in keep})
        for i in range(len(ls)):
            for j in range(i + 1, len(ls)):
                pairs.add((ls[i], ls[j]))
    return pairs
