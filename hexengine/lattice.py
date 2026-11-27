"""Rough hexagons for a regular hexagonal pattern (honeycomb, tumbling blocks) from its repeat lattice.

Marking dozens of cells by hand is slow and inconsistent. A regular comb or tiling repeats along two lattice
vectors, which show up as the strongest off-centre peaks of the image's autocorrelation. From those:
  * the six nearest repeats n_1..n_6 (sorted by angle) are the neighbouring cell centres;
  * the cell's corners sit at (n_k + n_{k+1}) / 3, where three cells meet (the Voronoi vertices of the lattice).
Given one cell centre picked by eye (the anchor), every other cell centre is anchor + i*a1 + j*a2. The corners are
only rough: hexengine.refine then snaps each one onto the real walls, and its quality gate rejects cells where the
lattice has drifted or the wall can't be found.
"""

from __future__ import annotations

import cv2
import numpy as np


def lattice_vectors(image: np.ndarray, min_period_px: float = 15.0, max_period_px: float | None = None,
                    highpass_sigma: float = 25.0, min_peak: float = 0.05):
    """The two shortest independent repeat vectors (2, 2) of a periodic image, from its autocorrelation.

    highpass_sigma removes lighting gradients (keep it above the cell spacing); min_peak is the smallest
    normalised autocorrelation peak accepted (lower it for small or faint patches)."""
    g = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) if image.ndim == 3 else image.astype(np.float32)
    g = g - cv2.GaussianBlur(g, (0, 0), highpass_sigma)   # remove slow lighting gradients
    g *= np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1])).astype(np.float32)
    f = np.fft.fft2(g)
    ac = np.fft.fftshift(np.real(np.fft.ifft2(f * np.conj(f))))
    ac /= ac.max()
    h, w = ac.shape
    cy, cx = h // 2, w // 2
    max_period_px = max_period_px or min(h, w) / 3
    peaks = (ac == cv2.dilate(ac, np.ones((5, 5), np.float32))) & (ac > min_peak)
    ys, xs = np.nonzero(peaks)
    vec = np.stack([xs - cx, ys - cy], axis=1).astype(float)
    r = np.hypot(*vec.T)
    keep = (r >= min_period_px) & (r <= max_period_px)
    vec, r, val = vec[keep], r[keep], ac[ys[keep], xs[keep]]
    order = np.argsort(-val)                             # strongest first
    vec, r = vec[order], r[order]
    if len(vec) < 2:
        raise ValueError("no repeating pattern found")
    a1 = vec[0]
    for v in vec[1:]:
        cross = abs(a1[0] * v[1] - a1[1] * v[0])
        if cross > 0.5 * np.hypot(*a1) * np.hypot(*v):   # clearly not parallel
            a2 = v
            break
    else:
        raise ValueError("could not find two independent lattice vectors")
    # reduce to the shortest basis (Lagrange-Gauss)
    for _ in range(10):
        if np.hypot(*a2) < np.hypot(*a1):
            a1, a2 = a2, a1
        m = np.round(np.dot(a1, a2) / np.dot(a1, a1))
        if m == 0:
            break
        a2 = a2 - m * a1
    return np.array([a1, a2])


def neighbour_vectors(basis: np.ndarray) -> np.ndarray:
    """The six nearest repeat vectors, sorted by angle (y down, so clockwise on screen)."""
    a1, a2 = basis
    cands = [a1, a2, a2 - a1, -a1, -a2, a1 - a2, a1 + a2, -(a1 + a2)]
    cands = sorted(cands, key=lambda v: np.hypot(*v))[:6]
    return np.array(sorted(cands, key=lambda v: np.arctan2(v[1], v[0])))


def cell_corners(centre, basis) -> np.ndarray:
    """Rough six corners (6, 2) of the cell at `centre`: (n_k + n_{k+1}) / 3 around it."""
    n = neighbour_vectors(basis)
    return np.array([np.asarray(centre, float) + (n[k] + n[(k + 1) % 6]) / 3.0 for k in range(6)])


def lattice_cells(anchor, basis, shape, margin_frac: float = 0.15, max_cells: int = 60) -> list[np.ndarray]:
    """Rough hexagons for every lattice cell whose corners all sit inside the image (with a margin)."""
    h, w = shape[:2]
    a1, a2 = basis
    span = int(np.ceil(max(h, w) / min(np.hypot(*a1), np.hypot(*a2)))) + 2
    m = margin_frac * np.hypot(*a1)
    out = []
    for i in range(-span, span + 1):
        for j in range(-span, span + 1):
            c = np.asarray(anchor, float) + i * a1 + j * a2
            corners = cell_corners(c, basis)
            if (corners[:, 0] >= m).all() and (corners[:, 1] >= m).all() and \
               (corners[:, 0] <= w - 1 - m).all() and (corners[:, 1] <= h - 1 - m).all():
                out.append(corners)
    out.sort(key=lambda c: np.hypot(*(c.mean(axis=0) - np.asarray(anchor, float))))
    return out[:max_cells]
