"""Fitting observed six-point hexagons to the canonical hexagon, and warping images into the canonical frame.

OWNER: Agent A (geometry). See CONTRACTS.md for coordinate and ordering conventions.

The maths in brief
------------------
We have six clicked image points p_0..p_5 (pixels) and six canonical vertices V_0..V_5 (canonical units,
circumradius 1). We look for a 3x3 matrix M with  M @ [p_i, 1]  ~=  [V_i, 1]  (homogeneous, y down) for
three families of transform, each with a different number of free parameters ("degrees of freedom", dof):

* similarity  (4 dof): rotation + uniform scale + translation. Only a *regular* hexagon fits exactly.
* affine      (6 dof): adds shear and non-uniform scale. Any parallel (orthographic) view of a regular
                       hexagon fits exactly.
* homography  (8 dof): adds perspective. Any pinhole-camera view of a flat regular hexagon fits exactly.

Six points give 12 equations (an x and a y for each point). So the affine fit has 12 - 6 = 6 equations
"left over", the homography 12 - 8 = 4 and the similarity 12 - 4 = 8. Because the system is
over-determined, a generic hexagon can NOT be matched exactly: we choose the M that minimises the sum of
squared distances between the mapped points M(p_i) and the canonical V_i (least squares over all six
points, no outlier rejection). Whatever distance remains is the *residual*: the part of the observed shape
that the transform family cannot explain. Residuals are measured in canonical units (1 = circumradius), so
they are independent of image size. See metrics.py for the scores built from them.

Fitting methods
---------------
* affine: ordinary linear least squares (the problem is linear in the 6 unknowns).
* similarity: written with complex numbers, w = a*z + b (z = x + iy, a = scale*e^{i*rotation}), which is
  again linear in the 2 complex unknowns a, b, so ordinary least squares gives the exact optimum.
* homography: the perspective divide makes the problem non-linear. We start from the normalised DLT
  (Hartley's Direct Linear Transform, which minimises an *algebraic* error) and then refine it with
  Levenberg-Marquardt (scipy.optimize.least_squares) so that it minimises the same *geometric* error as
  the other two (distance in canonical units). That keeps the three residuals directly comparable.
"""

from __future__ import annotations

import numpy as np

from . import canonical
from .types import HexFit, TransformKind

# Grey used for "outside the hexagon" pixels in warped images.
OUTSIDE_GREY = (30, 30, 30)

# A polygon whose |area| is below this fraction of (largest extent)^2 is treated as degenerate
# (e.g. all six clicks on a line, or all on the same spot).
_DEGENERATE_AREA_FRACTION = 1e-6


# --------------------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------------------

def _as_points(points: np.ndarray) -> np.ndarray:
    """Validate and return a (6, 2) float64 copy of `points`."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape != (6, 2):
        raise ValueError(f"expected exactly 6 points of shape (6, 2), got shape {pts.shape}")
    if not np.all(np.isfinite(pts)):
        raise ValueError("points contain NaN or infinite values")
    return pts.copy()


def signed_area(points: np.ndarray) -> float:
    """Signed shoelace area of a closed polygon in y-DOWN coordinates.

    area = 1/2 * sum_i (x_i * y_{i+1} - x_{i+1} * y_i)

    With y pointing down, a polygon traversed CLOCKWISE ON SCREEN has POSITIVE area (the canonical hexagon
    V0..V5 has area +2.598). Counter-clockwise on screen gives a negative area.
    """
    pts = np.asarray(points, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _wrap_angle(a: np.ndarray | float) -> np.ndarray | float:
    """Wrap angle(s) into (-pi, pi]."""
    return np.pi - np.mod(np.pi - np.asarray(a), 2.0 * np.pi)


# --------------------------------------------------------------------------------------------------------
# Ordering
# --------------------------------------------------------------------------------------------------------

def order_points(points: np.ndarray, rotation_offset: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (ordered_points (6,2), order (6,)) following the CONTRACTS.md ordering rule.

    Accepts clicks in either winding and starting at any vertex. Raises ValueError unless exactly 6 points.

    The clicks are assumed to go *around* the hexagon (consecutive clicks are neighbouring vertices); only
    the direction and starting vertex are free. Steps:

    1. Winding: compute the signed shoelace area (y down). If it is negative the clicks ran
       counter-clockwise on screen, so the list is reversed to make it clockwise, like V0..V5.
       Near-zero area (collinear / coincident clicks) raises ValueError.
    2. Start vertex: for each point take its angle from the centroid of the six points, atan2(dy, dx) in
       y-down coordinates. The point whose angle is closest to -90 deg (straight up on screen) becomes
       index 0. An exact tie (e.g. a flat-topped hexagon) goes to the leftmost of the tied points.
    3. rotation_offset (taken modulo 6): the clockwise list from step 2 is rolled so that its element at
       position `offset` becomes V0, i.e. ordered[k] = topfirst[(k + offset) % 6]. offset = 1 therefore
       makes the vertex clockwise-after the topmost one V0, which turns the normalised image 60 deg
       anticlockwise. An odd offset swaps which interior "Y" the cube reading uses.

    `order[k]` is the index in the caller's ORIGINAL list of the point now labelled V_k, so
    ordered_points == np.asarray(points)[order].
    """
    pts = _as_points(points)
    offset = int(rotation_offset) % 6

    extent = float(np.max(np.ptp(pts, axis=0)))
    area = signed_area(pts)
    if extent == 0.0 or abs(area) < _DEGENERATE_AREA_FRACTION * extent * extent:
        raise ValueError("degenerate hexagon: the six points enclose (almost) no area")

    # 1. winding -> clockwise on screen
    idx = np.arange(6)
    if area < 0:
        idx = idx[::-1]

    # 2. topmost-by-angle point becomes index 0
    centroid = pts.mean(axis=0)
    rel = pts[idx] - centroid
    angles = np.arctan2(rel[:, 1], rel[:, 0])
    dist_to_up = np.abs(_wrap_angle(angles - (-np.pi / 2.0)))
    tied = np.flatnonzero(dist_to_up <= dist_to_up.min() + 1e-9)
    start = int(tied[np.argmin(rel[tied, 0])])  # leftmost among (near-)exact ties
    idx = np.roll(idx, -start)

    # 3. user rotation offset
    order = np.roll(idx, -offset)
    return pts[order], order.astype(int)


# --------------------------------------------------------------------------------------------------------
# Least-squares fits (each returns a 3x3 matrix mapping image px -> canonical units)
# --------------------------------------------------------------------------------------------------------

def _fit_affine(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """6-dof affine: [u, v] = A @ [x, y, 1]. Linear least squares over all points."""
    X = np.column_stack([src, np.ones(len(src))])          # (N, 3)
    P, *_ = np.linalg.lstsq(X, dst, rcond=None)             # (3, 2): X @ P ~= dst
    M = np.eye(3)
    M[:2, :] = P.T
    return M


def _fit_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """4-dof similarity via complex least squares: w = a*z + b.

    a = s*(cos r + i sin r) encodes rotation r and uniform scale s, b the translation. Expanding a*z gives
    the matrix [[Re a, -Im a, Re b], [Im a, Re a, Im b]]. No reflection is possible (and none is needed,
    because order_points already made the winding match the canonical one).
    """
    z = src[:, 0] + 1j * src[:, 1]
    w = dst[:, 0] + 1j * dst[:, 1]
    X = np.column_stack([z, np.ones_like(z)])
    (a, b), *_ = np.linalg.lstsq(X, w, rcond=None)
    return np.array([[a.real, -a.imag, b.real],
                     [a.imag, a.real, b.imag],
                     [0.0, 0.0, 1.0]])


def _normalising_matrix(pts: np.ndarray) -> np.ndarray:
    """Hartley normalisation: move the centroid to 0 and scale so the mean distance from it is sqrt(2).

    Pixel coordinates are in the hundreds, so the DLT matrix would mix entries ~1 with entries ~1e6; this
    conditioning step is what makes the DLT numerically reliable.
    """
    c = pts.mean(axis=0)
    mean_dist = np.mean(np.linalg.norm(pts - c, axis=1))
    s = np.sqrt(2.0) / mean_dist
    return np.array([[s, 0.0, -s * c[0]],
                     [0.0, s, -s * c[1]],
                     [0.0, 0.0, 1.0]])


def _dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Direct Linear Transform for dst ~ H @ src. Each correspondence gives two rows of A @ h = 0;
    h is the right singular vector of A with the smallest singular value."""
    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    return vt[-1].reshape(3, 3)


def _fit_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """8-dof homography: normalised DLT, then geometric-error refinement (Levenberg-Marquardt)."""
    T = _normalising_matrix(src)
    src_n = apply_transform(T, src)

    Hn = _dlt(src_n, dst)
    if abs(Hn[2, 2]) > 1e-12:
        Hn = Hn / Hn[2, 2]

    def residuals(h: np.ndarray) -> np.ndarray:
        H = np.append(h, 1.0).reshape(3, 3)
        return (apply_transform(H, src_n) - dst).ravel()

    # Refine only if the DLT is not already exact (otherwise there is nothing to improve, and we avoid
    # perturbing a perfect solution by solver tolerances).
    if abs(Hn[2, 2]) > 1e-12 and np.max(np.abs(residuals(Hn.ravel()[:8]))) > 1e-12:
        try:
            from scipy.optimize import least_squares

            sol = least_squares(residuals, Hn.ravel()[:8], method="lm", xtol=1e-12, ftol=1e-12, gtol=1e-12)
            if np.all(np.isfinite(sol.x)) and np.sum(sol.fun ** 2) <= np.sum(residuals(Hn.ravel()[:8]) ** 2):
                Hn = np.append(sol.x, 1.0).reshape(3, 3)
        except Exception:  # pragma: no cover - fall back to plain DLT if scipy misbehaves
            pass

    H = Hn @ T
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


def fit_hexagon(points: np.ndarray, rotation_offset: int = 0) -> HexFit:
    """Least-squares fit of affine (6 dof), homography (8 dof) and similarity (4 dof) transforms,
    mapping the six ordered image points onto the six canonical vertices (image px -> canonical units).

    Uses all six points (no RANSAC). Affine and similarity are exact linear least-squares solutions;
    the homography is a normalised DLT refined to minimise the geometric (canonical-unit) error.
    """
    ordered, order = order_points(points, rotation_offset)
    dst = canonical.VERTICES
    return HexFit(
        image_points=ordered,
        order=order,
        rotation_offset=int(rotation_offset) % 6,
        affine=_fit_affine(ordered, dst),
        homography=_fit_homography(ordered, dst),
        similarity=_fit_similarity(ordered, dst),
    )


# --------------------------------------------------------------------------------------------------------
# Applying transforms
# --------------------------------------------------------------------------------------------------------

def apply_transform(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homogeneous transform to (N, 2) points, with perspective divide."""
    M = np.asarray(matrix, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    single = pts.ndim == 1
    pts = np.atleast_2d(pts)
    homog = np.column_stack([pts, np.ones(len(pts))]) @ M.T   # rows are (x', y', w')
    out = homog[:, :2] / homog[:, 2:3]
    return out[0] if single else out


def local_direction_to(matrix: np.ndarray, at: np.ndarray, theta: float, eps: float = 1e-3) -> float:
    """Map an orientation `theta` located at point `at` through `matrix` (finite difference, works for
    homographies) and return the resulting orientation in [0, pi) in the destination space.

    A homography bends directions differently at different places, so we cannot just transform a vector.
    Instead we take two points a tiny step (eps, in source units) either side of `at` along theta, map both,
    and measure the angle of the segment between their images (a central difference of the local Jacobian).
    """
    p = np.asarray(at, dtype=np.float64).reshape(2)
    d = np.array([np.cos(theta), np.sin(theta)])
    a, b = apply_transform(matrix, np.stack([p - eps * d, p + eps * d]))
    delta = b - a
    theta_out = float(np.mod(np.arctan2(delta[1], delta[0]), np.pi))
    # np.mod can round a tiny negative angle up to exactly pi; orientations live in [0, pi).
    return 0.0 if theta_out >= np.pi else theta_out


def warp_to_canonical(image: np.ndarray, fit: HexFit, kind: TransformKind | str = TransformKind.AFFINE,
                      size: int | None = None, mask_outside: bool = True) -> np.ndarray:
    """Warp `image` into the canonical frame (size x size, default canonical.FRAME_SIZE) using the chosen
    transform composed with canonical.canonical_to_frame_matrix(). If mask_outside, pixels outside the
    canonical hexagon are set to a neutral dark grey (30, 30, 30).

    The full chain is  image px --M--> canonical units --F--> frame px,  i.e. the matrix F @ M.
    Affine and similarity use cv2.warpAffine (the bottom row is [0, 0, 1]), the homography uses
    cv2.warpPerspective; both with bilinear interpolation. Areas that fall outside the source image are
    also filled with the dark grey.
    """
    import cv2

    size = canonical.FRAME_SIZE if size is None else int(size)
    kind = TransformKind(kind)
    full = canonical.canonical_to_frame_matrix(size) @ fit.matrix(kind)

    img = np.asarray(image)
    channels = 1 if img.ndim == 2 else img.shape[2]
    border = tuple([OUTSIDE_GREY[0]] * min(max(channels, 1), 4))

    if kind == TransformKind.HOMOGRAPHY:
        out = cv2.warpPerspective(img, full, (size, size), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    else:
        out = cv2.warpAffine(img, full[:2, :], (size, size), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=border)

    if mask_outside:
        out = out.copy()
        out[~canonical.hexagon_mask(size)] = OUTSIDE_GREY[0]
    return out
