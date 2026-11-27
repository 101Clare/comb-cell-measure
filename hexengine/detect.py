"""Secondary, optional auto-detection of a hexagon outline (suggests six points the user can accept or adjust).

OWNER: Agent E (UI). Manual clicking is the primary input; this is only a convenience, as the spec asks.

Pipeline (plain words)
----------------------
1. **Grey + blur.** Convert to a single grey channel and smooth with a Gaussian so texture and JPEG noise do not
   produce thousands of tiny edges.
2. **Two edge maps.** Canny finds thin, sharp edges; an adaptive threshold finds regions that are lighter/darker
   than their neighbourhood (good for filled cells with soft edges). A small morphological close joins short gaps.
   Contours from both maps are pooled, so whichever works better on a given image wins.
3. **Contours -> hexagon candidates.** For each contour, `cv2.approxPolyDP` simplifies it to a polygon. We try a
   range of tolerances (1 %..8 % of the perimeter) and keep the first simplification with exactly six corners
   that is convex. Tiny contours and ones that fill almost the whole image (the image border) are skipped.
4. **Pick one.** The reference point is the user's click (if given) or else the image centre. Among candidates
   that CONTAIN the reference point we keep those whose outline lies on strong image edges (mean gradient
   magnitude along the sides, at least 70 % of the best; this rejects "halo" outlines from the threshold maps)
   and take the largest of them. If none contain the point, the candidate whose centroid is nearest to it.
5. **Refine corners.** approxPolyDP corners are always actual contour pixels, which can sit a pixel or two off the
   true corner. We fit a straight line (cv2.fitLine) to the middle part of each side and intersect neighbouring
   side lines; each intersection replaces its corner unless it moved suspiciously far.
6. **Order** the six points with transforms.order_points (clockwise on screen, topmost first), the same convention
   as manual clicks.

Returns a (6, 2) float array in image pixels, or None if nothing hexagon-like is found.
"""

from __future__ import annotations

import cv2
import numpy as np

from .transforms import order_points

# Candidate filters, as fractions of the image area.
_MIN_AREA_FRACTION = 0.002   # ignore specks
_MAX_AREA_FRACTION = 0.95    # ignore the image border / whole-frame contours
# approxPolyDP tolerances to try, as fractions of the contour perimeter.
_EPS_FRACTIONS = (0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08)
# Trim this fraction off each end of a side before fitting its line (corners are rounded by blur).
_SIDE_TRIM = 0.2
# A refined corner may move at most this fraction of the shorter adjacent side from the approx corner.
_MAX_REFINE_SHIFT = 0.15


def _grey(image: np.ndarray) -> np.ndarray:
    """RGB / RGBA / grey image -> uint8 grey."""
    img = np.asarray(image)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def _edge_maps(grey: np.ndarray) -> list[np.ndarray]:
    """Binary maps whose contours may outline the hexagon (Canny edges and adaptive threshold, both polarities)."""
    diag = float(np.hypot(*grey.shape[:2]))
    blur_sigma = max(1.0, diag / 600.0)
    blurred = cv2.GaussianBlur(grey, (0, 0), blur_sigma)

    # Canny thresholds from the median intensity (a common automatic rule).
    med = float(np.median(blurred))
    lo, hi = int(max(0, 0.66 * med)), int(min(255, 1.33 * med))
    if hi - lo < 20:
        lo, hi = 30, 90
    edges = cv2.Canny(blurred, lo, hi)
    k = max(3, int(round(diag / 400)) | 1)            # odd kernel, grows with image size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    block = max(11, int(diag / 20) | 1)               # odd neighbourhood size for the adaptive threshold
    adapt = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 2)
    adapt = cv2.morphologyEx(adapt, cv2.MORPH_OPEN, kernel)

    # Global Otsu threshold too: very effective for a single dark/light shape on a plain background.
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return [edges, adapt, 255 - adapt, otsu, 255 - otsu]


def _hexagon_from_contour(contour: np.ndarray) -> np.ndarray | None:
    """First convex 6-corner approxPolyDP simplification of `contour`, as (6, 2) float, or None."""
    peri = cv2.arcLength(contour, True)
    for frac in _EPS_FRACTIONS:
        approx = cv2.approxPolyDP(contour, frac * peri, True)
        if len(approx) == 6 and cv2.isContourConvex(approx):
            return approx.reshape(6, 2).astype(np.float64)
        if len(approx) < 6:
            break  # larger tolerances only remove more corners
    return None


def _gradient_magnitude(grey: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude of a lightly blurred grey image (float32)."""
    g = cv2.GaussianBlur(grey.astype(np.float32), (0, 0), 1.0)
    return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))


def _edge_support(grad: np.ndarray, corners: np.ndarray, samples_per_side: int = 40) -> float:
    """Mean gradient magnitude sampled along the polygon's sides: high when the outline sits on a real edge."""
    h, w = grad.shape[:2]
    t = np.linspace(0.0, 1.0, samples_per_side, endpoint=False)[:, None]
    pts = np.vstack([a + t * (b - a) for a, b in zip(corners, np.roll(corners, -1, axis=0))])
    xs = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
    return float(grad[ys, xs].mean())


def _line_intersection(p1, d1, p2, d2) -> np.ndarray | None:
    """Intersection of lines p1 + s*d1 and p2 + t*d2, or None if (nearly) parallel."""
    a = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]], dtype=float)
    if abs(np.linalg.det(a)) < 1e-9:
        return None
    s, _ = np.linalg.solve(a, np.asarray(p2, float) - np.asarray(p1, float))
    return np.asarray(p1, float) + s * np.asarray(d1, float)


def _refine_corners(contour: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Sub-pixel corners: fit a line to the middle of each side of the contour and intersect neighbours."""
    pts = contour.reshape(-1, 2).astype(np.float64)
    n = len(pts)
    if n < 30:
        return corners
    # Index of the contour pixel nearest to each approx corner.
    idx = [int(np.argmin(np.hypot(*(pts - c).T))) for c in corners]
    lines = []
    for i in range(6):
        a, b = idx[i], idx[(i + 1) % 6]
        seg = pts[a:b + 1] if a <= b else np.vstack([pts[a:], pts[:b + 1]])   # contour is cyclic
        m = len(seg)
        trim = int(m * _SIDE_TRIM)
        core = seg[trim:m - trim] if m - 2 * trim >= 3 else seg
        if len(core) < 2:
            return corners
        vx, vy, x0, y0 = cv2.fitLine(core.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0]), np.array([vx, vy])))
    sides = np.hypot(*(np.roll(corners, -1, axis=0) - corners).T)
    refined = corners.copy()
    for i in range(6):
        # corner i joins side i-1 (from corner i-1 to i) and side i (from corner i to i+1)
        p = _line_intersection(*lines[i - 1], *lines[i])
        limit = _MAX_REFINE_SHIFT * min(sides[i - 1], sides[i])
        if p is not None and np.hypot(*(p - corners[i])) <= limit:
            refined[i] = p
    return refined


def detect_hexagon(image: np.ndarray, near: tuple[float, float] | np.ndarray | None = None) -> np.ndarray | None:
    """Suggest six hexagon corners (image px, (6, 2), ordered like manual clicks) or None.

    `near` is an optional (x, y) pixel position, typically the user's last click; the hexagon around it is
    preferred. Without it the image centre is used.
    """
    grey = _grey(image)
    h, w = grey.shape[:2]
    img_area = float(h * w)
    ref = np.array([w / 2.0, h / 2.0]) if near is None else np.asarray(near, dtype=float).reshape(2)

    candidates = []   # (contour, corners, area)
    for binary in _edge_maps(grey):
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            area = abs(cv2.contourArea(c))
            if not (_MIN_AREA_FRACTION * img_area <= area <= _MAX_AREA_FRACTION * img_area):
                continue
            corners = _hexagon_from_contour(c)
            if corners is not None:
                candidates.append((c, corners, abs(cv2.contourArea(corners.astype(np.float32)))))
    if not candidates:
        return None

    containing = [cand for cand in candidates
                  if cv2.pointPolygonTest(cand[1].astype(np.float32), (float(ref[0]), float(ref[1])), False) >= 0]
    if containing:
        # Threshold maps can also produce "halo" outlines a few px outside the real edge. Keep only candidates
        # whose outline sits on strong image edges (at least 70 % of the best edge support), then the largest.
        grad = _gradient_magnitude(grey)
        support = [_edge_support(grad, cand[1]) for cand in containing]
        top = max(support)
        good = [cand for cand, s in zip(containing, support) if s >= 0.7 * top]
        best = max(good, key=lambda cand: cand[2])
    else:
        best = min(candidates, key=lambda cand: float(np.hypot(*(cand[1].mean(axis=0) - ref))))

    corners = _refine_corners(best[0], best[1])
    try:
        ordered, _ = order_points(corners)
    except ValueError:
        return None
    return ordered
