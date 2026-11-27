"""Generate the synthetic test images in samples/ plus samples/ground_truth.json.

Each image has a KNOWN answer, so the maths in hexengine/ can be checked before anyone tries real photos.
Everything is seeded, so running this twice gives byte-identical files.

    python samples/generate_samples.py              # writes into samples/
    python samples/generate_samples.py --out DIR    # writes somewhere else (used by the tests)

Only numpy + OpenCV are used (plus hexengine.canonical for the exact reference vertices), so this script
does not depend on the rest of the engine being finished.

ground_truth.json layout::

    {
      "<filename>.png": {
        "vertices": [[x, y] * 6],     # image pixels, clockwise on screen, starting with the topmost vertex
        "description": "...",
        "image_size": [width, height],
        "expected": {...},            # what the engine should report (qualitative + reference numbers)
        "cells": [...]                # honeycomb only: centre cell + 3 neighbours
      }
    }

The "reference_scores" inside "expected" are computed here with plain least squares on the ground-truth
vertices (canonical units, circumradius 1). They are a sanity reference, not a definition: the engine's
own numbers should agree closely, but small differences in how a score is defined are fine.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hexengine.canonical import VERTICES  # noqa: E402  (exact regular hexagon, V0 at top, clockwise)

W, H = 800, 700          # image width, height in pixels
BASE_SEED = 20261124     # change this and every image changes; keep it fixed for reproducibility


# ----------------------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------------------

def rng_for(index: int) -> np.random.Generator:
    return np.random.default_rng(BASE_SEED + 1000 * index)


def hex_vertices(centre, radius: float, rot_deg: float = 0.0) -> np.ndarray:
    """Regular hexagon in image px: canonical vertices rotated by rot_deg (clockwise on screen), scaled, shifted."""
    t = np.deg2rad(rot_deg)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return VERTICES @ rot.T * radius + np.asarray(centre, dtype=float)


def order_cw_from_top(pts: np.ndarray) -> np.ndarray:
    """Clockwise on screen (y down), index 0 = the vertex whose angle from the centroid is closest to -90 deg.
    This is the same rule transforms.order_points() uses (see CONTRACTS.md)."""
    pts = np.asarray(pts, dtype=float)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]
    ang = np.sort(ang)
    diff = np.abs(np.angle(np.exp(1j * (ang + np.pi / 2))))
    return np.roll(pts, -int(np.argmin(diff)), axis=0)


def apply_h(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homogeneous matrix to (N, 2) points."""
    p = np.hstack([pts, np.ones((len(pts), 1))]) @ m.T
    return p[:, :2] / p[:, 2:3]


def lowfreq_noise(rng: np.random.Generator, cell: int = 40, h: int = H, w: int = W) -> np.ndarray:
    """Smooth random field in [0, 1], like uneven lighting or mottling."""
    small = rng.random((h // cell + 2, w // cell + 2)).astype(np.float32)
    big = cv2.resize(small, (w + 2 * cell, h + 2 * cell), interpolation=cv2.INTER_CUBIC)
    big = big[cell:cell + h, cell:cell + w]
    big -= big.min()
    return big / max(float(big.max()), 1e-6)


def poly_mask(pts: np.ndarray, h: int = H, w: int = W) -> np.ndarray:
    """Anti-aliased filled polygon mask, float32 in [0, 1] (sub-pixel accurate via shift=4)."""
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(pts) * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
    return m.astype(np.float32) / 255.0


def line_mask(segments, thickness: int, closed: bool = False, h: int = H, w: int = W) -> np.ndarray:
    """Anti-aliased polyline mask. `segments` is a list of (N, 2) point arrays."""
    m = np.zeros((h, w), np.uint8)
    arrs = [np.round(np.asarray(s) * 16).astype(np.int32) for s in segments]
    cv2.polylines(m, arrs, closed, 255, thickness, cv2.LINE_AA, shift=4)
    return m.astype(np.float32) / 255.0


def stripes(direction: np.ndarray, period: float, phase: float = 0.0, h: int = H, w: int = W) -> np.ndarray:
    """Field in [0, 1] whose stripes (lines) run PARALLEL to `direction` (a 2-vector in image px)."""
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    n = np.array([-d[1], d[0]])                       # perpendicular: stripes vary along this
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    s = xx * n[0] + yy * n[1]
    return 0.5 + 0.5 * np.sin(2 * np.pi * s / period + phase)


def blend(img: np.ndarray, mask: np.ndarray, colour, alpha: float = 1.0) -> np.ndarray:
    m = (mask * alpha)[..., None]
    return img * (1 - m) + np.asarray(colour, np.float32) * m


def background(rng: np.random.Generator) -> np.ndarray:
    """A soft two-colour gradient at a random angle plus blotchy low-frequency variation (RGB float)."""
    c1 = np.array([60, 66, 74], np.float32) + rng.uniform(-10, 10, 3).astype(np.float32)
    c2 = np.array([150, 146, 136], np.float32) + rng.uniform(-15, 15, 3).astype(np.float32)
    a = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    t = (xx * np.cos(a) + yy * np.sin(a))
    t = (t - t.min()) / (t.max() - t.min())
    bg = c1 * (1 - t[..., None]) + c2 * t[..., None]
    bg *= (0.85 + 0.3 * lowfreq_noise(rng, 90))[..., None]
    bg += (lowfreq_noise(rng, 6) - 0.5)[..., None] * 18     # fine paper/table grain
    return bg


def mottled_fill(rng: np.random.Generator, base) -> np.ndarray:
    """Full-canvas isotropic texture (no preferred direction) in a base colour."""
    tex = 0.7 + 0.45 * lowfreq_noise(rng, 30) + 0.15 * (lowfreq_noise(rng, 5) - 0.5)
    return np.asarray(base, np.float32) * tex[..., None]


def finish(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Make it look less synthetic: uneven lighting, vignette, sensor noise, a touch of lens blur."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r2 = ((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2
    light = (0.9 + 0.2 * lowfreq_noise(rng, 150)) * (1.0 - 0.18 * r2)
    img = img * light[..., None]
    img = img + rng.normal(0, 4.0, img.shape).astype(np.float32)
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    return np.clip(img, 0, 255).astype(np.uint8)


def save(img_rgb: np.ndarray, path: Path) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 6])


# ----------------------------------------------------------------------------------------------------
# Reference scores (plain least squares, canonical units, circumradius 1)
# ----------------------------------------------------------------------------------------------------

def reference_scores(pts: np.ndarray) -> dict[str, float]:
    """Mean vertex residuals for similarity / affine / homography fits of image points -> canonical V_i,
    plus a central-symmetry error. pts must already be ordered clockwise from the top."""
    pts = np.asarray(pts, float)
    v = VERTICES

    # similarity (rotation + uniform scale + translation, no reflection): w = a*z + b with complex a, b
    z = pts[:, 0] + 1j * pts[:, 1]
    wv = v[:, 0] + 1j * v[:, 1]
    A = np.stack([z, np.ones_like(z)], axis=1)
    coef, *_ = np.linalg.lstsq(A, wv, rcond=None)
    sim_pred = A @ coef
    sim = np.abs(sim_pred - wv)

    # affine (6 parameters)
    X = np.hstack([pts, np.ones((6, 1))])
    M, *_ = np.linalg.lstsq(X, v, rcond=None)
    aff = np.linalg.norm(X @ M - v, axis=1)

    # homography (8 parameters, least squares, no RANSAC)
    Hm, _ = cv2.findHomography(pts.astype(np.float64), v.astype(np.float64), 0)
    hom = np.linalg.norm(apply_h(Hm, pts) - v, axis=1)

    # central symmetry: after similarity normalisation, the long-diagonal midpoints (p_i + p_{i+3}) / 2 should
    # coincide for i = 0, 1, 2 (same formula as metrics.central_symmetry_error: mean distance from their mean)
    q = np.stack([sim_pred.real, sim_pred.imag], axis=1)
    mids = 0.5 * (q[:3] + q[3:])
    cs = float(np.mean(np.linalg.norm(mids - mids.mean(axis=0), axis=1)))

    return {
        "similarity_mean": round(float(sim.mean()), 4),
        "affine_mean": round(float(aff.mean()), 4),
        "homography_mean": round(float(hom.mean()), 4),
        "central_symmetry_error": round(cs, 4),
    }


def as_list(pts: np.ndarray) -> list[list[float]]:
    return [[round(float(x), 2), round(float(y), 2)] for x, y in pts]


# ----------------------------------------------------------------------------------------------------
# Individual samples
# ----------------------------------------------------------------------------------------------------

CENTRE = np.array([W / 2, H / 2])
RADIUS = 270.0
AMBER = (214, 158, 72)
INK = (40, 30, 22)


def textured_hexagon_layer(rng, verts: np.ndarray, base=AMBER, extra_tex: np.ndarray | None = None):
    """Foreground colour layer + alpha mask for a flat textured hexagon with a dark rim."""
    fg = mottled_fill(rng, base)
    if extra_tex is not None:
        fg *= (0.8 + 0.35 * extra_tex)[..., None]
    rim = line_mask([verts], 6, closed=True)
    fg = blend(fg, rim, INK, 0.85)
    mask = np.maximum(poly_mask(verts), rim)
    return fg, mask


def sample_regular(rng):
    verts = hex_vertices(CENTRE, RADIUS)
    fg, mask = textured_hexagon_layer(rng, verts)
    img = blend(background(rng), mask, 0) + fg * mask[..., None]
    return finish(img, rng), verts, {
        "description": "A perfect regular pointy-top hexagon with an isotropic mottled texture, face-on.",
        "expected": {
            "summary": "All residuals ~0 (similarity, affine, homography); central symmetry error ~0; verdict 'strong'. "
                       "Texture has no preferred direction, so ternary should be mostly 0 (low coherence).",
        },
    }


def sample_sheared(rng):
    base = hex_vertices(CENTRE, RADIUS * 0.9)
    t = np.deg2rad(23.0)
    rot = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    shear = np.array([[1.0, 0.38, 0], [0, 1.0, 0], [0, 0, 1]])
    scale = np.array([[1.08, 0, 0], [0, 0.82, 0], [0, 0, 1]])
    to0 = np.array([[1, 0, -CENTRE[0]], [0, 1, -CENTRE[1]], [0, 0, 1]])
    back = np.array([[1, 0, CENTRE[0]], [0, 1, CENTRE[1]], [0, 0, 1]])
    A = back @ rot @ shear @ scale @ to0
    # texture is drawn on the unwarped hexagon, so it is sheared along with the shape (like a real photo)
    fg, mask = textured_hexagon_layer(rng, base, extra_tex=stripes((1, 0), 22.0))
    fg = cv2.warpAffine(fg, A[:2], (W, H), flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(mask, A[:2], (W, H), flags=cv2.INTER_LINEAR)
    img = blend(background(rng), mask, 0) + fg * mask[..., None]
    verts = order_cw_from_top(apply_h(A, base))
    return finish(img, rng), verts, {
        "description": "The regular hexagon after an affine map (anisotropic scale, shear 0.38, rotation 23 deg). "
                       "Faint horizontal stripes (before warping) show how texture is carried by the transform.",
        "expected": {
            "summary": "Affine and homography residuals ~0; similarity residual clearly > 0; central symmetry error ~0; "
                       "verdict 'strong'. After normalisation the stripes should be horizontal again.",
            "affine_matrix_image_from_regular": [[round(float(x), 6) for x in row] for row in A],
        },
    }


def sample_perspective(rng):
    base = hex_vertices(CENTRE, RADIUS * 0.95)
    src = np.float32([[110, 60], [690, 60], [690, 640], [110, 640]])
    dst = np.float32([[205, 95], [600, 125], [745, 655], [55, 610]])
    Hm = cv2.getPerspectiveTransform(src, dst).astype(np.float64)
    fg, mask = textured_hexagon_layer(rng, base, extra_tex=stripes((0, 1), 26.0))
    fg = cv2.warpPerspective(fg, Hm, (W, H), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(mask, Hm, (W, H), flags=cv2.INTER_LINEAR)
    img = blend(background(rng), mask, 0) + fg * mask[..., None]
    verts = order_cw_from_top(apply_h(Hm, base))
    return finish(img, rng), verts, {
        "description": "The regular hexagon photographed at an angle (a full perspective homography / keystone). "
                       "Faint vertical stripes (before warping) converge in the image.",
        "expected": {
            "summary": "Homography residual ~0; affine residual noticeably > 0 (a parallel-projection model cannot "
                       "explain convergence); similarity residual largest. With the homography toggle on, the "
                       "normalised stripes should come out vertical and evenly spaced.",
            "homography_image_from_regular": [[round(float(x), 8) for x in row] for row in Hm],
        },
    }


def _shade_faces(rng, faces, img):
    """faces: list of (quad (4,2) image px with quad[0] = near vertex, quad[2] = opposite corner, colour)."""
    for quad, colour in faces:
        radial = quad[2] - quad[0]                     # long diagonal: near vertex -> opposite corner
        tex = stripes(radial, 15.0, phase=rng.uniform(0, 2 * np.pi))
        layer = np.asarray(colour, np.float32) * (0.72 + 0.4 * tex)[..., None]
        layer *= (0.93 + 0.14 * lowfreq_noise(rng, 50))[..., None]
        img = blend(img, poly_mask(quad), 0) + layer * poly_mask(quad)[..., None]
    return img


def sample_shaded_cube(rng):
    verts = hex_vertices(CENTRE, RADIUS)
    C = CENTRE
    V = verts
    # CONVEX (vertex-out) reading from CONTRACTS.md: near vertex at the centre, faces TOP / RIGHT / LEFT,
    # each quad listed as (centre, side vertex, outer vertex, side vertex) so quad[2] is the radial end.
    faces = [
        (np.array([C, V[5], V[0], V[1]]), (236, 214, 170)),   # TOP   - lit
        (np.array([C, V[1], V[2], V[3]]), (176, 130, 72)),    # RIGHT - mid
        (np.array([C, V[3], V[4], V[5]]), (104, 74, 44)),     # LEFT  - shadow
    ]
    img = _shade_faces(rng, faces, background(rng))
    edges = [np.array([C, V[k]]) for k in (1, 3, 5)]
    img = blend(img, line_mask(edges, 4), INK, 0.8)
    img = blend(img, line_mask([V], 5, closed=True), INK, 0.9)
    return finish(img, rng), verts, {
        "description": "Orthographic isometric render of a cube viewed along its body diagonal (near corner at the "
                       "centre). Three differently shaded faces: top light, right mid, left dark. Each face carries "
                       "line texture along its long diagonal, i.e. radially out from the centre (top: vertical, "
                       "right: towards V2, left: towards V4).",
        "near_vertex": as_list(C[None])[0],
        "expected": {
            "summary": "All residuals ~0 and central symmetry error ~0. The line texture runs along each convex face's "
                       "long diagonal (centre to V0, V2, V4), which are the CONCAVE corner's edges. With the default "
                       "'edges' ternary convention: CONVEX reading 0, 0, 0 (its edge regions straddle two faces whose "
                       "stripes differ by 60 deg, coherence ~0.3-0.5, no primary line); CONCAVE reading -1, -1, -1 "
                       "with a primary line. With the original 'faces' convention it is the mirror image: CONVEX "
                       "+1, +1, +1 and CONCAVE 0, 0, 0.",
            "ternary_convex": {"EDGE_V1": 0, "EDGE_V3": 0, "EDGE_V5": 0},
            "ternary_concave": {"EDGE_V0": -1, "EDGE_V2": -1, "EDGE_V4": -1},
            "ternary_faces_convention": {"CONVEX": [1, 1, 1], "CONCAVE": [0, 0, 0]},
        },
    }


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def sample_cube_off_axis(rng):
    """A cube seen orthographically from a direction that is NOT its body diagonal."""
    R = _rot_x(np.deg2rad(24.0)) @ _rot_y(np.deg2rad(-28.0))   # camera looks along -z, viewer at +z
    scale = 300.0
    corners = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])
    cam = corners @ R.T

    def to_img(p):
        return np.stack([CENTRE[0] + scale * p[:, 0], CENTRE[1] - scale * p[:, 1] + 10], axis=1)

    img_pts = to_img(cam)
    near = int(np.argmax(cam[:, 2]))
    faces = []
    shades = {0: (190, 140, 80), 1: (236, 214, 170), 2: (120, 86, 52)}   # by axis: x-faces, y-faces, z-faces
    for axis in range(3):
        for side in (-0.5, 0.5):
            normal = np.zeros(3)
            normal[axis] = np.sign(side)
            if (R @ normal)[2] <= 0:
                continue
            idx = [i for i in range(8) if corners[i, axis] == side]
            # order the 4 face corners starting at the near vertex, going round the face
            others = [i for i in idx if i != near]
            opp = max(others, key=lambda i: np.abs(corners[i] - corners[near]).sum())
            sides = [i for i in others if i != opp]
            faces.append((img_pts[[near, sides[0], opp, sides[1]]], shades[axis]))
    assert len(faces) == 3, "view direction must show exactly three faces"

    hull = cv2.convexHull(img_pts.astype(np.float32)).reshape(-1, 2).astype(float)
    assert len(hull) == 6, "silhouette must be a hexagon"
    verts = order_cw_from_top(hull)

    img = _shade_faces(rng, faces, background(rng))
    edges = [np.array([img_pts[near], q[1]]) for q, _ in faces]
    img = blend(img, line_mask(edges, 4), INK, 0.8)
    img = blend(img, line_mask([verts], 5, closed=True), INK, 0.9)
    return finish(img, rng), verts, {
        "description": "Orthographic (parallel) view of a cube from a NON-diagonal direction (rotated 24 deg about x "
                       "and -28 deg about y). The silhouette is centrally symmetric (opposite sides parallel and "
                       "equal) but is NOT an affine image of a regular hexagon, and the near corner is off-centre. "
                       "Faces carry radial (near-corner -> opposite-corner) line texture.",
        "near_vertex": as_list(img_pts[near][None])[0],
        "expected": {
            "summary": "Central symmetry error ~0 (it is a real cube view) but affine residual clearly > 0 and > the "
                       "regular / sheared samples. This is the case the central-symmetry score exists for: a genuine "
                       "cube that the affine residual alone would penalise.",
        },
    }


def sample_irregular(rng):
    ang = np.deg2rad(-90 + 60 * np.arange(6) + rng.uniform(-24, 24, 6))
    rad = RADIUS * rng.uniform(0.55, 1.2, 6)
    pts = np.stack([CENTRE[0] + rad * np.cos(ang), CENTRE[1] + rad * np.sin(ang)], axis=1)
    verts = order_cw_from_top(pts)
    fg, mask = textured_hexagon_layer(rng, verts, base=(150, 160, 120))
    img = blend(background(rng), mask, 0) + fg * mask[..., None]
    return finish(img, rng), verts, {
        "description": "A random irregular hexagon (random angles +/-24 deg and radii 0.55-1.2). Noise control.",
        "expected": {
            "summary": "All residuals high (comparable to the random-hexagon null baseline), central symmetry error "
                       "high, verdict 'noise'. If this scores 'strong', the thresholds are too loose.",
        },
    }


def sample_honeycomb(rng):
    R = 70.0
    wall = 11.0
    dx, dy = np.sqrt(3) * R, 1.5 * R
    centres = []
    for j in range(-4, 5):
        off = dx / 2 if j % 2 else 0.0
        for i in range(-5, 6):
            centres.append((i, j, CENTRE[0] + i * dx + off, CENTRE[1] + j * dy))

    wax = np.array([226, 196, 120], np.float32)
    wax_layer = wax * (0.82 + 0.3 * lowfreq_noise(rng, 12))[..., None]
    img = wax_layer.copy()                                  # walls fill the whole frame first

    inner_r = R - (wall / 2) / np.cos(np.pi / 6)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    for (_, _, cx, cy) in centres:
        capped = rng.random() < 0.18
        m = poly_mask(hex_vertices((cx, cy), inner_r))
        if m.max() == 0:
            continue
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / inner_r
        if capped:
            col = np.array([240, 222, 168], np.float32) * (1.0 - 0.12 * d)[..., None]
        else:
            honey = np.array([170, 96, 22], np.float32) * rng.uniform(0.8, 1.1)
            col = honey * (0.55 + 0.5 * np.clip(d, 0, 1))[..., None]   # deeper (darker) in the middle
            hx, hy = cx - 0.3 * R, cy - 0.35 * R                        # small specular highlight
            spec = np.exp(-(((xx - hx) ** 2 + (yy - hy) ** 2) / (2 * (0.12 * R) ** 2)))
            col = col + 70 * spec[..., None]
        img = blend(img, m, 0) + col * m[..., None]

    # wax walls have fine ridges running along the wall direction
    ridge = 0.5 + 0.5 * lowfreq_noise(rng, 3)
    img *= (0.96 + 0.08 * ridge)[..., None]

    def cell(name, ci, cj):
        for (i, j, cx, cy) in centres:
            if i == ci and j == cj:
                return name, hex_vertices((cx, cy), R)
        raise KeyError((ci, cj))

    # pointy-top neighbours of the centre cell (row 0): east (1, 0); north-west (-1, -1); south-west (-1, 1)
    cells = [cell("centre", 0, 0), cell("east", 1, 0), cell("north_west", -1, -1), cell("south_west", -1, 1)]

    shared = []
    for a in range(1, len(cells)):
        va, vb = cells[0][1], cells[a][1]
        ia = [k for k in range(6) if np.min(np.linalg.norm(vb - va[k], axis=1)) < 1e-6]
        ib = [int(np.argmin(np.linalg.norm(vb - va[k], axis=1))) for k in ia]
        shared.append({"cells": [cells[0][0], cells[a][0]], "edge_a": ia, "edge_b": ib})

    verts = cells[0][1]
    return finish(img, rng), verts, {
        "description": "A small tiled honeycomb of pointy-top cells (circumradius 70 px, wall-centreline vertices) "
                       "with pale wax walls, dark honey-filled cells and some capped cells. 'vertices' is the "
                       "centre cell; 'cells' lists the centre plus three edge-sharing neighbours.",
        "cells": [{"name": n, "vertices": as_list(v)} for n, v in cells],
        "expected": {
            "summary": "Every cell is a regular hexagon: residuals ~0 for each. Neighbour mode should find exactly one "
                       "shared edge between the centre and each neighbour. This sample tests edge FINDING, not "
                       "continuity: once the walls are masked out the cell interiors carry little oriented texture, "
                       "so continuity scores are mixed and mostly 'unrelated'.",
            "shared_edges": shared,
        },
    }


def sample_flat_top(rng):
    rot = 32.0    # 30 deg would be exactly flat-topped with two tied 'top' vertices; 2 deg extra breaks the tie
    verts_raw = hex_vertices(CENTRE, RADIUS, rot)
    verts = order_cw_from_top(verts_raw)
    top_edge = verts_raw[5] - verts_raw[0]   # the near-horizontal top edge
    fg, mask = textured_hexagon_layer(rng, verts_raw, base=(120, 150, 190), extra_tex=stripes(top_edge, 20.0))
    img = blend(background(rng), mask, 0) + fg * mask[..., None]
    # a marker dot near the upper-left vertex, so the effect of rotation_offset is easy to see
    marker = poly_mask(hex_vertices(CENTRE + 0.72 * (verts[0] - CENTRE), 14))
    img = blend(img, marker, (230, 40, 40))
    return finish(img, rng), verts, {
        "description": "A regular hexagon turned flat-top (rotated 32 deg: flat-top plus 2 deg so the 'top' vertex is "
                       "unambiguous). Stripes run parallel to the top edge; a red dot marks the upper-left vertex.",
        "expected": {
            "summary": "Residuals all ~0 (rotation is part of the similarity). Auto-ordering picks the upper-left "
                       "vertex (the red dot) as V0, so the normalised image is turned ~28 deg relative to the photo; "
                       "stepping rotation_offset 0..5 moves the dot round the canonical hexagon and odd offsets swap "
                       "which interior 'Y' is read as the near corner.",
        },
    }


SAMPLES = [
    ("regular_hexagon.png", sample_regular),
    ("sheared_hexagon.png", sample_sheared),
    ("perspective_hexagon.png", sample_perspective),
    ("shaded_cube.png", sample_shaded_cube),
    ("cube_off_axis.png", sample_cube_off_axis),
    ("irregular_hexagon.png", sample_irregular),
    ("honeycomb.png", sample_honeycomb),
    ("rotated_flat_top.png", sample_flat_top),
]


def generate(out_dir: Path | str | None = None) -> dict:
    """Write every sample PNG and ground_truth.json into out_dir (default: this folder). Returns the truth dict."""
    out = Path(out_dir) if out_dir else Path(__file__).resolve().parent
    out.mkdir(parents=True, exist_ok=True)
    truth: dict[str, dict] = {}
    for index, (name, fn) in enumerate(SAMPLES):
        img, verts, info = fn(rng_for(index))
        save(img, out / name)
        verts = np.asarray(verts, float)
        entry = {"vertices": as_list(verts), "description": info.pop("description"), "image_size": [W, H]}
        expected = info.pop("expected")
        expected["reference_scores"] = reference_scores(np.array(entry["vertices"]))
        entry["expected"] = expected
        entry.update(info)
        truth[name] = entry
    (out / "ground_truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return truth


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=None, help="output folder (default: the samples/ folder)")
    args = ap.parse_args(argv)
    truth = generate(args.out)
    for name, e in truth.items():
        print(f"{name:26s} {e['expected']['reference_scores']}")


if __name__ == "__main__":
    main()
