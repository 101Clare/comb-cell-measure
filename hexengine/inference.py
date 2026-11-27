"""Layer and ternary inference on a normalised (canonical-frame) image.

OWNER: Agent C (inference). Ternary rule: see CONTRACTS.md.
Stages (each independently callable so the UI can toggle/inspect them):
  1. orientation_field  — structure tensor
  2. analyse_faces      — per-face dominant orientation + coherence for the current CubeState
  3. ternary assignment — inside analyse_faces, via assign_ternary
  4. primary_line       — the +1 (primary-sign) line vs the neutral vertical reference

Plain-words guide (for Clare and Ben)
-------------------------------------
**Structure tensor.** At every pixel we measure the brightness gradient (gx, gy): an arrow pointing
"uphill" across an edge or stripe. One arrow on its own is noisy, and arrows on the two sides of a stripe
point in opposite directions, so they cannot simply be averaged. Instead we form the 2x2 matrix

    J = [[gx*gx, gx*gy],
         [gx*gy, gy*gy]]

at each pixel and blur each of its three entries with a Gaussian (``sigma_tensor``). This is the
structure tensor. A gradient g and its opposite -g give the same J, so the opposite arrows reinforce
rather than cancel. The two eigenvalues l1 >= l2 >= 0 of the blurred J say how much gradient energy
there is along the dominant direction (l1) and across it (l2):

* l1 >> l2   -> a single clear direction (stripes, an edge): high coherence
* l1 ~= l2   -> no preferred direction (noise, a corner, isotropic texture): low coherence
* l1 = l2 = 0 -> flat region: no information at all (we set coherence 0)

The eigenvector for l1 points across the stripes (along the gradient). The LINE orientation we report,
``theta``, is perpendicular to it, i.e. along the stripes. With Jxx, Jyy, Jxy the blurred entries:

    gradient angle  phi   = 0.5 * atan2(2 Jxy, Jxx - Jyy)
    line angle      theta = phi + pi/2   (wrapped into [0, pi))
    energy                = Jxx + Jyy = l1 + l2   (trace)
    coherence             = (l1 - l2) / (l1 + l2) = sqrt((Jxx - Jyy)^2 + 4 Jxy^2) / (Jxx + Jyy)

We use this *linear* coherence form (not its square), so it is in [0, 1] and reads directly as
"fraction of the gradient energy that agrees on one direction".

**Doubled-angle trick.** Orientations are undirected: a line at 10 deg and a line at 190 deg are the same
line, and 5 deg and 175 deg are only 10 deg apart. Averaging raw angles gets this wrong (it would give
90 deg). So each orientation theta is turned into the unit vector (cos 2theta, sin 2theta): doubling the
angle makes theta and theta+pi land on the same vector, and 5 deg / 175 deg land close together. We
average these vectors (weighted), then halve the angle of the result to get back an orientation. The
length of the averaged vector (0..1) measures how much the pixels agree: this is the face coherence.

**Ternary rule** (CONTRACTS.md). Three rhombic regions are read, each with one corner at the hexagon centre.
Its *radial* line is the long diagonal running out from the centre; its *tangential* line is the short
diagonal, at right angles, wrapping around the centre. For a region with dominant line orientation theta
and coherence c:

* c below ``coherence_min``                      -> 0   (not enough structure to say anything)
* theta within ``angle_tol`` of the radial line  -> state.sign  (+1 convex, -1 concave: "primary")
* theta within ``angle_tol`` of the tangential   -> -state.sign
* anything else (oblique)                        -> 0

All angle differences are undirected (taken modulo pi, so they lie in [0, pi/2]).

**Which regions (the sign convention).** By default ("edges") the regions are centred on the reading's own
three cube edges, so a region's radial line IS one of that reading's edges. Calibrated on the tumbling-block
tile control: its real edges lie on the convex spokes, and it scores +1, +1, +1 under the convex reading and
0, 0, 0 under the concave one. The original rule ("faces") read the reading's own faces instead; their
boundaries are the edges, which the margin hides, so it scored that same tile -1, -1, -1 under CONCAVE.
In one line: **+1 = structure runs along the convex cube's edges; -1 = along the concave corner's edges.**

**Edge margin: the outline must not count as evidence.** The boundaries of each face are strong edges
in their own right: the hexagon outline (the warp's grey cut-off, or shape against background) and the
spokes (shading steps between faces). The structure tensor would happily read those as "structure":
e.g. the TOP face's two outer edges at 30 and 150 deg average (doubled angle) to horizontal, which is its
tangential line, so a plain, textureless hexagon would come out as -1. To avoid this, each face is read
only through a shrunken mask that stays a margin away from EVERY face boundary (outer edges and spokes).
The margin is ``edge_margin_frac`` (default 0.12) of the circumradius, but never less than the reach of
the blurs (4 * sigma_tensor + 4 * sigma_grad + 2 px, 32 px with the defaults), because the smoothing smears
a boundary's gradient that far inwards. On a textureless face even a faint smear would be the only signal.
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np

from . import canonical, cube
from .types import CubeState, Face, FaceResult, OrientationField, PrimaryLine, TextureLock

# Default margin kept clear of face boundaries, as a fraction of the canonical circumradius (see face_masks).
DEFAULT_EDGE_MARGIN_FRAC = 0.12

# Below this tensor trace (relative to the image maximum, and in absolute terms) we treat a pixel as flat.
_REL_ENERGY_EPS = 1e-6
_ABS_ENERGY_EPS = 1e-12


# ---------------------------------------------------------------------------------------------------------
# small angle helpers
# ---------------------------------------------------------------------------------------------------------

def _wrap_orientation(theta):
    """Wrap any angle (scalar or array) into the orientation range [0, pi)."""
    out = np.mod(theta, np.pi)
    # np.mod can return exactly pi for tiny negative inputs because of rounding; fold that back to 0.
    return np.where(out >= np.pi, 0.0, out) if isinstance(out, np.ndarray) else (0.0 if out >= np.pi else float(out))


def undirected_diff(a: float, b: float) -> float:
    """Angle between two undirected lines, in radians, in [0, pi/2]."""
    d = abs(float(a) - float(b)) % np.pi
    return min(d, np.pi - d)


def _to_grey_float(image: np.ndarray) -> np.ndarray:
    """RGB/RGBA/grey image of any dtype -> float64 grey in roughly [0, 1]."""
    img = np.asarray(image)
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]
        if img.shape[2] == 1:
            img = img[:, :, 0]
        else:
            # ITU-R 601 luma weights (RGB channel order, as loaded by PIL / our pipeline)
            img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
    img = img.astype(np.float64)
    if np.asarray(image).dtype == np.uint8:
        img /= 255.0
    return img


# ---------------------------------------------------------------------------------------------------------
# 1. orientation field
# ---------------------------------------------------------------------------------------------------------

def orientation_field(image: np.ndarray, sigma_grad: float = 1.5, sigma_tensor: float = 6.0) -> OrientationField:
    """Structure tensor of an RGB or grey canonical-frame image (Sobel/Gaussian derivatives, tensor smoothing).
    theta = LINE orientation (perpendicular to gradient), in [0, pi), y-down frame pixels.
    coherence = (l1 - l2) / (l1 + l2)  (linear form, see module docstring); 0 where energy ~ 0.

    Gradients are "derivative of Gaussian": the grey image is first blurred with sigma_grad, then 3x3 Sobel
    filters take d/dx and d/dy (x right, y down). The three tensor products are blurred with sigma_tensor.
    Mask-agnostic: the whole raster is processed; callers restrict to faces with a mask afterwards.
    """
    grey = _to_grey_float(image)
    if sigma_grad > 0:
        grey = cv2.GaussianBlur(grey, (0, 0), sigmaX=sigma_grad, sigmaY=sigma_grad, borderType=cv2.BORDER_REFLECT)
    # Sobel with scale 1/8 gives a true finite-difference derivative (intensity units per pixel).
    gx = cv2.Sobel(grey, cv2.CV_64F, 1, 0, ksize=3, scale=1.0 / 8.0, borderType=cv2.BORDER_REFLECT)
    gy = cv2.Sobel(grey, cv2.CV_64F, 0, 1, ksize=3, scale=1.0 / 8.0, borderType=cv2.BORDER_REFLECT)

    jxx, jyy, jxy = gx * gx, gy * gy, gx * gy
    if sigma_tensor > 0:
        blur = lambda a: cv2.GaussianBlur(a, (0, 0), sigmaX=sigma_tensor, sigmaY=sigma_tensor,
                                          borderType=cv2.BORDER_REFLECT)
        jxx, jyy, jxy = blur(jxx), blur(jyy), blur(jxy)

    energy = jxx + jyy                                       # = l1 + l2
    anisotropy = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy ** 2)  # = l1 - l2

    eps = max(_ABS_ENERGY_EPS, _REL_ENERGY_EPS * float(energy.max(initial=0.0)))
    valid = energy > eps
    coherence = np.zeros_like(energy)
    coherence[valid] = anisotropy[valid] / energy[valid]
    coherence = np.clip(coherence, 0.0, 1.0)

    phi = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)             # dominant GRADIENT orientation
    theta = _wrap_orientation(phi + np.pi / 2.0)             # LINE orientation, perpendicular to gradient
    theta = np.where(valid, theta, 0.0)
    energy = np.where(valid, energy, 0.0)

    return OrientationField(theta=theta.astype(np.float64), coherence=coherence.astype(np.float64),
                            energy=energy.astype(np.float64))


# ---------------------------------------------------------------------------------------------------------
# 2. face projection
# ---------------------------------------------------------------------------------------------------------

# Buffer added to the measured wall width. The patch specified 4 px; on the baseline photos 4 px (and 6 px) still
# left a false six-fold texture lock in soap froth (pooled shuffle p = 0.0002 and 0.016): the film shades gradually
# for several pixels beyond the dark wall line. From 8 px that artefact was gone (p = 0.17 and higher) while the
# snowflakes' genuine lock was unchanged (p = 0.0025 at every buffer). So the default is 8 px; it is a setting.
SOURCE_MARGIN_BUFFER_PX = 8.0


def source_margin_frac(wall_width_px: float, radius_px: float, buffer_px: float = SOURCE_MARGIN_BUFFER_PX) -> float:
    """Edge margin set in the ORIGINAL photo's pixels, as a fraction of the circumradius.

    margin (source px) = estimated wall width + buffer (default 8 px, see SOURCE_MARGIN_BUFFER_PX); divided by the
    hexagon's radius in source px it becomes the same fraction of the canonical circumradius, which is what
    edge_margin_frac means everywhere here.

    Why: a fixed fraction of the normalised frame is far too thin for small cells. A 25 px cell is enlarged about
    9x when normalised, so its 3 px blurred wall becomes ~27 frame px of "structure" right where the faces are
    read; the baseline test showed it producing a false six-fold texture lock in foam. Measuring the margin in
    the photo's own pixels removes the outline however much the cell is enlarged.
    """
    if radius_px <= 0:
        return DEFAULT_EDGE_MARGIN_FRAC
    return float(max(wall_width_px, 0.0) + buffer_px) / float(radius_px)


def edge_margin_px(size: int, edge_margin_frac: float = DEFAULT_EDGE_MARGIN_FRAC,
                   sigma_tensor: float | None = None, sigma_grad: float | None = None) -> float:
    """Margin in frame pixels kept clear of every face boundary.

    margin = edge_margin_frac * (pixels per canonical unit at this frame size), with a floor equal to the
    reach of the blurs when sigma_tensor is given:

        floor = 4 * sigma_tensor + 4 * sigma_grad + 2  px

    Why that floor: OpenCV cuts its float Gaussian kernels off at 4 sigma, so a boundary's gradient spreads
    inwards by at most 4*sigma_grad (pre-blur) + 1 (Sobel) + 4*sigma_tensor (tensor blur); +1 more for
    polygon rasterisation. Beyond that the outline contributes exactly nothing. A smaller floor (e.g. 2.5
    sigma) still lets a few percent of the outline's energy through, and on an untextured face that leaked
    energy is the ONLY energy, so it would decide the face's orientation. Defaults (6.0, 1.5) give 32 px,
    slightly above 0.12 * FRAME_SCALE = 27.6 px; a face rhombus has an inradius of about 100 px, so plenty
    remains.
    """
    margin = float(edge_margin_frac) * 0.45 * float(size)
    if sigma_tensor is not None:
        sg = 0.0 if sigma_grad is None else float(sigma_grad)
        margin = max(margin, 4.0 * float(sigma_tensor) + 4.0 * sg + 2.0)
    return max(margin, 0.0)


CONVENTIONS = ("edges", "faces")
DEFAULT_CONVENTION = "edges"


def regions(state: CubeState, convention: str = DEFAULT_CONVENTION) -> list[Face]:
    """The three rhombic regions that are read for this Necker state.

    convention "edges" (default): the regions centred on THIS reading's three cube edges (spokes). Each one is a
        rhombus whose long diagonal is a spoke of `state`; geometrically these are the faces of the dual reading.
        Structure along that diagonal is structure along the reading's own edge, so a real convex cube (whose
        brightness steps lie on the convex spokes) scores +1 under the convex reading, and a real concave corner
        scores -1 under the concave reading. Region names say which edge they hold, e.g. "EDGE_V1".
    convention "faces" (the original rule): the reading's own three visible faces. A real cube's edges are those
        faces' boundaries, which the edge margin excludes, so this convention cannot see them; the baseline test
        found it credited a convex cube's edges to the concave reading instead. Kept for comparison.
    """
    if convention not in CONVENTIONS:
        raise ValueError(f"convention must be one of {CONVENTIONS}")
    if convention == "faces":
        return cube.faces(state)
    dual = CubeState.CONCAVE if CubeState(state) is CubeState.CONVEX else CubeState.CONVEX
    return [dataclasses.replace(f, name=f"EDGE_V{f.outer_vertex_ids[1]}") for f in cube.faces(dual)]


def face_masks(state: CubeState, size: int | None = None, *,
               edge_margin_frac: float = DEFAULT_EDGE_MARGIN_FRAC,
               min_margin_px: float = 0.0, convention: str = DEFAULT_CONVENTION) -> list[np.ndarray]:
    """Boolean (size, size) masks for each region of regions(state, convention), in the canonical frame.

    Each mask is shrunk so it excludes every pixel within the margin of ANY of that region's boundary lines
    (the outer hexagon edges and the inner spokes alike): margin_px = max(edge_margin_frac * FRAME_SCALE
    scaled to `size`, min_margin_px). Pass edge_margin_frac=0 for the full, un-eroded rhombus (e.g. drawing).
    """
    size = canonical.FRAME_SIZE if size is None else int(size)
    margin = max(edge_margin_px(size, edge_margin_frac), float(min_margin_px))
    masks = []
    for face in regions(state, convention):
        m = np.zeros((size, size), dtype=np.uint8)
        poly = np.round(canonical.to_frame(face.polygon, size)).astype(np.int32)
        cv2.fillPoly(m, [poly], 1)
        if margin > 0:
            # distance (px) from each inside pixel to the nearest pixel outside the rhombus
            dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
            masks.append(dist > margin)
        else:
            masks.append(m.astype(bool))
    return masks


def dominant_orientation(field: OrientationField, mask: np.ndarray) -> tuple[float, float]:
    """(theta, coherence) aggregated over a mask by averaging the doubled-angle vectors weighted by energy*coherence.
    Returned coherence is the length of the mean doubled-angle vector, in [0, 1].

    Weight w = energy * coherence, so strong, clean structure dominates and flat / noisy pixels count little.
        V = sum(w * (cos 2theta, sin 2theta)) / sum(w)
        theta_face = 0.5 * atan2(V_y, V_x)  wrapped to [0, pi);  coherence_face = |V|
    Returns (0.0, 0.0) when the mask is empty or carries no weight.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0.0, 0.0
    t = field.theta[mask]
    w = field.energy[mask] * field.coherence[mask]
    wsum = float(w.sum())
    if not np.isfinite(wsum) or wsum <= _ABS_ENERGY_EPS:
        return 0.0, 0.0
    vx = float((w * np.cos(2.0 * t)).sum()) / wsum
    vy = float((w * np.sin(2.0 * t)).sum()) / wsum
    coherence = float(np.clip(np.hypot(vx, vy), 0.0, 1.0))
    theta = float(_wrap_orientation(0.5 * np.arctan2(vy, vx)))
    return theta, coherence


# ---------------------------------------------------------------------------------------------------------
# 3. ternary assignment
# ---------------------------------------------------------------------------------------------------------

def assign_ternary(theta: float, coherence: float, radial_theta: float, tangential_theta: float,
                   state: CubeState, angle_tol_deg: float = 20.0, coherence_min: float = 0.15) -> int:
    """Ternary value of one face per the CONTRACTS.md rule (see module docstring)."""
    if coherence < coherence_min:
        return 0
    tol = np.deg2rad(angle_tol_deg)
    d_rad = undirected_diff(theta, radial_theta)
    d_tan = undirected_diff(theta, tangential_theta)
    # If the tolerance is >= 45 deg both tests could pass; the closer line wins.
    if d_rad <= tol and d_rad <= d_tan:
        return int(state.sign)
    if d_tan <= tol:
        return -int(state.sign)
    return 0


def analyse_faces(field: OrientationField, state: CubeState, angle_tol_deg: float = 20.0,
                  coherence_min: float = 0.15, *, edge_margin_frac: float = DEFAULT_EDGE_MARGIN_FRAC,
                  sigma_tensor: float | None = 6.0, sigma_grad: float | None = 1.5,
                  convention: str = DEFAULT_CONVENTION) -> list[FaceResult]:
    """Dominant orientation, coherence, ternary value and alignments for each region of `state`
    (see regions() for the two conventions). The field must be on a square canonical frame.

    Regions are read through margin-eroded masks (see face_masks / edge_margin_px) so the hexagon outline and
    the region boundaries are not counted as evidence. For a margin set in the photo's own pixels, pass
    edge_margin_frac=source_margin_frac(wall_width_px, radius_px). sigma_tensor / sigma_grad should match the
    values used to build `field`; they set the margin floor 4*sigma_tensor + 4*sigma_grad + 2 (None disables it).
    """
    size = field.theta.shape[0]
    margin = edge_margin_px(size, edge_margin_frac, sigma_tensor, sigma_grad)
    results = []
    masks = face_masks(state, size, edge_margin_frac=edge_margin_frac, min_margin_px=margin, convention=convention)
    for face, mask in zip(regions(state, convention), masks):
        theta, coh = dominant_orientation(field, mask)
        tern = assign_ternary(theta, coh, face.radial_theta, face.tangential_theta, state,
                              angle_tol_deg=angle_tol_deg, coherence_min=coherence_min)
        results.append(FaceResult(
            face=face,
            dominant_theta=theta,
            coherence=coh,
            ternary=tern,
            radial_alignment=abs(float(np.cos(undirected_diff(theta, face.radial_theta)))),
            tangential_alignment=abs(float(np.cos(undirected_diff(theta, face.tangential_theta)))),
        ))
    return results


# ---------------------------------------------------------------------------------------------------------
# 4. primary line
# ---------------------------------------------------------------------------------------------------------

def primary_line(results: list[FaceResult], state: CubeState) -> PrimaryLine | None:
    """Radial line of the face with the strongest primary-signed alignment (coherence * radial_alignment among
    faces whose ternary == state.sign). None if no face carries the primary sign.

    start = hexagon centre, end = the face's middle outer vertex (end of the long diagonal).
    angle_vs_reference_deg is the signed angle from the neutral reference direction V3->V0 (straight up)
    to the direction centre->end, in degrees in (-180, 180]; positive = clockwise on screen (y down).
    E.g. a line to V1 gives +60, to V5 gives -60, to V3 gives 180.
    """
    candidates = [r for r in results if r.ternary == state.sign and r.ternary != 0]
    if not candidates:
        return None
    best = max(candidates, key=lambda r: r.coherence * r.radial_alignment)
    poly = np.asarray(best.face.polygon, dtype=float)
    start = poly[0].copy()
    end = poly[2].copy()           # polygon = centre, then three outer vertices; the middle one ends the long diagonal
    d = end - start
    ref = canonical.VERTICES[0] - canonical.VERTICES[3]
    cross = ref[0] * d[1] - ref[1] * d[0]
    dot = ref[0] * d[0] + ref[1] * d[1]
    angle = float(np.degrees(np.arctan2(cross, dot)))
    if angle <= -180.0:
        angle += 360.0
    return PrimaryLine(start=start, end=end, theta=float(best.face.radial_theta), ternary=int(best.ternary),
                       angle_vs_reference_deg=angle, face_name=best.face.name)


# ---------------------------------------------------------------------------------------------------------
# 5. six-fold texture lock (with a rotation-shuffle control)
# ---------------------------------------------------------------------------------------------------------

LOCK_P = 0.01            # significance level for "locked"
LOCK_MIN_ORDER = 0.10    # below this the texture has no six-fold pattern to speak of


def texture_lock(field: OrientationField, edge_margin_frac: float = DEFAULT_EDGE_MARGIN_FRAC,
                 sigma_tensor: float | None = 6.0, sigma_grad: float | None = 1.5) -> TextureLock:
    """Six-fold order of the texture inside the hexagon (outline margin excluded) and its rotation-shuffle p.

    Z = sum w * exp(i 6 (theta - 90 deg)) / sum w, with w = energy * coherence. |Z| is how six-fold the texture
    is; psi = arg(Z) / 6 is where that pattern sits relative to the hexagon's own directions (0 = along the
    cube axes, +-30 deg = across them; both count as lined up, so the test uses cos(12 psi)).

    Shuffle control, done exactly: rotating the cell's content by a random angle shifts 12 psi uniformly, so the
    chance that a randomly rotated cell lines up at least this closely is offset / 15 deg, where offset is the
    distance to the nearest of the 12 hexagon directions. A single cell can therefore only reach p < 0.01 if it
    is within 0.15 deg; pooling cells (pooled_lock_test) is what gives the test real power.
    """
    size = field.theta.shape[0]
    m = canonical.hexagon_mask(size).astype(np.uint8)
    margin = edge_margin_px(size, edge_margin_frac, sigma_tensor, sigma_grad)
    inside = cv2.distanceTransform(m, cv2.DIST_L2, 5) > margin
    w = (field.energy * field.coherence)[inside]
    if w.size == 0 or float(w.sum()) <= _ABS_ENERGY_EPS:
        return TextureLock(order=0.0, phase_deg=0.0, offset_deg=15.0, p_value=1.0, locked=False)
    z = complex(np.sum(w * np.exp(1j * 6 * (field.theta[inside] - np.pi / 2))) / w.sum())
    phi12 = 2.0 * float(np.angle(z))                    # = 12 psi, in (-2 pi, 2 pi]
    wrapped = abs((phi12 + np.pi) % (2 * np.pi) - np.pi)   # [0, pi]
    offset = float(np.degrees(wrapped) / 12.0)             # [0, 15] deg
    p = float(wrapped / np.pi)
    order = abs(z)
    return TextureLock(order=float(order), phase_deg=float(np.degrees(np.angle(z)) / 6.0), offset_deg=offset,
                       p_value=p, locked=bool(p < LOCK_P and order >= LOCK_MIN_ORDER))


def pooled_lock_test(locks: list[TextureLock], n_shuffle: int = 5000, seed: int = 0) -> dict | None:
    """Do several cells line up with their own hexagon directions more than randomly rotated cells would?

    L = order-weighted mean of cos(12 psi) over the cells (+1 = every cell exactly lined up, ~0 = random).
    Shuffle: give every cell an independent random rotation and recompute L, n_shuffle times;
    p = share of shuffles with L at least as high as observed. Cells with order below LOCK_MIN_ORDER are left out.
    Returns None when fewer than 2 cells qualify.
    """
    use = [lk for lk in locks if lk.order >= LOCK_MIN_ORDER]
    if len(use) < 2:
        return None
    rng = np.random.default_rng(seed)
    w = np.array([lk.order for lk in use])
    psi = np.radians([lk.phase_deg for lk in use])
    obs = float(np.sum(w * np.cos(12 * psi)) / w.sum())
    rot = rng.uniform(0, np.pi / 3, (n_shuffle, len(use)))
    null = np.sum(w * np.cos(12 * (psi + rot)), axis=1) / w.sum()
    p = float((np.sum(null >= obs) + 1) / (n_shuffle + 1))
    return {"n": len(use), "L": obs, "p": p, "locked": p < LOCK_P}
