"""Drawing overlays and building multi-panel comparison sheets. All functions return NEW RGB uint8 arrays.

OWNER: Agent F (render & export). Uses OpenCV drawing (anti-aliased) — no Matplotlib windows.
Palette: points = cyan, manifold guides = gold (255, 200, 40), +1 = (60, 200, 90), 0 = grey, -1 = (220, 70, 70).

Conventions used throughout this module
---------------------------------------
* Every image is RGB uint8 (H, W, 3). Colours are therefore given as (R, G, B) tuples — OpenCV does not care about
  channel order when drawing, so we simply pass RGB tuples straight through.
* Inputs are never modified: each function copies its input first and returns the copy.
* "Frame" functions (draw_construction, draw_polylines, draw_orientation, draw_ternary) take a canonical-frame image
  (square, see canonical.py) and geometry in canonical units; we convert with canonical.to_frame(pts, frame.shape[0]).
* Semi-transparency: draw onto an `overlay` copy, then blend with cv2.addWeighted(overlay, a, base, 1 - a). Pixels
  that were not drawn on are identical in both layers, so they come through unchanged.
* OpenCV's Hershey fonts are ASCII only, so labels avoid symbols such as the degree sign.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from . import canonical
from .types import CubeState, EdgeContinuity, FaceResult, HexFit, OrientationField, Polyline, PrimaryLine, Scores

# ---------------------------------------------------------------------------------------------------------------------
# Palette (RGB)
# ---------------------------------------------------------------------------------------------------------------------

CYAN = (0, 230, 255)
GOLD = (255, 200, 40)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
TERNARY_COLOURS = {+1: (60, 200, 90), 0: (150, 150, 150), -1: (220, 70, 70)}
VERDICT_COLOURS = {"regular": (60, 200, 90), "irregular": (240, 170, 40), "random-like": (220, 70, 70),
                   # legacy fixed-threshold verdicts
                   "strong": (60, 200, 90), "weak": (240, 170, 40), "noise": (220, 70, 70)}
CONTINUITY_GOOD, CONTINUITY_FAIR, CONTINUITY_POOR = (60, 200, 90), (240, 170, 40), (220, 70, 70)
SHEET_BG = (28, 30, 34)       # dark background for panels / sheets
SHEET_TITLE_BG = (45, 48, 56)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX
LINE = cv2.LINE_AA


# ---------------------------------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------------------------------

def _rgb_copy(image: np.ndarray) -> np.ndarray:
    """Return a fresh, contiguous RGB uint8 copy of `image` (accepts grey, RGB or RGBA; floats in [0,1] or [0,255])."""
    img = np.asarray(image)
    if img.dtype != np.uint8:
        img = img.astype(np.float64)
        if img.size and img.max() <= 1.0:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2RGB)
    return np.ascontiguousarray(img).copy()


def _blend(overlay: np.ndarray, base: np.ndarray, opacity: float) -> np.ndarray:
    """Alpha-blend `overlay` onto `base` (both same shape). opacity 1 = overlay only, 0 = base only."""
    a = float(np.clip(opacity, 0.0, 1.0))
    return cv2.addWeighted(overlay, a, base, 1.0 - a, 0.0)


def _ipt(p) -> tuple[int, int]:
    """Point -> integer pixel tuple for OpenCV."""
    return int(round(float(p[0]))), int(round(float(p[1])))


def _diag(shape) -> float:
    """Image diagonal in pixels — the yardstick for line widths, marker and font sizes."""
    return math.hypot(shape[0], shape[1])


def _frame_scale(frame: np.ndarray) -> float:
    """Size factor relative to the standard 512 px canonical frame (1.0 at 512 px)."""
    return frame.shape[0] / float(canonical.FRAME_SIZE)


def _to_px(points, frame: np.ndarray) -> np.ndarray:
    """Canonical units (N, 2) -> frame pixels (N, 2) float."""
    return canonical.to_frame(np.asarray(points, dtype=float).reshape(-1, 2), frame.shape[0])


def _text(img: np.ndarray, text: str, org, scale: float, color, thickness: int = 1, font=FONT,
          outline: bool = True) -> None:
    """putText with a dark outline so labels stay readable on any background (draws in place)."""
    org = _ipt(org)
    if outline:
        cv2.putText(img, text, org, font, scale, BLACK, thickness + max(2, thickness), LINE)
    cv2.putText(img, text, org, font, scale, color, thickness, LINE)


def _text_centered(img: np.ndarray, text: str, centre, scale: float, color, thickness: int = 1, font=FONT,
                   outline: bool = True) -> None:
    """putText centred on `centre` (both axes)."""
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    _text(img, text, (centre[0] - tw / 2.0, centre[1] + th / 2.0), scale, color, thickness, font, outline)


def _dashed_polyline(img: np.ndarray, pts: np.ndarray, color, thickness: int, dash: float, gap: float,
                     closed: bool = False) -> None:
    """Draw a dashed polyline (pixel coords) in place. The dash pattern continues smoothly around corners."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    if closed and len(pts) > 2:
        pts = np.vstack([pts, pts[:1]])
    period = dash + gap
    phase = 0.0  # arc length already consumed within the current dash+gap period
    for a, b in zip(pts[:-1], pts[1:]):
        seg = b - a
        length = float(np.hypot(*seg))
        if length < 1e-9:
            continue
        u = seg / length
        s = 0.0
        while s < length:
            if phase < dash:                       # inside a dash: draw until dash ends or segment ends
                step = min(dash - phase, length - s)
                cv2.line(img, _ipt(a + u * s), _ipt(a + u * (s + step)), color, thickness, LINE)
            else:                                  # inside a gap: skip
                step = min(period - phase, length - s)
            s += step
            phase = (phase + step) % period


def _arrowheads(img: np.ndarray, pts: np.ndarray, color, size: float, spacing: float, closed: bool = False) -> None:
    """Filled arrowheads pointing along the path, placed every `spacing` px of arc length (pixel coords, in place)."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    if closed and len(pts) > 2:
        pts = np.vstack([pts, pts[:1]])
    seglen = np.hypot(*(pts[1:] - pts[:-1]).T) if len(pts) > 1 else np.zeros(0)
    total = float(seglen.sum())
    if total < 1e-6:
        return
    # Evenly spaced positions, offset by half a spacing so arrows avoid the very ends of the path.
    n = max(1, int(total // spacing))
    targets = (np.arange(n) + 0.5) * (total / n)
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    for t in targets:
        i = int(np.clip(np.searchsorted(cum, t, side="right") - 1, 0, len(seglen) - 1))
        if seglen[i] < 1e-9:
            continue
        u = (pts[i + 1] - pts[i]) / seglen[i]
        tip = pts[i] + u * (t - cum[i]) + u * size * 0.5
        normal = np.array([-u[1], u[0]])
        base = tip - u * size
        tri = np.array([tip, base + normal * size * 0.5, base - normal * size * 0.5])
        cv2.fillConvexPoly(img, np.round(tri).astype(np.int32), color, LINE)


def _distinct_colour(i: int) -> tuple[int, int, int]:
    """Bright, well-separated colours for cell outlines (golden-ratio hue walk)."""
    hue = int((i * 0.618033988749895 % 1.0) * 180)
    hsv = np.uint8([[[hue, 200, 255]]])
    r, g, b = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
    return int(r), int(g), int(b)


def _ternary_label(t: int) -> str:
    return {1: "+1", 0: "0", -1: "-1"}.get(int(t), str(t))


# ---------------------------------------------------------------------------------------------------------------------
# Original-image overlays
# ---------------------------------------------------------------------------------------------------------------------

def draw_points(image: np.ndarray, points: np.ndarray, labels: bool = True, closed: bool = True,
                color=(0, 230, 255)) -> np.ndarray:
    """Numbered markers + connecting outline on the original image. Marker size scales with image size.

    Points are in image pixels, numbered 1..N in the order given. Sizes are proportional to the image diagonal so the
    markers read the same on a 4000 px photo and on a 500 px crop.
    """
    out = _rgb_copy(image)
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) == 0:
        return out
    d = _diag(out.shape)
    line_w = max(1, int(round(d / 550)))
    radius = max(4, int(round(d / 110)))
    font_scale = max(0.5, d / 1000)
    font_w = max(1, int(round(d / 600)))

    if len(pts) >= 2:
        poly = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [poly], bool(closed and len(pts) >= 3), BLACK, line_w + 2, LINE)  # dark halo
        cv2.polylines(out, [poly], bool(closed and len(pts) >= 3), color, line_w, LINE)
    for i, p in enumerate(pts):
        c = _ipt(p)
        cv2.circle(out, c, radius + max(1, line_w), BLACK, -1, LINE)
        cv2.circle(out, c, radius, color, -1, LINE)
        cv2.circle(out, c, max(1, radius // 4), BLACK, -1, LINE)  # precise centre dot
        if labels:
            off = radius * 1.4
            _text(out, str(i + 1), (p[0] + off, p[1] - off), font_scale, WHITE, font_w, FONT_BOLD)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Canonical-frame overlays
# ---------------------------------------------------------------------------------------------------------------------

def draw_construction(frame: np.ndarray, state: CubeState, opacity: float = 0.6) -> np.ndarray:
    """Canonical hexagon outline, visible cube edges solid, hidden edges dashed, for the given Necker state."""
    from . import cube  # local import: cube.py is owned by another module and only needed here

    base = _rgb_copy(frame)
    overlay = base.copy()
    k = _frame_scale(base)
    w = max(1, int(round(2 * k)))
    # Hidden (back) edges first so visible edges sit on top where they meet at the centre.
    for seg in cube.hidden_edges(state):
        _dashed_polyline(overlay, _to_px(seg, base), WHITE, max(1, w - 1), dash=10 * k, gap=7 * k)
    for seg in cube.visible_edges(state):
        cv2.polylines(overlay, [np.round(_to_px(seg, base)).astype(np.int32).reshape(-1, 1, 2)], False, WHITE, w, LINE)
    # Vertex labels V0..V5 just outside the hexagon (drawn opaque on the overlay too).
    for i, v in enumerate(canonical.VERTICES):
        _text_centered(overlay, f"V{i}", _to_px(v * 1.09, base)[0], 0.42 * k, WHITE, 1)
    return _blend(overlay, base, opacity)


def draw_polylines(frame: np.ndarray, polylines: list[Polyline], color=(255, 200, 40), opacity: float = 0.7,
                   width: int = 3) -> np.ndarray:
    """Semi-transparent guides; hidden=True -> dashed; arrows=True -> arrowheads along path."""
    base = _rgb_copy(frame)
    overlay = base.copy()
    k = _frame_scale(base)
    width = max(1, int(width))
    for pl in polylines:
        pts = _to_px(pl.points, base)
        if len(pts) < 2:
            continue
        if pl.hidden:
            _dashed_polyline(overlay, pts, color, max(1, width - 1), dash=4.0 * width + 4, gap=3.0 * width + 4,
                             closed=pl.closed)
        else:
            cv2.polylines(overlay, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], bool(pl.closed), color, width,
                          LINE)
        if pl.arrows:
            _arrowheads(overlay, pts, color, size=4.0 * width + 6 * k, spacing=70 * k, closed=pl.closed)
    return _blend(overlay, base, opacity)


def draw_orientation(frame: np.ndarray, field: OrientationField, step: int = 16, min_coherence: float = 0.1) -> np.ndarray:
    """Short line glyphs on a grid, length/alpha by coherence, inside the hexagon only.

    Glyphs are bucketed into four coherence bands, each blended with its own opacity, which gives the
    "alpha by coherence" look without a per-glyph blend.
    """
    out = _rgb_copy(frame)
    theta = np.asarray(field.theta, dtype=float)
    coh = np.clip(np.asarray(field.coherence, dtype=float), 0.0, 1.0)
    h, w = theta.shape[:2]
    mask = canonical.hexagon_mask(out.shape[0])
    step = max(2, int(step))
    half_len = 0.48 * step
    thick = max(1, int(round(_frame_scale(out))))

    bands = [(0.0, 0.25, 0.35), (0.25, 0.5, 0.55), (0.5, 0.75, 0.8), (0.75, 1.01, 1.0)]  # (lo, hi, opacity)
    layers = [out.copy() for _ in bands]
    drew = [False] * len(bands)
    for y in range(step // 2, min(h, out.shape[0]), step):
        for x in range(step // 2, min(w, out.shape[1]), step):
            c = coh[y, x]
            if c < min_coherence or not mask[y, x]:
                continue
            t = theta[y, x]
            dx, dy = math.cos(t) * half_len * (0.3 + 0.7 * c), math.sin(t) * half_len * (0.3 + 0.7 * c)
            for bi, (lo, hi, _) in enumerate(bands):
                if lo <= c < hi:
                    cv2.line(layers[bi], _ipt((x - dx, y - dy)), _ipt((x + dx, y + dy)), CYAN, thick, LINE)
                    drew[bi] = True
                    break
    original = out.copy()
    for bi, (_, _, op) in enumerate(bands):
        if drew[bi]:
            # Only pixels this band drew on differ from the original, so blend just those.
            changed = np.any(layers[bi] != original, axis=2)
            blended = _blend(layers[bi], out, op)
            out[changed] = blended[changed]
    return out


def draw_ternary(frame: np.ndarray, results: list[FaceResult], primary: PrimaryLine | None,
                 opacity: float = 0.35) -> np.ndarray:
    """Tint each face by ternary colour, draw its dominant orientation through the face centroid, label +1/0/-1,
    draw the primary line thick and the neutral vertical reference thin dashed white."""
    base = _rgb_copy(frame)
    k = _frame_scale(base)

    # 1) Translucent face tints.
    overlay = base.copy()
    for r in results:
        poly = np.round(_to_px(r.face.polygon, base)).astype(np.int32)
        cv2.fillPoly(overlay, [poly], TERNARY_COLOURS.get(int(r.ternary), TERNARY_COLOURS[0]), LINE)
    out = _blend(overlay, base, opacity)

    # 2) Opaque annotations on top.
    for r in results:
        poly_px = _to_px(r.face.polygon, base)
        cv2.polylines(out, [np.round(poly_px).astype(np.int32).reshape(-1, 1, 2)], True, (220, 220, 220),
                      max(1, int(round(k))), LINE)
        centre = np.mean(r.face.polygon, axis=0)
        half = 0.22  # canonical units either side of the centroid
        u = np.array([math.cos(r.dominant_theta), math.sin(r.dominant_theta)])
        a, b = _to_px(np.stack([centre - u * half, centre + u * half]), base)
        cv2.line(out, _ipt(a), _ipt(b), BLACK, max(2, int(round(5 * k))), LINE)
        cv2.line(out, _ipt(a), _ipt(b), WHITE, max(1, int(round(3 * k))), LINE)

    # 3) Neutral reference (V3 -> V0), thin dashed white.
    ref = _to_px(np.stack([canonical.VERTICES[3], canonical.VERTICES[0]]), base)
    _dashed_polyline(out, ref, WHITE, max(1, int(round(k))), dash=8 * k, gap=6 * k)

    # 4) Primary line, thick, coloured by its ternary value, with a direction arrowhead at the outer end.
    if primary is not None:
        p = _to_px(np.stack([primary.start, primary.end]), base)
        colour = TERNARY_COLOURS.get(int(primary.ternary), GOLD)
        thick = max(2, int(round(6 * k)))
        cv2.line(out, _ipt(p[0]), _ipt(p[1]), BLACK, thick + 3, LINE)
        cv2.line(out, _ipt(p[0]), _ipt(p[1]), colour, thick, LINE)
        _arrowheads(out, p, colour, size=18 * k, spacing=1e9)  # one arrow mid-way along the line
        _text(out, f"primary {primary.face_name}: {primary.angle_vs_reference_deg:+.1f} deg",
              (10 * k, base.shape[0] - 12 * k), 0.5 * k, colour, 1)

    # 5) Face labels last so no line covers them. Nudged sideways off the face's long diagonal, which is where
    #    the primary line runs.
    for r in results:
        centre = np.mean(r.face.polygon, axis=0)
        side = np.array([-math.sin(r.face.radial_theta), math.cos(r.face.radial_theta)])  # perpendicular to radial
        label_pos = _to_px(centre + 0.16 * side, base)[0]
        _text_centered(out, _ternary_label(r.ternary), label_pos, 0.8 * k,
                       TERNARY_COLOURS.get(int(r.ternary), WHITE), max(1, int(round(2 * k))), FONT_BOLD)
        _text_centered(out, f"c={r.coherence:.2f}", label_pos + np.array([0, 22 * k]), 0.42 * k, WHITE, 1)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Neighbour cluster overlay (original image)
# ---------------------------------------------------------------------------------------------------------------------

def draw_cluster(image: np.ndarray, fits: list[HexFit], continuity: list[EdgeContinuity],
                 face_results: list[list[FaceResult]] | None = None) -> np.ndarray:
    """Original image with every cell outlined (distinct colours, cell number at centroid), shared edges coloured
    green/amber/red by continuity score (>=0.9 / >=0.6 / else), and short orientation arrows either side.

    Cell numbers are 1-based on screen (cell index 0 is labelled "1"). If `face_results` is given, each cell also
    shows its three face ternary values under its number.
    """
    out = _rgb_copy(image)
    d = _diag(out.shape)
    line_w = max(1, int(round(d / 600)))
    font_scale = max(0.45, d / 1300)
    font_w = max(1, int(round(d / 700)))
    centroids = []

    for i, fit in enumerate(fits):
        pts = np.asarray(fit.image_points, dtype=float).reshape(-1, 2)
        centroids.append(pts.mean(axis=0) if len(pts) else np.zeros(2))
        colour = _distinct_colour(i)
        poly = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [poly], True, BLACK, line_w + 2, LINE)
        cv2.polylines(out, [poly], True, colour, line_w, LINE)
        _text_centered(out, str(i + 1), centroids[-1], font_scale * 1.3, colour, font_w + 1, FONT_BOLD)
        if face_results is not None and i < len(face_results) and face_results[i]:
            tern = " ".join(_ternary_label(fr.ternary) for fr in face_results[i])
            _text_centered(out, tern, centroids[-1] + np.array([0, font_scale * 40]), font_scale * 0.8, WHITE, font_w)

    for ec in continuity:
        seg = np.asarray(ec.edge.image_segment, dtype=float).reshape(2, 2)
        colour = CONTINUITY_GOOD if ec.score >= 0.9 else CONTINUITY_FAIR if ec.score >= 0.6 else CONTINUITY_POOR
        cv2.line(out, _ipt(seg[0]), _ipt(seg[1]), BLACK, 3 * line_w + 2, LINE)
        cv2.line(out, _ipt(seg[0]), _ipt(seg[1]), colour, 3 * line_w, LINE)

        # Orientation glyphs either side of the edge midpoint, on the side of the cell they belong to.
        mid = seg.mean(axis=0)
        edge_len = float(np.hypot(*(seg[1] - seg[0])))
        normal = np.array([-(seg[1] - seg[0])[1], (seg[1] - seg[0])[0]]) / max(edge_len, 1e-9)
        if ec.edge.cell_a < len(centroids) and np.dot(centroids[ec.edge.cell_a] - mid, normal) < 0:
            normal = -normal  # make `normal` point into cell a
        offset, half = 0.22 * edge_len, 0.18 * edge_len
        for side, th, tern in ((+1, ec.theta_a_image, ec.ternary_a), (-1, ec.theta_b_image, ec.ternary_b)):
            c = mid + side * normal * offset
            u = np.array([math.cos(th), math.sin(th)])
            a, b = c - u * half, c + u * half
            gcol = TERNARY_COLOURS.get(int(tern), WHITE)
            cv2.line(out, _ipt(a), _ipt(b), BLACK, 2 * line_w + 2, LINE)
            cv2.line(out, _ipt(a), _ipt(b), gcol, 2 * line_w, LINE)
            _arrowheads(out, np.stack([a, b]), gcol, size=max(6.0, 0.35 * half), spacing=1e9)
        _text(out, f"{ec.score:.2f}", mid + np.array([4.0, -4.0]), font_scale * 0.7, colour, font_w)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Scores panel and comparison sheet
# ---------------------------------------------------------------------------------------------------------------------

def scores_panel(scores: Scores, width: int = 512, height: int = 512) -> np.ndarray:
    """A rendered text panel listing the scores, null percentiles and verdict (for comparison sheets).

    Layout (top to bottom): heading; the headline affine mean residual in large type; a small table of mean / max
    residuals for affine, homography and similarity; central symmetry error; null percentiles (% of random hexagons
    that score LOWER, i.e. better); and the verdict in a coloured box (strong green / weak amber / noise red).
    """
    width, height = max(160, int(width)), max(160, int(height))
    img = np.full((height, width, 3), SHEET_BG, dtype=np.uint8)
    k = min(width, height) / 512.0
    pad = int(round(22 * k))
    muted, bright = (170, 175, 185), (235, 238, 242)
    y = pad

    def line(text, scale, colour, thick=1, font=FONT, x=pad, gap=10):
        nonlocal y
        (_, th), base = cv2.getTextSize(text, font, scale * k, max(1, int(round(thick * k))))
        y += th
        cv2.putText(img, text, (int(x), int(y)), font, scale * k, colour, max(1, int(round(thick * k))), LINE)
        y += base + int(round(gap * k))

    line("SCORES  (canonical units, R = 1)", 0.55, muted)
    line("Affine residual (mean)", 0.6, bright)
    line(f"{scores.affine_mean:.4f}", 1.7, GOLD, 3, FONT_BOLD, gap=14)

    # Residual table: columns at fixed x positions.
    cols = [pad, pad + int(190 * k), pad + int(320 * k)]
    def row(cells, colour, scale=0.55, thick=1):
        nonlocal y
        (_, th), base = cv2.getTextSize("Ag", FONT, scale * k, 1)
        y += th
        for x, text in zip(cols, cells):
            cv2.putText(img, text, (x, int(y)), FONT, scale * k, colour, max(1, int(round(thick * k))), LINE)
        y += base + int(round(8 * k))

    row(["transform", "mean", "max"], muted)
    for name, m, mx in (("affine", scores.affine_mean, scores.affine_max),
                        ("homography", scores.homography_mean, scores.homography_max),
                        ("similarity", scores.similarity_mean, scores.similarity_max)):
        row([name, f"{m:.4f}", f"{mx:.4f}"], bright)
    y += int(6 * k)
    row(["central symmetry", f"{scores.central_symmetry_error:.4f}", ""], bright)
    floor = getattr(scores, "noise_floor", float("nan"))
    if np.isfinite(floor):
        row([f"noise floor ({scores.verdict_basis})", f"{floor:.4f}", f"excess {scores.excess:+.4f}"], GOLD)
        row([f"corners +-{scores.corner_sigma_px:.2f} px ({scores.sigma_source})", "", ""], muted, scale=0.45)

    if scores.null_percentile:
        y += int(6 * k)
        line("Null percentile (% random hexagons scoring better)", 0.45, muted, gap=6)
        for metric, pct in scores.null_percentile.items():
            row([str(metric), f"{float(pct):5.1f} %", ""], bright, scale=0.5)

    # Verdict box pinned to the bottom of the panel.
    verdict = (scores.verdict or "").strip().lower()
    colour = VERDICT_COLOURS.get(verdict, (120, 120, 120))
    text = f"VERDICT: {verdict.upper() if verdict else 'n/a'}"
    if verdict:
        text += f" ({getattr(scores, 'verdict_basis', 'affine')})"
    scale, thick = 0.9 * k, max(1, int(round(2 * k)))
    (tw, th), base = cv2.getTextSize(text, FONT_BOLD, scale, thick)
    box_h = th + base + int(20 * k)
    top = max(y + int(4 * k), height - pad - box_h)
    cv2.rectangle(img, (pad, top), (width - pad, top + box_h), colour, -1, LINE)
    cv2.putText(img, text, (int((width - tw) / 2), int(top + (box_h + th) / 2)), FONT_BOLD, scale, (20, 20, 20),
                thick, LINE)
    return img


def _letterbox(image: np.ndarray, size: int) -> np.ndarray:
    """Resize to fit inside size x size keeping aspect ratio, centred on the dark background."""
    img = _rgb_copy(image)
    tile = np.full((size, size, 3), SHEET_BG, dtype=np.uint8)
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return tile
    s = size / max(h, w)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    x0, y0 = (size - nw) // 2, (size - nh) // 2
    tile[y0:y0 + nh, x0:x0 + nw] = resized
    return tile


def comparison_sheet(panels: dict[str, np.ndarray], title: str = "", tile: int = 512) -> np.ndarray:
    """Lay out labelled panels in a grid (max 3 per row), each resized to tile x tile keeping aspect, with a title bar."""
    tile = max(64, int(tile))
    k = tile / 512.0
    n = len(panels)
    cols = min(3, max(1, n))
    rows = max(1, math.ceil(n / cols)) if n else 0
    margin = int(round(12 * k))
    label_h = int(round(34 * k))
    title_h = int(round(56 * k))

    width = cols * tile + (cols + 1) * margin
    height = title_h + rows * (label_h + tile) + (rows + 1) * margin
    sheet = np.full((height, width, 3), SHEET_BG, dtype=np.uint8)

    # Title bar.
    sheet[:title_h] = SHEET_TITLE_BG
    if title:
        scale, thick = 0.95 * k, max(1, int(round(2 * k)))
        (_, th), _ = cv2.getTextSize(title, FONT_BOLD, scale, thick)
        cv2.putText(sheet, title, (margin, int((title_h + th) / 2)), FONT_BOLD, scale, (240, 240, 240), thick, LINE)

    # Tiles with labels above.
    for idx, (label, img) in enumerate(panels.items()):
        r, c = divmod(idx, cols)
        x = margin + c * (tile + margin)
        y = title_h + margin + r * (label_h + tile + margin)
        scale = 0.65 * k
        (_, th), _ = cv2.getTextSize(str(label), FONT, scale, 1)
        cv2.putText(sheet, str(label), (x + int(4 * k), y + int((label_h + th) / 2)), FONT, scale, (225, 228, 235),
                    max(1, int(round(1.5 * k))), LINE)
        sheet[y + label_h:y + label_h + tile, x:x + tile] = _letterbox(img, tile)
        cv2.rectangle(sheet, (x - 1, y + label_h - 1), (x + tile, y + label_h + tile), (70, 74, 84), 1)
    return sheet
