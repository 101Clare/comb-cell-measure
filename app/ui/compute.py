"""Cached wrappers around the hexengine pipeline. No geometry is done here: every number comes from hexengine.

Caching strategy
----------------
* `st.cache_resource` (not `cache_data`) is used for images, warps and orientation fields: these are large numpy
  arrays that we never modify (all render functions copy their inputs), so sharing one object avoids re-pickling
  megabytes on every rerun.
* Keys are plain hashable values: image_id (a content hash), the six points as a tuple of rounded floats, the
  rotation offset and the transform name, plus whichever settings the step actually depends on. So changing,
  say, the ternary tolerance re-runs only the cheap face analysis, never the warp or the structure tensor.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import streamlit as st

from hexengine import detect, inference, manifolds, metrics, refine, render, session, transforms
from hexengine.types import CubeState, FaceResult, HexFit, OrientationField, PrimaryLine, Scores, TextureLock

PointsKey = tuple[tuple[float, float], ...]
DEFAULT_WALL_PX = 3.0      # wall width assumed for the margin when it was neither measured nor set
SNAPPED_COLOUR = (255, 200, 40)


def points_key(points) -> PointsKey:
    """Hashable, rounding-stable key for a list of (x, y) points."""
    return tuple((round(float(x), 3), round(float(y), 3)) for x, y in np.asarray(points, float).reshape(-1, 2))


# ---------------------------------------------------------------------------------------------------------------------
# Start-up and images
# ---------------------------------------------------------------------------------------------------------------------

@st.cache_resource(show_spinner="Building the random-hexagon null baseline (first run only, a few seconds)...")
def warm_null_baseline() -> bool:
    """Compute metrics.null_distribution() once per server process (it is lru_cached inside hexengine too)."""
    metrics.null_distribution()
    return True


@st.cache_resource(max_entries=32, show_spinner=False)
def image(image_id: str) -> np.ndarray:
    """Full-resolution RGB uint8 image for an upload (read-only array, shared between reruns)."""
    img = session.load_image(image_id)
    img.setflags(write=False)
    return img


@st.cache_resource(max_entries=32, show_spinner=False)
def display_base(image_id: str, width: int) -> tuple[np.ndarray, float]:
    """The image resized to `width` px wide for the click widget, and the scale factor display/full."""
    full = image(image_id)
    h, w = full.shape[:2]
    width = max(50, int(width))
    scale = width / float(w)
    disp = cv2.resize(full, (width, max(1, int(round(h * scale)))),
                      interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    disp.setflags(write=False)
    return disp, scale


def detect_points(image_id: str, near=None) -> np.ndarray | None:
    """Secondary auto-detection (hexengine.detect) on the full-resolution image."""
    return detect.detect_hexagon(image(image_id), near=near)


# ---------------------------------------------------------------------------------------------------------------------
# Fit, warp, field
# ---------------------------------------------------------------------------------------------------------------------

@st.cache_resource(max_entries=512, show_spinner=False)
def fit(pk: PointsKey, rotation_offset: int) -> HexFit:
    """transforms.fit_hexagon (raises ValueError for degenerate point sets; exceptions are not cached)."""
    return transforms.fit_hexagon(np.asarray(pk, dtype=float), rotation_offset)


@st.cache_resource(max_entries=128, show_spinner=False)
def warped(image_id: str, pk: PointsKey, rotation_offset: int, transform: str) -> np.ndarray:
    """The image warped into the 512 px canonical frame, outside-hexagon pixels masked."""
    out = transforms.warp_to_canonical(image(image_id), fit(pk, rotation_offset), transform)
    out.setflags(write=False)
    return out


@st.cache_resource(max_entries=128, show_spinner=False)
def field(image_id: str, pk: PointsKey, rotation_offset: int, transform: str, sigma_grad: float = 1.5,
          sigma_tensor: float = 6.0) -> OrientationField:
    """Structure-tensor orientation field of the warped image."""
    return inference.orientation_field(warped(image_id, pk, rotation_offset, transform),
                                       sigma_grad=sigma_grad, sigma_tensor=sigma_tensor)


def _sigmas(settings: dict) -> tuple[float, float]:
    return float(settings.get("sigma_grad", 1.5)), float(settings.get("sigma_tensor", 6.0))


@st.cache_resource(max_entries=64, show_spinner=False)
def analysed_region_outline(state_name: str, size: int, edge_margin: float, sigma_grad: float,
                            sigma_tensor: float, convention: str = inference.DEFAULT_CONVENTION) -> list[np.ndarray]:
    """Contours (frame px) of the inner regions that analyse_faces actually reads (after the edge margin)."""
    state = CubeState[state_name]
    margin = inference.edge_margin_px(size, edge_margin, sigma_tensor, sigma_grad)
    masks = inference.face_masks(state, size, edge_margin_frac=edge_margin, min_margin_px=margin, convention=convention)
    contours = []
    for m in masks:
        found, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(found)
    return contours


def draw_analysed_region(frame: np.ndarray, settings: dict, margin_frac: float) -> np.ndarray:
    """Faint outline of the analysed (non-margin) part of each region."""
    sg, stn = _sigmas(settings)
    contours = analysed_region_outline(settings["state"].name, frame.shape[0], round(float(margin_frac), 4), sg, stn,
                                       settings.get("convention", inference.DEFAULT_CONVENTION))
    layer = np.array(frame)
    cv2.drawContours(layer, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.addWeighted(layer, 0.55, np.asarray(frame), 0.45, 0.0)


# ---------------------------------------------------------------------------------------------------------------------
# Snapping clicks onto the walls (hexengine.refine)
# ---------------------------------------------------------------------------------------------------------------------

@st.cache_resource(max_entries=256, show_spinner="Snapping corners to the walls...")
def refined(image_id: str, pk: PointsKey, wall_type: str, wall_px: float) -> tuple[refine.Refined | None, str]:
    """Refine six clicked corners onto the walls. Returns (result or None, note for the user).

    wall_type "auto" tries every wall type (and, with wall_px = 0, several scales) and keeps the best trusted fit;
    otherwise the given type is used with scale = wall_px / 2 (or 2 px when wall_px = 0)."""
    img = image(image_id)
    rough = np.asarray(pk, dtype=float)
    try:
        if wall_type == "auto":
            scales = (max(0.75, wall_px / 2.0),) if wall_px > 0 else (1.0, 2.0, 4.0)
            r = refine.refine_auto(img, rough, scales=scales)
        else:
            r = refine.refine_hexagon(img, rough, wall_type, scale=max(0.75, wall_px / 2.0) if wall_px > 0 else 2.0,
                                      bootstrap=100)
    except ValueError as exc:
        return None, f"Not snapped: {exc}. Using your clicks."
    if not r.qc_pass:
        return r, "Not snapped: " + "; ".join(r.qc_reasons) + ". Using your clicks."
    return r, (f"Snapped to the walls ({r.kind.replace('_', ' ')}): corners moved {np.mean(r.shift):.1f} px on "
               f"average, located to ±{r.sigma_px:.2f} px; wall ≈ {r.wall_width_px:.1f} px wide.")


# ---------------------------------------------------------------------------------------------------------------------
# One full analysis
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class Analysis:
    """Everything the UI shows for one six-point cell under the current settings."""

    fit: HexFit
    scores: Scores
    warped: np.ndarray
    field: OrientationField
    faces: list[FaceResult]
    primary: PrimaryLine | None
    points_used: np.ndarray        # the corners actually fitted (snapped if snapping succeeded, else the clicks)
    snapped: bool
    snap_note: str
    margin_frac: float             # edge margin used, as a fraction of the circumradius
    margin_note: str
    lock: TextureLock
    wall_width_px: float = 0.0     # measured while snapping (0 when the clicks were used)


def margin_for(settings: dict, radius_px: float, wall_px_measured: float | None) -> tuple[float, str]:
    """The edge margin (fraction of the circumradius) and a note saying how it was set."""
    if settings.get("margin_mode", "source") == "fraction":
        frac = float(settings.get("edge_margin", inference.DEFAULT_EDGE_MARGIN_FRAC))
        return frac, f"edge margin {frac:.2f} of the radius (fixed fraction)"
    wall = float(settings.get("wall_px", 0.0)) or (wall_px_measured if wall_px_measured else DEFAULT_WALL_PX)
    buffer = float(settings.get("margin_buffer_px", inference.SOURCE_MARGIN_BUFFER_PX))
    frac = inference.source_margin_frac(wall, radius_px, buffer)
    src = "set" if settings.get("wall_px") else ("measured" if wall_px_measured else "assumed")
    return frac, (f"edge margin = wall {wall:.1f} px ({src}) + {buffer:.0f} px in the photo = {frac:.2f} of the "
                  f"radius")


def analyse(image_id: str, points, settings: dict) -> Analysis:
    """Run the whole pipeline for six points (raises ValueError if the points are degenerate)."""
    pk_click = points_key(points)
    r, note = (None, "Snapping is off: using your clicks.")
    if settings.get("snap", True):
        r, note = refined(image_id, pk_click, settings.get("wall_type", "auto"), float(settings.get("wall_px", 0.0)))
    snapped = r is not None and r.qc_pass
    pk = points_key(r.points) if snapped else pk_click
    off, kind = int(settings["rotation_offset"]), settings["transform"]
    f = fit(pk, off)
    scores = metrics.compute_scores(f, thresholds=settings["thresholds"], verdict_kind=kind,
                                    corner_sigma_px=r.sigma_px if snapped else None)
    sg, stn = _sigmas(settings)
    fld = field(image_id, pk, off, kind, sg, stn)
    margin_frac, margin_note = margin_for(settings, scores.radius_px, r.wall_width_px if snapped else None)
    state: CubeState = settings["state"]
    faces = inference.analyse_faces(fld, state, angle_tol_deg=settings["angle_tol"],
                                    coherence_min=settings["coherence_min"], edge_margin_frac=margin_frac,
                                    sigma_tensor=stn, sigma_grad=sg,
                                    convention=settings.get("convention", inference.DEFAULT_CONVENTION))
    return Analysis(fit=f, scores=scores, warped=warped(image_id, pk, off, kind), field=fld, faces=faces,
                    primary=inference.primary_line(faces, state), points_used=np.asarray(pk, float),
                    snapped=snapped, snap_note=note, margin_frac=margin_frac, margin_note=margin_note,
                    lock=inference.texture_lock(fld, margin_frac, stn, sg),
                    wall_width_px=r.wall_width_px if snapped else 0.0)


def try_analyse(image_id: str, points, settings: dict) -> tuple[Analysis | None, str | None]:
    """(analysis, None) for a complete valid cell, (None, reason) otherwise."""
    if len(points) != 6:
        # No live count here: while clicking, only the click image reruns, so a count would go stale.
        return None, "fewer than 6 points marked (the count is shown under the image)"
    try:
        return analyse(image_id, points, settings), None
    except ValueError as exc:
        return None, str(exc)


# ---------------------------------------------------------------------------------------------------------------------
# Overlays on the normalised image
# ---------------------------------------------------------------------------------------------------------------------

def overlay_key(settings: dict) -> tuple:
    """The subset of settings that affects the normalised overlay image (hashable)."""
    return (settings["state"].name, float(settings["angle_tol"]), float(settings["coherence_min"]),
            float(settings.get("edge_margin", 0.12)), _sigmas(settings), bool(settings["construction"]),
            tuple(settings["templates"]), bool(settings["orientation"]), bool(settings["ternary"]),
            float(settings["opacity"]), int(settings["line_width"]), snap_key(settings))


def snap_key(settings: dict) -> tuple:
    """Settings that change which corners are fitted, the margin or the ternary regions (hashable)."""
    return (bool(settings.get("snap", True)), settings.get("wall_type", "auto"), float(settings.get("wall_px", 0.0)),
            settings.get("margin_mode", "source"), settings.get("convention", inference.DEFAULT_CONVENTION),
            float(settings.get("margin_buffer_px", inference.SOURCE_MARGIN_BUFFER_PX)))


def compose(a: Analysis, settings: dict, *, construction=None, templates=None, orientation=None,
            ternary=None) -> np.ndarray:
    """Normalised image with the selected overlays. Keyword arguments override the sidebar toggles."""
    construction = settings["construction"] if construction is None else construction
    templates = settings["templates"] if templates is None else templates
    orientation = settings["orientation"] if orientation is None else orientation
    ternary = settings["ternary"] if ternary is None else ternary
    state = settings["state"]

    out = np.array(a.warped)
    if orientation:
        out = render.draw_orientation(out, a.field)
    if ternary:
        out = render.draw_ternary(out, a.faces, a.primary)
        out = draw_analysed_region(out, settings, a.margin_frac)   # only the inner part of each region is read
    if construction:
        out = render.draw_construction(out, state)
    for name in templates:
        out = render.draw_polylines(out, manifolds.template(name, state), opacity=settings["opacity"],
                                    width=settings["line_width"])
    return out


@st.cache_resource(max_entries=128, show_spinner=False)
def _overlay_cached(image_id: str, pk: PointsKey, off: int, kind: str, okey: tuple, thresholds: tuple,
                    _settings: dict) -> np.ndarray:
    # `_settings` (leading underscore = not hashed) carries the same values as the hashed key arguments.
    out = compose(analyse(image_id, pk, _settings), _settings)
    out.setflags(write=False)
    return out


def overlay(image_id: str, points, settings: dict) -> np.ndarray:
    """Cached `compose` with the current sidebar toggles."""
    return _overlay_cached(image_id, points_key(points), int(settings["rotation_offset"]), settings["transform"],
                           overlay_key(settings), tuple(sorted(settings["thresholds"].items())), settings)


def points_image(image_id: str, cells: list, colours: list | None = None, max_side: int | None = 1600) -> np.ndarray:
    """Original image (optionally downscaled for speed) with every cell's numbered points drawn."""
    full = image(image_id)
    scale = 1.0
    if max_side and max(full.shape[:2]) > max_side:
        scale = max_side / float(max(full.shape[:2]))
        full = cv2.resize(full, (int(full.shape[1] * scale), int(full.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    out = full
    for i, pts in enumerate(cells):
        if len(pts):
            colour = colours[i] if colours else render.CYAN
            out = render.draw_points(out, np.asarray(pts, float) * scale, closed=len(pts) == 6, color=colour)
    return np.array(out)


def clicks_and_snapped(image_id: str, points, a: Analysis, max_side: int | None = 1600) -> np.ndarray:
    """Original image with the clicks (cyan) and, when snapping succeeded, the snapped corners (gold) on top."""
    out = points_image(image_id, [points], max_side=max_side)
    if a.snapped:
        full = image(image_id)
        scale = out.shape[1] / float(full.shape[1])
        out = render.draw_points(out, a.points_used * scale, labels=False, color=SNAPPED_COLOUR)
    return out


def comparison_panels(image_id: str, points, a: Analysis, settings: dict) -> dict[str, np.ndarray]:
    """The standard per-image comparison-sheet panels (dict order = sheet order)."""
    tpl = settings["templates"] or list(manifolds.TEMPLATE_NAMES)
    return {
        "Original: clicks (cyan), snapped (gold)" if a.snapped else "Original + points":
            clicks_and_snapped(image_id, points, a),
        "Normalised": np.array(a.warped),
        "Manifolds": compose(a, settings, construction=True, templates=tpl, orientation=False, ternary=False),
        "Orientation field": compose(a, settings, construction=False, templates=[], orientation=True, ternary=False),
        "Ternary": compose(a, settings, construction=False, templates=[], orientation=False, ternary=True),
        "Scores": render.scores_panel(a.scores),
    }
