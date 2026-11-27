"""Sidebar: uploads, sample loader, image picker with status, and every setting.

`render()` returns (selected image_id or None, settings dict). The settings dict is the single source of truth
passed to every panel:

  state            CubeState            Necker reading
  transform        "affine" | "homography"   which fit warps the image
  rotation_offset  0..5
  construction     bool                 construction lines overlay
  templates        list[str]            enabled manifold templates (manifolds.TEMPLATE_NAMES)
  orientation      bool                 orientation-field overlay
  ternary          bool                 ternary face colouring overlay
  opacity          float                manifold opacity
  line_width       int                  manifold line width (px in the 512 px frame)
  angle_tol        float (deg)          ternary angle tolerance
  coherence_min    float                ternary coherence threshold
  edge_margin      float                fixed margin (fraction of circumradius), used when margin_mode == "fraction"
  margin_mode      "source" | "fraction"   margin from the photo's pixels (wall + buffer) or the fixed fraction
  convention       "edges" | "faces"    ternary regions (see inference.regions)
  snap             bool                 snap clicked corners onto the walls (hexengine.refine)
  wall_type        "auto" | refine.FEATURES   what the walls look like
  wall_px          float (px)           wall thickness in the photo; 0 = measure while snapping
  sigma_grad       float (px)           structure-tensor gradient pre-blur
  sigma_tensor     float (px)           structure-tensor smoothing
  thresholds       {"strong", "weak"}   LEGACY fixed-threshold verdict, logged only
  display_width    int                  width of the click image in px
"""

from __future__ import annotations

import streamlit as st

from hexengine import inference, manifolds, metrics, session
from hexengine.types import CubeState

from . import state

NECKER_OPTIONS = {"Convex / vertex-out (+1)": CubeState.CONVEX, "Concave / vertex-in (−1)": CubeState.CONCAVE}
WALL_TYPES = {"auto": "Auto", "dark_ridge": "Dark line", "bright_ridge": "Bright line", "edge": "Edge between tones"}
UPLOAD_TYPES = ["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"]


def _rotate_60() -> None:
    st.session_state["rotation_offset"] = (int(st.session_state.get("rotation_offset", 0)) + 1) % 6


def _load_samples_cb() -> None:
    settings = st.session_state.get("settings")
    use_gt = bool(st.session_state.get("samples_use_gt", True))
    ids = state.load_samples(settings, use_gt) if settings else []
    if ids:
        st.session_state["image_id"] = ids[0]
        state.flash(f"Loaded {len(ids)} sample images" + (" with ground-truth points." if use_gt else "."), "success")
    else:
        state.flash(f"No sample images found in {state.samples_dir()}.", "warning")


def _status(item: dict) -> str:
    sc = item.get("scores")
    if sc:
        return f"scored · {sc.get('verdict') or '?'}"
    return "marked" if item.get("has_points") else "unmarked"


def _save_new_uploads(files) -> str | None:
    """Store uploaded files (once per file per session). Returns the id of the last new one."""
    last = None
    for f in files or []:
        sig = (f.name, f.size)
        if sig in st.session_state["seen_uploads"]:
            continue
        last = session.save_upload(f.getvalue(), f.name)
        st.session_state["seen_uploads"].add(sig)
    return last


def render() -> tuple[str | None, dict]:
    sb = st.sidebar
    sb.title("Hex test engine")

    # --- Images -------------------------------------------------------------------------------------------------
    sb.header("Images")
    files = sb.file_uploader("Upload images", type=UPLOAD_TYPES, accept_multiple_files=True, key="uploader")
    new_id = _save_new_uploads(files)
    c1, c2 = sb.columns([3, 2])
    c1.button("Load sample images", key="load_samples", on_click=_load_samples_cb, use_container_width=True)
    c2.checkbox("Use ground-truth points", value=True, key="samples_use_gt",
                help="When loading the bundled samples, also load their exact vertices from samples/ground_truth.json.")

    uploads = session.list_uploads()
    st.session_state["uploads"] = uploads
    ids = [u["image_id"] for u in uploads]
    labels = {u["image_id"]: f"{u['filename']} — {_status(u)}" for u in uploads}
    if new_id and new_id in ids:
        st.session_state["image_id"] = new_id
    if st.session_state.get("image_id") not in ids:
        st.session_state.pop("image_id", None)

    image_id = None
    if ids:
        image_id = sb.radio("Image", ids, format_func=lambda i: labels.get(i, i), key="image_id")
    else:
        sb.info("No images yet: upload some, or load the samples.")

    # --- Normalisation (and the retired cube reading) ----------------------------------------------------------
    sb.header("Normalisation")
    cube_layers = sb.checkbox("Show cube / Necker layers (unconfirmed)", value=False, key="cube_layers",
                              help="The cube reading, ternary states, manifold paths and texture lock. Tested and not "
                                   "supported for natural hexagons (see the findings report and CLOSING_NOTE.md), so "
                                   "they are hidden by default. Shape and size measurements don't depend on them.")
    necker = list(NECKER_OPTIONS)[0]
    if cube_layers:
        necker = sb.radio("Necker state", list(NECKER_OPTIONS), key="necker",
                          help="Which interior 'Y' is the near corner. Flips the cube edges, signs and manifold "
                               "visibility. Sign convention: +1 = structure runs along the CONVEX cube's edges "
                               "(spokes to V1, V3, V5); −1 = along the CONCAVE corner's edges (spokes to V0, V2, V4). "
                               "Calibrated on the tumbling-block tile, whose real edges score +1, +1, +1 under Convex.")
    transform = sb.radio("Warp transform", ["affine", "homography"], key="transform", horizontal=True,
                         help="Which least-squares fit is used to warp the image into the canonical frame.")
    # The offset labels the vertices of ONE image, so it is remembered per image (and restored from its saved
    # record): show the selected image's own value whenever the selection changes.
    if image_id is not None and st.session_state.get("offset_image") != image_id:
        st.session_state["rotation_offset"] = state.offset_for(image_id)
        st.session_state["offset_image"] = image_id
    st.session_state.setdefault("rotation_offset", 0)
    r1, r2 = sb.columns([3, 2])
    rotation = r1.slider("Rotation offset", 0, 5, key="rotation_offset",
                         help="Which ordered point becomes V0 (each step = 60°). Odd values swap the cube's 'Y'. "
                              "Remembered per image.")
    r2.button("Rotate 60°", on_click=_rotate_60, key="rotate60", use_container_width=True)
    if image_id is not None:
        st.session_state["offsets"][image_id] = int(rotation)

    # --- Corners -------------------------------------------------------------------------------------------------
    sb.header("Corners")
    snap = sb.checkbox("Snap corners to walls", value=True, key="snap",
                       help="Fit a straight line along each wall and use the intersections as corners (sub-pixel). "
                            "Your clicks only need to be roughly right. If the walls cannot be found cleanly, your "
                            "clicks are used and the app says so.")
    wall_type = sb.selectbox("Wall type", list(WALL_TYPES), key="wall_type", disabled=not snap,
                             format_func=lambda k: WALL_TYPES[k],
                             help="What the cell boundaries look like. Auto tries all three and keeps the best fit.")
    wall_px = sb.number_input("Wall width (px, 0 = measure it)", 0.0, 60.0, 0.0, 0.5, key="wall_px",
                              help="Thickness of the walls in the photo. 0 = measured while snapping (or 3 px if "
                                   "not snapped). Sets the edge margin below.")

    # --- Overlays -----------------------------------------------------------------------------------------------
    sb.header("Overlays")
    orientation = sb.checkbox("Orientation field", value=False, key="ov_orientation")
    # Defaults while the cube layers are hidden: nothing cube-related is drawn.
    construction, templates, ternary, opacity, line_width = False, [], False, 0.7, 3
    angle_tol, coherence_min, convention = 20.0, 0.15, inference.DEFAULT_CONVENTION
    margin_mode, margin_buffer, edge_margin = "source", float(inference.SOURCE_MARGIN_BUFFER_PX), 0.12
    if cube_layers:
        construction = sb.checkbox("Construction lines (cube edges)", value=True, key="ov_construction")
        templates = [name for i, name in enumerate(manifolds.TEMPLATE_NAMES)
                     if sb.checkbox(f"Manifold: {name}", value=(i == 0), key=f"ov_tpl_{i}")]
        ternary = sb.checkbox("Ternary colouring", value=False, key="ov_ternary")
        opacity = sb.slider("Manifold opacity", 0.0, 1.0, 0.7, 0.05, key="ov_opacity")
        line_width = sb.slider("Manifold line width", 1, 10, 3, key="ov_width")

        # --- Inference ------------------------------------------------------------------------------------------
        sb.header("Ternary inference")
        angle_tol = sb.slider("Angle tolerance (°)", 1.0, 45.0, 20.0, 1.0, key="angle_tol",
                              help="How close a face's dominant orientation must be to its radial/tangential line.")
        coherence_min = sb.slider("Coherence minimum", 0.0, 1.0, 0.15, 0.01, key="coherence_min",
                                  help="Faces with less coherent structure than this get ternary 0.")
        margin_mode = sb.radio("Edge margin", ["source", "fraction"], key="margin_mode", horizontal=True,
                               format_func=lambda m: "Photo pixels" if m == "source" else "Fixed fraction",
                               help="Band kept clear of every region boundary so the outline is not read as "
                                    "structure. Photo pixels (default): wall width + a buffer, measured in the "
                                    "original photo, so small cells enlarged by normalising stay clean. Fixed "
                                    "fraction: the old slider below. Never smaller than the blur reach "
                                    "(4·σ_tensor + 4·σ_grad + 2 frame px).")
        margin_buffer = sb.slider("Margin buffer (photo px)", 0.0, 20.0, float(inference.SOURCE_MARGIN_BUFFER_PX),
                                  1.0, key="margin_buffer_px", disabled=margin_mode != "source",
                                  help="Added to the wall width. 8 px is the smallest value that removed the false "
                                       "texture lock in soap froth on the baseline photos (4 px did not) while "
                                       "keeping the snowflakes' real one.")
        edge_margin = sb.slider("Fixed margin (fraction of radius)", 0.0, 0.3, 0.12, 0.01, key="edge_margin",
                                disabled=margin_mode != "fraction")
        convention = sb.radio("Ternary regions", list(inference.CONVENTIONS), key="convention", horizontal=True,
                              format_func=lambda c: "Edges (calibrated)" if c == "edges" else "Faces (original)",
                              help="Edges: read the regions centred on the reading's own cube edges, so a real "
                                   "convex cube scores +1 under Convex. Faces: the original rule (reads the faces, "
                                   "whose boundaries are the edges); kept for comparison.")

    with sb.expander("Advanced"):
        st.caption("Legacy fixed-threshold verdict (logged next to the calibrated one, not used for it):")
        strong = st.number_input("Legacy 'strong' if residual <", 0.0, 1.0,
                                 float(metrics.DEFAULT_THRESHOLDS["strong"]), 0.005, format="%.3f", key="thr_strong")
        weak = st.number_input("Legacy 'weak' if residual <", 0.0, 2.0,
                               float(metrics.DEFAULT_THRESHOLDS["weak"]), 0.005, format="%.3f", key="thr_weak")
        if weak < strong:
            st.warning("'weak' threshold is below 'strong'; nothing will be rated weak.")
        sigma_grad = st.slider("σ gradient (px)", 0.5, 5.0, 1.5, 0.25, key="sigma_grad",
                               help="Pre-blur before the Sobel derivatives (structure tensor).")
        sigma_tensor = st.slider("σ tensor (px)", 1.0, 20.0, 6.0, 0.5, key="sigma_tensor",
                                 help="Smoothing of the structure tensor: larger = broader, steadier orientations.")
        display_width = st.slider("Click image width (px)", 300, 1400, 700, 50, key="display_width",
                                  help="Resolution of the image you click on (points are always stored in "
                                       "full-resolution pixels).")
        fit_click = st.checkbox("Fit click image to column", value=True, key="fit_click_image",
                                help="Scale the click image to the column width. Untick to show it at exactly "
                                     "the width above (it may then be cut off in narrow windows).")
        st.caption(f"Data folder: `{session.data_dir()}`")

    settings = {
        "state": NECKER_OPTIONS[necker],
        "transform": transform,
        "rotation_offset": int(rotation),
        "construction": construction,
        "templates": templates,
        "orientation": orientation,
        "ternary": ternary,
        "opacity": float(opacity),
        "line_width": int(line_width),
        "angle_tol": float(angle_tol),
        "coherence_min": float(coherence_min),
        "edge_margin": float(edge_margin),
        "cube_layers": bool(cube_layers),
        "margin_mode": margin_mode,
        "margin_buffer_px": float(margin_buffer),
        "convention": convention,
        "snap": bool(snap),
        "wall_type": wall_type,
        "wall_px": float(wall_px),
        "sigma_grad": float(sigma_grad),
        "sigma_tensor": float(sigma_tensor),
        "thresholds": {"strong": float(strong), "weak": float(weak)},
        "display_width": int(display_width),
        "fit_click_image": bool(fit_click),
    }
    st.session_state["settings"] = settings
    return image_id, settings
