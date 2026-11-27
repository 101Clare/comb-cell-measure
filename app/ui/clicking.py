"""The click-to-mark widget: shows an image, turns clicks into full-resolution points, Undo / Clear / Auto-detect.

How clicks become points
------------------------
1. The full-resolution image is resized to the chosen display width (compute.display_base) and every cell's points
   are drawn on that small copy with render.draw_points (scaled by the same factor), so the user sees numbered
   markers as they click.
2. streamlit_image_coordinates shows it and returns {"x", "y", "width", "height"}: the click position in DISPLAYED
   pixels plus the displayed size. Full-resolution coordinates are therefore
       x_full = x * W_full / width,   y_full = y * H_full / height
   Because we use the size the browser reports, this stays exact when the image is scaled to fit the column
   (the default, "Fit click image to column") or shown at exactly the chosen display width.
3. The component keeps returning its last value on every rerun, so a click is only accepted if it differs from the
   last one we consumed (stored per widget key in st.session_state["last_click"]). Undo / Clear also bump a nonce
   in the widget key, which resets the component so the same spot can be clicked again.
4. Speed matters: a click that arrives while the script is still running can be lost. So the image lives in an
   st.fragment (a click reruns only the image, not the whole page with its hidden tabs), and the click is read from
   st.session_state BEFORE the image is drawn, so the new marker appears in that same run with no extra st.rerun().
   Only when the cell reaches six points do we rerun the whole app, so the results panels appear.
"""

from __future__ import annotations

import numpy as np
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from hexengine import render, transforms

from . import compute, state


def _widget_key(prefix: str, image_id: str, cell: int) -> str:
    return f"{prefix}_{image_id}_{cell}"


# ---- button callbacks (run before the script, so they can change state freely) ---------------------------------------

def _undo(image_id: str, cell: int, wkey: str) -> None:
    cells = state.get_cells(image_id)
    if cells[cell]:
        cells[cell].pop()
    st.session_state["cluster"].pop(image_id, None)   # the cluster summary no longer matches the points
    state.bump_nonce(wkey)


def _clear(image_id: str, cell: int, wkey: str) -> None:
    state.get_cells(image_id)[cell] = []
    st.session_state["cluster"].pop(image_id, None)
    state.bump_nonce(wkey)


def _autodetect(image_id: str, cell: int, wkey: str) -> None:
    cells = state.get_cells(image_id)
    near = cells[cell][-1] if cells[cell] else None   # prefer the hexagon around the last click, if any
    found = compute.detect_points(image_id, near=near)
    if found is None:
        state.flash("Auto-detect found no convex six-corner outline. Click the vertices manually.", "warning")
        return
    state.set_cell(image_id, cell, found)
    st.session_state["cluster"].pop(image_id, None)
    state.bump_nonce(wkey)
    state.flash("Auto-detected six points (a suggestion: Undo and re-click any that are off).", "info")


def _save(image_id: str) -> None:
    state.save(image_id, st.session_state["settings"])


def _ground_truth(image_id: str) -> None:
    if state.apply_ground_truth(image_id, st.session_state["settings"]):
        state.flash("Loaded the ground-truth points for this sample.", "success")


# ---- the widget ------------------------------------------------------------------------------------------------------

def winding_label(points) -> str:
    """'clockwise' / 'counter-clockwise' on screen (y down) for >= 3 points, else ''."""
    if len(points) < 3:
        return ""
    return "clockwise (CW)" if transforms.signed_area(np.asarray(points, float)) > 0 else "counter-clockwise (CCW)"


def render_clicker(image_id: str, settings: dict, cell: int, prefix: str, colours: list | None = None,
                   max_points: int = 6) -> None:
    """Draw the click widget for `cell` of `image_id` (other cells drawn too, in their colours)."""
    cells = state.get_cells(image_id)
    wkey = _widget_key(prefix, image_id, cell)
    pts = cells[cell]

    # Buttons first: their callbacks have already run, so the image below reflects them.
    b = st.columns(3) + st.columns(2)    # two short rows stay readable in narrow columns
    b[0].button("Undo", key=f"{wkey}_undo", on_click=_undo, args=(image_id, cell, wkey), disabled=not pts,
                use_container_width=True)
    b[1].button("Clear", key=f"{wkey}_clear", on_click=_clear, args=(image_id, cell, wkey), disabled=not pts,
                use_container_width=True)
    b[2].button("Auto-detect", key=f"{wkey}_detect", on_click=_autodetect, args=(image_id, cell, wkey),
                help="Secondary helper: contour + approxPolyDP finds a convex six-corner outline near your last "
                     "click (or the image centre).", use_container_width=True)
    b[3].button("Save points", key=f"{wkey}_save", on_click=_save, args=(image_id,), type="primary",
                use_container_width=True)
    if state.ground_truth_cells(state.filename_of(image_id)):
        b[4].button("Use ground-truth points", key=f"{wkey}_gt", on_click=_ground_truth, args=(image_id,),
                    help="Bundled sample: load its exact vertices from samples/ground_truth.json.",
                    use_container_width=True)

    _click_image(image_id, settings, cell, wkey, colours, max_points)


@st.fragment
def _click_image(image_id: str, settings: dict, cell: int, wkey: str, colours: list | None, max_points: int) -> None:
    """The clickable image. As a fragment, a click reruns only this function (see the module docstring, step 4)."""
    cells = state.get_cells(image_id)
    pts = cells[cell]
    disp, scale = compute.display_base(image_id, settings["display_width"])
    full_h, full_w = compute.image(image_id).shape[:2]
    img_key = f"{wkey}_img{st.session_state['nonce'].get(wkey, 0)}"

    # Consume a new click first, so the marker is drawn in this same run.
    completed = False
    value = st.session_state.get(img_key)
    if value and "x" in value:
        sig = (value["x"], value["y"], value.get("width"), value.get("height"))
        if st.session_state["last_click"].get(wkey) != sig:
            st.session_state["last_click"][wkey] = sig
            if len(pts) >= max_points:
                st.toast(f"Already {max_points} points: Undo or Clear first.")
            else:
                # displayed px -> full-resolution px, using the size the browser actually drew
                dw = float(value.get("width") or disp.shape[1])
                dh = float(value.get("height") or disp.shape[0])
                x = float(np.clip(value["x"] * full_w / dw, 0, full_w - 1))
                y = float(np.clip(value["y"] * full_h / dh, 0, full_h - 1))
                pts.append([x, y])
                st.session_state["cluster"].pop(image_id, None)
                completed = len(pts) == max_points

    # Display copy with every cell's points (active cell drawn last so it sits on top).
    shown = np.array(disp)
    order = [i for i in range(len(cells)) if i != cell] + [cell]
    for i in order:
        if cells[i]:
            colour = colours[i] if colours else render.CYAN
            shown = render.draw_points(shown, np.asarray(cells[i], float) * scale, closed=len(cells[i]) == 6,
                                       color=colour)

    if settings.get("fit_click_image", True):
        # Scale to the column width in the browser; the returned width/height are the rendered size.
        streamlit_image_coordinates(shown, use_column_width="always", key=img_key)
    else:
        streamlit_image_coordinates(shown, width=shown.shape[1], key=img_key)

    info = f"{len(pts)}/{max_points} points"
    if len(pts) >= 3:
        info += f" · winding {winding_label(pts)}"
    st.caption(info + f" · image {full_w}×{full_h} px, drawn at {shown.shape[1]} px wide")

    if completed:
        st.rerun()   # whole app: the results panels outside this fragment now have six points
