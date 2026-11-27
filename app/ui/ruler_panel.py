"""'Calibrate scale (mm)' panel: two clicks on a ruler + the known distance -> pixels per mm, then the marked cell's
width in millimetres and its size class (hexengine.morphometry).

The ruler clicks use their own click image, so they never mix with the hexagon's six points. Points are stored in
full-resolution pixels per image (state.ruler_for) and saved with the image's record.
"""

from __future__ import annotations

import numpy as np
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from .compat import FILL

from hexengine import morphometry, render

from . import compute, state

RULER_COLOUR = (255, 0, 255)


def _clear(image_id: str) -> None:
    state.ruler_for(image_id)["points"] = []
    st.session_state["nonce"][f"ruler_{image_id}"] = st.session_state["nonce"].get(f"ruler_{image_id}", 0) + 1


def render_panel(image_id: str, settings: dict, analysis) -> None:
    ruler = state.ruler_for(image_id)
    with st.expander("Calibrate scale (mm)", expanded=not ruler["points"]):
        st.caption("Click two points on the ruler (for example two centimetre marks), then enter the distance between "
                   "them. Use the longest stretch of ruler you can, and the ruler that lies on the comb itself.")
        c1, c2 = st.columns([3, 2])
        # The box follows the stored distance when that changes elsewhere (e.g. a loaded session), and the user's
        # typing otherwise. Newer Streamlit ignores value= once a keyed widget exists, so set its state directly.
        key, seen = f"ruler_mm_{image_id}", st.session_state.setdefault("ruler_mm_seen", {})
        if seen.get(image_id) != float(ruler["mm"]):
            st.session_state[key] = float(ruler["mm"])
        mm = c2.number_input("Known distance (mm)", min_value=0.1, max_value=1000.0, step=1.0, key=key)
        ruler["mm"] = seen[image_id] = float(mm)
        c2.button("Clear ruler points", key=f"ruler_clear_{image_id}", on_click=_clear, args=(image_id,),
                  disabled=not ruler["points"])
        disp, scale_f = compute.display_base(image_id, settings["display_width"])
        shown = np.array(disp)
        if ruler["points"]:
            pts = np.asarray(ruler["points"], float) * scale_f
            shown = render.draw_points(shown, pts, closed=False, color=RULER_COLOUR)
        wkey = f"ruler_{image_id}"
        value = streamlit_image_coordinates(shown, **FILL,
                                            key=f"{wkey}_img{st.session_state['nonce'].get(wkey, 0)}")
        if value and "x" in value:
            sig = (value["x"], value["y"], value.get("width"), value.get("height"))
            if st.session_state["last_click"].get(wkey) != sig and len(ruler["points"]) < 2:
                st.session_state["last_click"][wkey] = sig
                full_h, full_w = compute.image(image_id).shape[:2]
                ruler["points"].append([value["x"] * full_w / float(value["width"] or shown.shape[1]),
                                        value["y"] * full_h / float(value["height"] or shown.shape[0])])
                st.rerun()
        scale = state.scale_for(image_id)
        if scale is None:
            c1.info(f"{len(ruler['points'])} of 2 ruler points marked.")
            return
        c1.metric("Scale", f"{scale.px_per_mm:.3f} px/mm", delta=f"{scale.mm_per_px:.4f} mm/px", delta_color="off",
                  help=f"{scale.length_px:.1f} px between the two points = {scale.known_mm:g} mm.")

    scale = state.scale_for(image_id)
    if scale is None or analysis is None:
        return
    m = morphometry.measure_cell(analysis.points_used, scale, analysis.wall_width_px)
    st.subheader("Cell size (mm)")
    c = st.columns(4)
    c[0].metric("Flat-to-flat width", f"{m.width_mm:.3f} mm",
                help="Mean of the three opposite-wall distances, wall centre to wall centre: the quantity a "
                     "'10 cells = 48 mm' ruler count measures." + ("" if analysis.snapped else
                                                                   " Measured from your clicks (not snapped)."))
    c[1].metric("Anisotropy", f"{m.anisotropy_mm:.3f} mm", help="Widest minus narrowest of the three widths.")
    c[2].metric("Inner width (approx.)", f"{m.inner_mm:.3f} mm",
                help="Width minus the measured wall thickness. Rough on small images.")
    c[3].metric("Size class", m.size_class, help="Indicative Apis mellifera ranges: "
                + "; ".join(f"{lo:.2f}-{hi:.2f} mm {lab}" for lo, hi, lab in morphometry.BENCHMARKS))
    st.caption("Widths: " + ", ".join(f"{w:.3f}" for w in m.widths_mm) + " mm. Accuracy depends on the ruler lying "
               "in the same plane as the comb; a ruler resting on top of the cells is nearer the camera.")
