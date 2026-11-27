"""'Single image' tab: mark six points, then original | normalised side by side, score tiles, verdict, stage panels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from hexengine import inference, metrics

from . import clicking, compute, export_panel, ruler_panel, state

VERDICT_STYLE = {
    "regular": ("#2e9e4f", "Regular: within the noise floor, indistinguishable from a perfect projected hexagon"),
    "irregular": ("#d99a1e", "Irregular: a real hexagon, distorted beyond marking error"),
    "random-like": ("#c94040", "Random-like: no better than an arbitrary hexagon"),
}
LOCK_STYLE = {True: ("#2e9e4f", "Texture locked to the hexagon's axes (shuffle p < 0.01)"),
              False: ("#6b7280", "No texture lock (shuffle p ≥ 0.01)")}

# (Scores attribute, tile label, null-percentile key, short help)
SECOND_TILES = [
    ("homography_mean", "Homography residual", "homography_mean",
     "Same, allowing perspective (8 dof). 0 = a perspective view of a flat regular hexagon."),
    ("similarity_mean", "Similarity residual", "similarity_mean",
     "Same, allowing only rotation + scale. 0 = the hexagon is itself regular."),
]


def _badge(colour: str, title: str, text: str) -> None:
    st.markdown(f"<div style='background:{colour};color:white;padding:0.5em 0.9em;border-radius:0.5em;"
                f"font-weight:600;display:inline-block;margin:0.2em 0'>{title}"
                f"<span style='font-weight:400;font-size:0.85em'> &nbsp;{text}</span></div>", unsafe_allow_html=True)


def verdict_badges(scores, lock) -> None:
    colour, text = VERDICT_STYLE.get(scores.verdict, ("#777", scores.verdict or "n/a"))
    _badge(colour, f"Shape: {scores.verdict.upper()}", f"{text} (judged on the {scores.verdict_basis} residual)")
    colour, text = LOCK_STYLE[bool(lock.locked)]
    _badge(colour, "Texture: " + ("LOCKED" if lock.locked else "NOT LOCKED"), text)


def _pct(scores, key) -> str | None:
    pct = scores.null_percentile.get(key)
    if pct is None or not np.isfinite(pct):
        return None
    return f"{pct:.1f}% of random hexagons score better"   # metrics.null_percentile: strictly lower


def score_tiles(scores, lock) -> None:
    """Row 1: the verdict-basis residual, its noise floor, central symmetry, texture lock. Row 2: the others."""
    basis = scores.verdict_basis
    c = st.columns(4)
    c[0].metric(f"Mean vertex residual ({basis})", f"{getattr(scores, basis + '_mean'):.4f}",
                delta=_pct(scores, basis + "_mean"), delta_color="off",
                help="Mean distance (circumradius = 1) between the normalised corners and the perfect hexagon.")
    c[1].metric("Noise floor", f"{scores.noise_floor:.4f}",
                delta=f"excess {scores.excess:+.4f}", delta_color="inverse" if scores.excess > 0 else "normal",
                help=f"What a PERFECT hexagon of this size ({scores.radius_px:.0f} px) would score with corners off "
                     f"by ±{scores.corner_sigma_px:.2f} px ({scores.sigma_source}). Residual at or below it = "
                     f"indistinguishable from perfect; only the excess is real shape information.")
    c[2].metric("Central-symmetry error", f"{scores.central_symmetry_error:.4f}",
                delta=_pct(scores, "central_symmetry_error"), delta_color="off",
                help="0 = opposite sides parallel and equal (any parallel view of a cube is centrally symmetric).")
    c[3].metric("Texture lock p", f"{lock.p_value:.3f}", delta=f"order {lock.order:.2f} · offset {lock.offset_deg:.1f}°",
                delta_color="off",
                help="Rotation-shuffle p: the chance a randomly rotated cell's texture lines up with its hexagon "
                     "directions this closely. One cell rarely reaches p < 0.01; Batch review pools cells.")
    c2 = st.columns(4)
    for col, (attr, label, null_key, help_text) in zip(c2, SECOND_TILES):
        col.metric(label, f"{getattr(scores, attr):.4f}", delta=_pct(scores, null_key), delta_color="off",
                   help=help_text)
    c2[2].metric("Affine residual (raw)", f"{scores.affine_mean:.4f}", help="Always shown, whatever the warp.")
    c2[3].metric("Legacy verdict", scores.threshold_verdict or "n/a",
                 help="The original fixed-threshold verdict (Advanced), logged for comparison only.")


def face_table(faces) -> pd.DataFrame:
    return pd.DataFrame([{
        "face": r.face.name,
        "theta (°)": round(float(np.degrees(r.dominant_theta)), 1),
        "radial line (°)": round(float(np.degrees(r.face.radial_theta)), 1),
        "tangential line (°)": round(float(np.degrees(r.face.tangential_theta)), 1),
        "coherence": round(float(r.coherence), 3),
        "radial align": round(float(r.radial_alignment), 3),
        "tangential align": round(float(r.tangential_alignment), 3),
        "ternary": int(r.ternary),
    } for r in faces])


def stage_panels(a: compute.Analysis, settings: dict) -> None:
    """Expanders for each inference stage."""
    with st.expander("Stage 1 · Orientation field (structure tensor)"):
        c1, c2 = st.columns(2)
        c1.image(compute.compose(a, settings, construction=False, templates=[], orientation=True, ternary=False),
                 caption="Dominant line orientation (glyph length/opacity = coherence)", use_column_width=True)
        c2.image((np.clip(a.field.coherence, 0, 1) * 255).astype(np.uint8),
                 caption="Coherence map (white = one clear direction)", use_column_width=True, clamp=True)
    with st.expander("Stage 2–3 · Face projection and ternary assignment", expanded=True):
        c1, c2 = st.columns([1, 1])
        c1.image(compute.compose(a, settings, construction=True, templates=[], orientation=False, ternary=True),
                 caption=f"Faces for {settings['state'].name.lower()} reading", use_column_width=True)
        c2.dataframe(face_table(a.faces), hide_index=True, use_container_width=True)
        regions = ("regions centred on the reading's own cube edges (EDGE_Vk = the edge to Vk)"
                   if settings.get("convention", "edges") == "edges" else "the reading's own faces (original rule)")
        c2.caption(f"Read: {regions}. Rule: coherence < {settings['coherence_min']:.2f} → 0; within "
                   f"{settings['angle_tol']:.0f}° of the radial line → {settings['state'].sign:+d}; of the tangential "
                   f"line → {-settings['state'].sign:+d}; otherwise 0. Sign convention: +1 = structure along the "
                   f"convex cube's edges, −1 = along the concave corner's edges. Only the inner part of each region "
                   f"(inside the faint white outline) is read: {a.margin_note}.")
    with st.expander("Stage 4 · Primary line vs neutral reference"):
        p = a.primary
        if p is None:
            st.info("No face carries the primary sign, so there is no primary line for this reading.")
        else:
            c = st.columns(3)
            c[0].metric("Primary region", p.face_name)
            c[1].metric("Angle vs reference (V3→V0)", f"{p.angle_vs_reference_deg:+.1f}°")
            c[2].metric("Ternary", f"{p.ternary:+d}")
            st.caption("Positive angle = clockwise on screen from straight up.")


def render(image_id: str | None, settings: dict) -> None:
    if image_id is None:
        st.info("Upload images or click **Load sample images** in the sidebar to begin.")
        return
    cells = state.get_cells(image_id)
    pts = cells[0]

    left, right = st.columns(2)
    with left:
        st.subheader("Original + points")
        st.caption("Click the six vertices in order around the hexagon (any start, either direction).")
        clicking.render_clicker(image_id, settings, cell=0, prefix="single")
    analysis, reason = compute.try_analyse(image_id, pts, settings)
    if analysis is not None:
        with left:
            (st.success if analysis.snapped else st.warning)(analysis.snap_note)
            if analysis.snapped:
                with st.expander("Clicks (cyan) vs snapped corners (gold)"):
                    st.image(compute.clicks_and_snapped(image_id, pts, analysis), use_column_width=True)
    with right:
        st.subheader("Normalised")
        if analysis is None:
            st.info(f"Results appear once six valid points are marked ({reason}).")
        else:
            st.image(compute.overlay(image_id, pts, settings), use_column_width=True,
                     caption=f"{settings['transform']} warp · offset {settings['rotation_offset']} · "
                             f"{settings['state'].name.lower()}")

    ruler_panel.render_panel(image_id, settings, analysis)
    if analysis is None:
        return
    state.autosave(image_id, settings)

    st.divider()
    score_tiles(analysis.scores, analysis.lock)
    if settings.get("cube_layers"):
        verdict_badges(analysis.scores, analysis.lock)
    else:
        colour, text = VERDICT_STYLE.get(analysis.scores.verdict, ("#777", analysis.scores.verdict or "n/a"))
        _badge(colour, f"Shape: {analysis.scores.verdict.upper()}",
               f"{text} (judged on the {analysis.scores.verdict_basis} residual)")
    st.caption("Units: canonical (circumradius = 1). Shape verdict: residual within this cell's noise floor = "
               f"regular; above it = irregular; no better than {metrics.RANDOM_LIKE_PCT:.0f}% of 1,000 random "
               "hexagons = random-like. Texture verdict: rotation-shuffle p < 0.01. Raw residuals and the legacy "
               "verdict are kept in the record and CSV.")
    with st.expander("Per-vertex residuals"):
        st.dataframe(pd.DataFrame(analysis.scores.per_vertex, index=[f"V{i}" for i in range(6)]).round(4),
                     use_container_width=True)
    if settings.get("cube_layers"):
        stage_panels(analysis, settings)

    st.divider()
    export_panel.render(image_id, pts, analysis, settings)
