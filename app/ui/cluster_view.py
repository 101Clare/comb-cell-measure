"""'Neighbour cluster' tab: mark a centre cell and up to three neighbours, then measure continuity across shared edges.

Flow (all maths in hexengine.neighbours / inference):
  complete cells --snap_points--> fit each --> analyse_faces on each warped cell --> find_shared_edges
  --> edge_continuity --> render.draw_cluster + per-edge table + cluster_summary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from hexengine import inference, neighbours, render

from . import clicking, compute, state

# Same colours draw_cluster uses for fits 0..3 (render's golden-ratio hue walk), so markers and outlines match.
CELL_COLOURS = [render._distinct_colour(i) for i in range(state.N_CELLS)]


def _hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(c) for c in rgb)


def run_cluster(image_id: str, cells: list, settings: dict, tol: float):
    """Returns (cell indices used, analyses, continuity) or raises ValueError."""
    used = [i for i, c in enumerate(cells) if len(c) == 6]
    snapped = neighbours.snap_points([np.asarray(cells[i], float) for i in used], tol_px=tol)
    analyses = [compute.analyse(image_id, pts, settings) for pts in snapped]
    fits = [a.fit for a in analyses]
    faces = [a.faces for a in analyses]
    edges = neighbours.find_shared_edges(fits, tol_px=tol)
    cont = neighbours.edge_continuity(fits, faces, settings["state"], edges, kind=settings["transform"])
    return used, analyses, cont


def render_tab(image_id: str | None, settings: dict) -> None:
    if image_id is None:
        st.info("Upload images or load the samples first.")
        return
    cells = state.get_cells(image_id)

    top = st.columns([3, 2])
    labels = [f"{lab} ({len(cells[i])}/6)" for i, lab in enumerate(state.CELL_LABELS)]
    active = top[0].radio("Cell to click", range(state.N_CELLS), format_func=lambda i: labels[i], horizontal=True,
                          key=f"cluster_cell_{image_id}")
    tol = top[1].slider("Snap tolerance (px)", 1.0, 60.0, 12.0, 1.0, key="snap_tol",
                        help="Clicked vertices of different cells closer than this are merged, so shared edges "
                             "coincide exactly.")
    st.markdown(" ".join(f"<span style='color:{_hex(CELL_COLOURS[i])};font-weight:700'>■ {lab}</span>"
                         for i, lab in enumerate(state.CELL_LABELS)), unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        clicking.render_clicker(image_id, settings, cell=active, prefix="cluster", colours=CELL_COLOURS)

    complete = [i for i, c in enumerate(cells) if len(c) == 6]
    with right:
        if len(complete) < 2:
            # Drop any summary left from an earlier state (e.g. after Undo), so it is not saved or exported.
            st.session_state["cluster"].pop(image_id, None)
            st.info("Mark at least two complete cells (the centre and a neighbour) to measure continuity.")
            return
        try:
            used, analyses, cont = run_cluster(image_id, cells, settings, tol)
            fits, faces = [a.fit for a in analyses], [a.faces for a in analyses]
        except ValueError as exc:
            st.session_state["cluster"].pop(image_id, None)
            st.error(f"Could not fit a cell: {exc}")
            return
        st.image(render.draw_cluster(compute.image(image_id), fits, cont, faces), use_column_width=True,
                 caption="Shared edges: green ≥ 0.9, amber ≥ 0.6, red below. Glyphs show each side's face orientation.")
        if used != list(range(len(used))):
            st.caption("Numbers on this image count complete cells in order: "
                       + ", ".join(f"{k + 1} = {state.CELL_LABELS[i]}" for k, i in enumerate(used)))

    summary = neighbours.cluster_summary(cont)
    lock = inference.pooled_lock_test([a.lock for a in analyses])
    if lock is not None:
        st.caption(f"Texture lock across these {lock['n']} cells: L = {lock['L']:+.3f}, shuffle p = {lock['p']:.4f} "
                   f"({'LOCKED' if lock['locked'] else 'not locked'}; locked if p < {inference.LOCK_P}).")
    m = st.columns(5)
    m[0].metric("Mean continuity", f"{summary['mean_score']:.3f}", help="Mean |cos Δ| over shared edges (1 = perfect).")
    m[1].metric("Shared edges", summary["n_edges"])
    m[2].metric("Same", summary["n_same"])
    m[3].metric("Flipped", summary["n_flipped"])
    m[4].metric("Unrelated", summary["n_unrelated"])
    if not cont:
        st.warning("No shared edges found: neighbouring cells need two vertices within the snap tolerance.")

    rows = [{
        "cells": f"{state.CELL_LABELS[used[c.edge.cell_a]]} ↔ {state.CELL_LABELS[used[c.edge.cell_b]]}",
        "edge a": f"V{c.edge.edge_a[0]}–V{c.edge.edge_a[1]}",
        "edge b": f"V{c.edge.edge_b[0]}–V{c.edge.edge_b[1]}",
        "Δ (°)": round(c.delta_deg, 1),
        "score": round(c.score, 3),
        "ternary a": c.ternary_a,
        "ternary b": c.ternary_b,
        "relation": c.relation,
    } for c in cont]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # Keep the summary with the record (CSV reads record["cluster"]["continuity_mean"]).
    cluster = {"continuity_mean": summary["mean_score"] if cont else None, **summary,
               "snap_tol_px": tol, "cells_used": used,
               "edges": [{k: v for k, v in r.items()} for r in rows]}
    st.session_state["cluster"][image_id] = cluster
    state.autosave(image_id, settings)
