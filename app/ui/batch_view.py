"""'Batch review' tab: every upload with saved points, re-scored under the CURRENT settings, plus bulk export.

Scores are recomputed rather than read from the saved records, so changing a threshold or the Necker state updates
the whole table. The expensive steps (warp, orientation field) are cached per image_id + points + settings in
compute.py, so this stays fast after the first pass.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from hexengine import inference, session

from . import compute, export_panel, state

GALLERY_COLUMNS = 4


def _rows(settings: dict) -> list[dict]:
    """One entry per upload: {'item', 'points', 'settings', 'analysis' | None, 'reason'}.

    'settings' are the current settings with that image's own rotation offset (state.settings_for)."""
    out = []
    for item in st.session_state.get("uploads", []):
        iid = item["image_id"]
        pts = state.get_cells(iid)[0]
        s = state.settings_for(iid, settings)
        analysis, reason = compute.try_analyse(iid, pts, s) if pts else (None, "unmarked")
        out.append({"item": item, "points": pts, "settings": s, "analysis": analysis, "reason": reason})
    return out


def _table(rows: list[dict]) -> pd.DataFrame:
    recs = []
    for r in rows:
        a, item = r["analysis"], r["item"]
        rec = {"filename": item["filename"], "status": "scored" if a else r["reason"],
               "offset": int(r["settings"]["rotation_offset"])}
        if a:
            s = a.scores
            rec.update({
                "shape": s.verdict,
                "residual": round(getattr(s, s.verdict_basis + "_mean"), 4),
                "noise floor": round(s.noise_floor, 4),
                "excess": round(s.excess, 4),
                "corners": "snapped" if a.snapped else "clicked",
                "affine": round(s.affine_mean, 4),
                "homography": round(s.homography_mean, 4),
                "similarity": round(s.similarity_mean, 4),
                "central sym.": round(s.central_symmetry_error, 4),
                "null % (affine)": round(s.null_percentile.get("affine_mean", np.nan), 1),
                "lock p": round(a.lock.p_value, 3),
                "ternary": " ".join(f"{f.ternary:+d}" for f in a.faces),
                "primary (°)": round(a.primary.angle_vs_reference_deg, 1) if a.primary else None,
                "legacy verdict": s.threshold_verdict,
            })
        scale = state.scale_for(item["image_id"])
        if a and scale is not None:
            from hexengine import morphometry
            m = morphometry.measure_cell(a.points_used, scale, a.wall_width_px)
            rec.update({"px/mm": round(scale.px_per_mm, 3), "width (mm)": round(m.width_mm, 3),
                        "anisotropy (mm)": round(m.anisotropy_mm, 3), "size class": m.size_class})
        cl = st.session_state["cluster"].get(item["image_id"])
        rec["continuity"] = round(cl["continuity_mean"], 3) if cl and cl.get("continuity_mean") is not None else None
        recs.append(rec)
    return pd.DataFrame(recs)


def _pooled_lock(analyses: list) -> None:
    """Rotation-shuffle test over every scored image: does texture line up with each cell's own axes?"""
    res = inference.pooled_lock_test([a.lock for a in analyses])
    if res is None:
        st.caption("Texture-lock test (pooled): needs at least two scored images with six-fold texture "
                   f"(order ≥ {inference.LOCK_MIN_ORDER:.2f}).")
        return
    word = "LOCKED" if res["locked"] else "not locked"
    st.markdown(f"**Texture lock across {res['n']} cells: {word}** · L = {res['L']:+.3f}, shuffle p = {res['p']:.4f} "
                f"(locked if p < {inference.LOCK_P}). L is +1 when every cell's texture sits on its own hexagon "
                "directions and about 0 when cells are oriented at random.")


def _export_all(rows: list[dict], settings: dict) -> None:
    """Sheets for every scored image, CSV of all records (re-scored), ZIP of everything."""
    paths, records = [], []
    progress = st.progress(0.0, text="Rendering comparison sheets...")
    scored = [r for r in rows if r["analysis"] is not None]
    for k, r in enumerate(scored):
        iid = r["item"]["image_id"]
        path, _ = export_panel.sheet_for(iid, r["points"], r["analysis"], r["settings"])
        paths.append(path)
        records.append(state.build_record(iid, r["settings"]))
        progress.progress((k + 1) / max(1, len(scored)), text=f"Rendered {k + 1}/{len(scored)}")
    progress.empty()
    csv_path = session.export_csv(records)
    points_dir = session.data_dir() / "points"
    zip_path = session.export_zip(paths + [csv_path] + ([points_dir] if points_dir.is_dir() else []))
    st.session_state["batch_export"] = {"csv": str(csv_path), "zip": str(zip_path), "n": len(paths)}


def render(settings: dict) -> None:
    uploads = st.session_state.get("uploads", [])
    if not uploads:
        st.info("No images yet. Upload some or load the samples from the sidebar.")
        return
    rows = _rows(settings)
    n_scored = sum(r["analysis"] is not None for r in rows)
    st.caption(f"{len(rows)} images, {n_scored} scored with the current settings "
               f"({settings['transform']}, {settings['state'].name.lower()}). Click a column header to sort.")
    st.dataframe(_table(rows), hide_index=True, use_container_width=True)
    _pooled_lock([r["analysis"] for r in rows if r["analysis"] is not None])

    c1, c2, c3 = st.columns(3)
    if c1.button("Export all", type="primary", disabled=n_scored == 0, key="export_all"):
        _export_all(rows, settings)
    done = st.session_state.get("batch_export")
    if done and Path(done["zip"]).exists():
        c2.download_button("Download ZIP", Path(done["zip"]).read_bytes(), file_name=Path(done["zip"]).name,
                           mime="application/zip", key="dl_zip")
        c3.download_button("Download CSV", Path(done["csv"]).read_bytes(), file_name=Path(done["csv"]).name,
                           mime="text/csv", key="dl_csv")
        st.caption(f"Exported {done['n']} sheets to {Path(done['zip']).parent}")

    st.subheader("Gallery")
    scored = [r for r in rows if r["analysis"] is not None]
    if not scored:
        st.info("Mark six points on an image (Single image tab) to see it here.")
        return
    for start in range(0, len(scored), GALLERY_COLUMNS):
        cols = st.columns(GALLERY_COLUMNS)
        for col, r in zip(cols, scored[start:start + GALLERY_COLUMNS]):
            a = r["analysis"]
            col.image(compute.overlay(r["item"]["image_id"], r["points"], r["settings"]), use_column_width=True,
                      caption=f"{r['item']['filename']} · {a.scores.verdict} · residual "
                              f"{getattr(a.scores, a.scores.verdict_basis + '_mean'):.4f} "
                              f"(floor {a.scores.noise_floor:.4f})")
