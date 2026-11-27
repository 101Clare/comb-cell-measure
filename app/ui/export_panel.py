"""Single-image exports: comparison sheet PNG (written to data/exports/) and the JSON record, as downloads."""

from __future__ import annotations

import io
import json
from pathlib import Path

import streamlit as st
from PIL import Image

from hexengine import render as hexrender  # aliased: this module's own entry point is called render()
from hexengine import session

from . import compute, state


def png_bytes(image) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return buf.getvalue()


def sheet_for(image_id: str, points, analysis: compute.Analysis, settings: dict) -> tuple[Path, bytes]:
    """Build, save and return (path, png bytes) of the comparison sheet for one image."""
    name = Path(state.filename_of(image_id)).stem
    title = (f"{name}  |  {settings['transform']}  |  {settings['state'].name.lower()}  |  offset "
             f"{settings['rotation_offset']}  |  verdict {analysis.scores.verdict}")
    sheet = hexrender.comparison_sheet(compute.comparison_panels(image_id, points, analysis, settings), title=title)
    path = session.save_png(sheet, f"{name}_{image_id}_sheet")
    return path, path.read_bytes()


def render(image_id: str, points, analysis: compute.Analysis, settings: dict) -> None:
    st.subheader("Export")
    c1, c2 = st.columns(2)
    key = f"sheet_{image_id}"
    # Everything the sheet depends on: points, offset, warp, overlays and verdict thresholds.
    fingerprint = (compute.points_key(points), int(settings["rotation_offset"]), settings["transform"],
                   compute.overlay_key(settings), tuple(sorted(settings["thresholds"].items())))
    with c1:
        if st.button("Build comparison sheet", key=f"build_{key}"):
            with st.spinner("Rendering sheet..."):
                path, data = sheet_for(image_id, points, analysis, settings)
            st.session_state[key] = (str(path), data, fingerprint)
        built = st.session_state.get(key)
        if built:
            path, data, okey = built
            if okey != fingerprint:
                st.caption("Settings changed since this sheet was built; rebuild to update it.")
            st.image(data, caption=f"Saved to {path}", use_column_width=True)
            st.download_button("Download sheet (PNG)", data, file_name=Path(path).name, mime="image/png",
                               key=f"dl_{key}")
    with c2:
        record = state.build_record(image_id, settings)
        st.download_button("Download record (JSON)", json.dumps(record, indent=2, default=str),
                           file_name=f"{Path(record['filename']).stem}_{image_id}.json", mime="application/json",
                           key=f"dl_json_{image_id}")
        st.caption("Points (full-resolution px), settings and scores. Also saved automatically under data/points/.")
