"""Hexagonal Geometric Test Engine — Streamlit entry point.

Run from the repository root:
    streamlit run app/streamlit_app.py

This file is deliberately thin: it sets up the page, warms the null baseline, draws the sidebar and dispatches to
one module per tab in app/ui/. All geometry lives in the `hexengine` package.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `hexengine` and `app.ui` importable however Streamlit was launched.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from app.ui import batch_view, cluster_view, compute, help_panel, sidebar, single_view, state  # noqa: E402


def main() -> None:
    st.set_page_config(page_title="Hex test engine", page_icon="⬡", layout="wide")
    state.init()
    if not st.session_state.get("data_dir_ok"):
        problem = state.data_dir_problem()
        if problem:
            st.error(problem)
            st.stop()
        st.session_state["data_dir_ok"] = True
    compute.warm_null_baseline()

    image_id, settings = sidebar.render()
    state.show_flash()

    tabs = st.tabs(["Single image", "Neighbour cluster", "Batch review", "Help"])
    with tabs[0]:
        single_view.render(image_id, settings)
    with tabs[1]:
        cluster_view.render_tab(image_id, settings)
    with tabs[2]:
        batch_view.render(settings)
    with tabs[3]:
        help_panel.render()


main()
