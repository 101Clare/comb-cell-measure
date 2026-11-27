"""Headless tests of the beekeeper app (app/bee_app.py)."""

from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")

APP = str(Path(__file__).resolve().parents[1] / "app" / "bee_app.py")


def test_bee_app_starts_and_asks_for_a_photo():
    at = st_testing.AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("Upload a photo" in i.value for i in at.info)


def test_bee_app_measures_after_four_clicks():
    from test_comb import synthetic_comb

    at = st_testing.AppTest.from_file(APP, default_timeout=300)
    at.session_state["photo_key"] = "synthetic"
    at.session_state["img"] = synthetic_comb(5.2)
    at.session_state["clicks"] = [(100.0, 20.0), (600.0, 20.0), (80.0, 60.0), (620.0, 460.0)]
    at.session_state["last"] = None
    at.session_state["report"] = None
    at.session_state["nonce"] = 0
    # The uploader can't be driven headlessly; stub it so the app sees an upload with the same key.
    import streamlit as st

    class Fake:
        name, size = "synthetic", 0

    orig = st.file_uploader
    st.file_uploader = lambda *a, **k: Fake()
    try:
        at.session_state["photo_key"] = "synthetic-0"
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        at.number_input[0].set_value(50.0)
        next(b for b in at.button if "Measure" in b.label).click().run()
        assert not at.exception, [e.value for e in at.exception]
        rep = at.session_state["report"]
        assert rep is not None and abs(rep.median_width_mm - 5.2) < 0.05
        assert any(m.label == "Median cell width" for m in at.metric)
    finally:
        st.file_uploader = orig
