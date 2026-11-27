"""Headless tests of the Streamlit app (streamlit.testing.v1.AppTest).

The click widget itself (a custom component) cannot be clicked headlessly, so the full pipeline is exercised through
the "Load sample images" + "Use ground-truth points" path, which fills the same session state that clicks would.
"""

from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = st_testing.AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
TIMEOUT = 120


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HEX_DATA_DIR", str(tmp_path))
    return tmp_path


def _no_exceptions(at):
    assert not at.exception, [e.value for e in at.exception]


def test_app_loads_with_no_uploads(data_dir):
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    _no_exceptions(at)
    assert [t.label for t in at.tabs] == ["Single image", "Neighbour cluster", "Batch review", "Help"]
    assert any("Load sample images" in b.label for b in at.button)
    # The help tab renders its markdown.
    assert any("The pipeline" in m.value for m in at.markdown)


def test_samples_with_ground_truth_full_flow(data_dir):
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    at.button(key="load_samples").click().run()
    _no_exceptions(at)

    # Every sample is stored, and records with points + scores were saved.
    records = list((data_dir / "points").glob("*.json"))
    assert len(records) >= 8
    labels = at.radio(key="image_id").options
    assert all("scored" in lab for lab in labels), labels

    # Pick the honeycomb (has four ground-truth cells).
    at.radio(key="image_id").set_value(_id_for(at, "honeycomb")).run()
    _no_exceptions(at)

    metric_labels = [m.label for m in at.metric]
    assert "Mean vertex residual (affine)" in metric_labels
    assert "Mean continuity" in metric_labels
    affine = next(m for m in at.metric if m.label == "Mean vertex residual (affine)")
    assert float(affine.value) < 0.01                       # regular cells
    cont = next(m for m in at.metric if m.label == "Shared edges")
    assert int(cont.value) == 3                             # centre shares an edge with each neighbour


def _id_for(at, name: str) -> str:
    """image_id whose radio label contains `name` (the radio stores ids, labels come from format_func)."""
    from hexengine import session

    for item in session.list_uploads():
        if name in item["filename"]:
            return item["image_id"]
    raise AssertionError(name)


def test_settings_toggles_rerender(data_dir):
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    at.button(key="load_samples").click().run()
    at.radio(key="image_id").set_value(_id_for(at, "shaded_cube")).run()
    _no_exceptions(at)
    at.checkbox(key="cube_layers").check().run()         # the cube layers are hidden by default
    at.radio(key="necker").set_value("Concave / vertex-in (−1)").run()
    _no_exceptions(at)
    at.radio(key="transform").set_value("homography").run()
    for key in ("ov_orientation", "ov_ternary", "ov_tpl_1", "ov_tpl_2", "ov_tpl_3"):
        at.checkbox(key=key).check()
    at.run()
    _no_exceptions(at)
    at.button(key="rotate60").click().run()
    _no_exceptions(at)
    assert at.slider(key="rotation_offset").value == 1


def test_single_export_and_batch_export(data_dir):
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    at.button(key="load_samples").click().run()
    iid = _id_for(at, "regular_hexagon")
    at.radio(key="image_id").set_value(iid).run()
    at.button(key=f"build_sheet_{iid}").click().run()
    _no_exceptions(at)
    assert list((data_dir / "exports").glob("*_sheet.png"))

    at.button(key="export_all").click().run()
    _no_exceptions(at)
    assert list((data_dir / "exports").glob("hex_scores_*.csv"))
    assert list((data_dir / "exports").glob("hex_export_*.zip"))


def test_rotation_offset_is_per_image_and_survives_reload(data_dir):
    from hexengine import session

    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    at.button(key="load_samples").click().run()
    cube, regular = _id_for(at, "shaded_cube"), _id_for(at, "regular_hexagon")
    at.radio(key="image_id").set_value(cube).run()
    before = session.load_record(cube)["scores"]
    at.button(key="rotate60").click().run()
    _no_exceptions(at)
    assert at.slider(key="rotation_offset").value == 1
    at.radio(key="image_id").set_value(regular).run()
    assert at.slider(key="rotation_offset").value == 0          # the other image keeps its own offset
    at.radio(key="image_id").set_value(cube).run()
    assert at.slider(key="rotation_offset").value == 1
    rec = session.load_record(cube)
    assert rec["rotation_offset"] == 1 and rec["settings"]["rotation_offset"] == 1
    assert session.load_record(regular)["rotation_offset"] == 0
    # Residual scores do not depend on which vertex is V0, so the saved scores are unchanged (up to float
    # round-off: snapped corners leave a small non-zero residual whose last digits depend on the labelling).
    for key in ("affine_mean", "homography_mean", "similarity_mean", "central_symmetry_error"):
        assert rec["scores"][key] == pytest.approx(before[key], abs=1e-7)

    # A fresh session restores the saved offset and reproduces the saved scores.
    at2 = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    at2.radio(key="image_id").set_value(cube).run()
    _no_exceptions(at2)
    assert at2.slider(key="rotation_offset").value == 1
    affine = next(m for m in at2.metric if m.label == "Mean vertex residual (affine)")
    assert affine.value == f"{rec['scores']['affine_mean']:.4f}"
    assert session.load_record(cube)["faces"] == rec["faces"]


def test_cluster_summary_dropped_when_cells_are_undone(data_dir):
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    at.button(key="load_samples").click().run()
    iid = _id_for(at, "honeycomb")
    at.radio(key="image_id").set_value(iid).run()
    assert iid in at.session_state["cluster"]
    for cell in (1, 2, 3):
        at.radio(key=f"cluster_cell_{iid}").set_value(cell).run()
        at.button(key=f"cluster_{iid}_{cell}_undo").click().run()
        _no_exceptions(at)
    # Only the centre cell is complete now: no stale continuity summary may remain (it feeds records and CSV).
    assert iid not in at.session_state["cluster"]
    assert "Mean continuity" not in [m.label for m in at.metric]


def test_unwritable_data_dir_gives_a_clear_message(tmp_path, monkeypatch):
    blocker = tmp_path / "not_a_folder"
    blocker.write_text("x")
    monkeypatch.setenv("HEX_DATA_DIR", str(blocker))
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    _no_exceptions(at)
    assert any("Cannot write to the data folder" in e.value for e in at.error)


def test_clear_and_undo_buttons(data_dir):
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    at.button(key="load_samples").click().run()
    iid = _id_for(at, "regular_hexagon")
    at.radio(key="image_id").set_value(iid).run()
    at.button(key=f"single_{iid}_0_undo").click().run()
    _no_exceptions(at)
    assert any("5/6 points" in c.value for c in at.caption)
    at.button(key=f"single_{iid}_0_clear").click().run()
    _no_exceptions(at)
    assert any("0/6 points" in c.value for c in at.caption)
    # Auto-detect suggests six points again on the clean synthetic sample.
    at.button(key=f"single_{iid}_0_detect").click().run()
    _no_exceptions(at)
    assert any("6/6 points" in c.value for c in at.caption)


def test_snapping_noise_floor_and_calibrated_verdicts(data_dir):
    """Patch items 3 and 4: corners snap to the walls, the noise floor sits next to the residual and the verdict is
    noise-floor based, with the raw numbers and the legacy verdict still recorded."""
    from hexengine import session
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    at.button(key="load_samples").click().run()
    cube = _id_for(at, "shaded_cube")
    at.radio(key="image_id").set_value(cube).run()
    _no_exceptions(at)
    labels = [m.label for m in at.metric]
    assert "Noise floor" in labels and "Central-symmetry error" in labels and "Texture lock p" in labels
    assert any("Snapped to the walls" in s.value for s in at.success)
    rec = session.load_record(cube)
    sc = rec["scores"]
    assert sc["verdict"] in ("regular", "irregular", "random-like")
    assert sc["threshold_verdict"] in ("strong", "weak", "noise")          # legacy verdict kept
    assert sc["noise_floor"] > 0 and sc["sigma_source"] == "refined" and rec["snapped"] is True
    assert "p" in rec["texture_lock"]

    # Edge convention on the shaded cube sample: its line texture runs along each convex face's long diagonal,
    # i.e. along the CONCAVE corner's edges (V0, V2, V4), so it reads 0s under Convex and -1s under Concave.
    assert [f["ternary"] for f in rec["faces"]] == [0, 0, 0]
    assert all(f["face"].startswith("EDGE_V") for f in rec["faces"])
    at.checkbox(key="cube_layers").check().run()
    at.radio(key="necker").set_value("Concave / vertex-in (−1)").run()
    _no_exceptions(at)
    rec = session.load_record(cube)
    assert [f["ternary"] for f in rec["faces"]] == [-1, -1, -1]

    # Turning snapping off falls back to the clicks with an assumed click error.
    at.checkbox(key="snap").uncheck().run()
    _no_exceptions(at)
    assert any("Snapping is off" in w.value for w in at.warning)


def test_ruler_calibration_gives_mm_width(data_dir):
    """Ruler calibration: two stored ruler points + a distance give px/mm and the marked cell's width in mm."""
    from hexengine import session
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    at.button(key="load_samples").click().run()
    reg = _id_for(at, "regular_hexagon")
    at.radio(key="image_id").set_value(reg).run()
    assert not any(c.value for c in at.checkbox if c.key == "cube_layers")    # retired layers hidden by default
    assert not [r for r in at.radio if r.key == "necker"]
    at.session_state["rulers"] = {reg: {"points": [[100.0, 650.0], [300.0, 650.0]], "mm": 20.0}}   # 10 px/mm
    at.run()
    _no_exceptions(at)
    labels = {m.label: m.value for m in at.metric}
    assert labels["Scale"].startswith("10.000 px/mm")
    assert "Flat-to-flat width" in labels and labels["Flat-to-flat width"].endswith("mm")
    rec = session.load_record(reg)
    assert rec["ruler"]["mm"] == 20.0 and abs(rec["morphometry"]["px_per_mm"] - 10.0) < 1e-9
