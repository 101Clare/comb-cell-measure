"""Session-state helpers and record persistence for the UI.

Points being edited live in st.session_state["cells"][image_id]: a list of N_CELLS lists of [x, y] in FULL-resolution
image pixels (cell 0 = centre; single-image mode uses cell 0). They are loaded from data/points/<id>.json the first
time an image is opened, and written back with `save()` (explicit Save button, and automatically whenever a complete
centre cell is scored under new points/settings).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import streamlit as st

from hexengine import session

from . import compute

N_CELLS = 4
CELL_LABELS = ["Cell 1 (centre)", "Cell 2", "Cell 3", "Cell 4"]   # matches render.draw_cluster's 1-based numbers
REPO_ROOT = Path(__file__).resolve().parents[2]


def samples_dir() -> Path:
    """Folder with the bundled synthetic samples ($HEX_SAMPLES_DIR overrides)."""
    return Path(os.environ.get("HEX_SAMPLES_DIR") or REPO_ROOT / "samples")


def init() -> None:
    """Create the session-state containers this app uses (idempotent)."""
    ss = st.session_state
    ss.setdefault("cells", {})          # image_id -> [[ [x, y], ... ] * N_CELLS]
    ss.setdefault("last_click", {})     # widget key -> last (x, y, w, h) seen, for de-duplication
    ss.setdefault("nonce", {})          # widget key -> int, bumped to reset a click widget
    ss.setdefault("cluster", {})        # image_id -> cluster summary dict (for the saved record / CSV)
    ss.setdefault("saved_fp", {})       # image_id -> fingerprint of the last auto-saved record
    ss.setdefault("offsets", {})        # image_id -> rotation offset (it labels THAT image's vertices)
    ss.setdefault("rulers", {})         # image_id -> {"points": [[x, y], ...] (<= 2), "mm": float}
    ss.setdefault("seen_uploads", set())
    ss.setdefault("flash", [])          # one-shot messages shown on the next run


def data_dir_problem() -> str | None:
    """None if the data folder can be created and written to, else a plain-language explanation.

    Checked once per session at start-up, so a read-only bind mount gives a clear message instead of a traceback
    the first time something is saved. Only the root is created here; sub-folders are still created lazily.
    """
    root = session.data_dir()
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return (f"Cannot write to the data folder `{root}` ({exc.__class__.__name__}: {exc}). Uploads, points and "
                "exports are stored there. Make sure the folder exists and is writable. With Docker, that is the "
                "`data` folder next to docker-compose.yml (on a Linux host: `sudo chown -R 1000:1000 data`); "
                "running locally, set HEX_DATA_DIR to a writable folder.")
    return None


def flash(msg: str, kind: str = "info") -> None:
    """Queue a message to show at the top of the next run (callbacks cannot draw directly)."""
    st.session_state.setdefault("flash", []).append((kind, msg))


def show_flash() -> None:
    for kind, msg in st.session_state.get("flash", []):
        getattr(st, kind, st.info)(msg)
    st.session_state["flash"] = []


# ---------------------------------------------------------------------------------------------------------------------
# Cells (points being edited)
# ---------------------------------------------------------------------------------------------------------------------

def _pad(cells) -> list[list[list[float]]]:
    cells = [[[float(x), float(y)] for x, y in (c or [])] for c in (cells or [])][:N_CELLS]
    return cells + [[] for _ in range(N_CELLS - len(cells))]


def get_cells(image_id: str) -> list[list[list[float]]]:
    """Editable cells for an image (loaded from its saved record the first time)."""
    store = st.session_state["cells"]
    if image_id not in store:
        rec = session.load_record(image_id) or {}
        store[image_id] = _pad(rec.get("cells"))
        if rec.get("cluster"):
            st.session_state["cluster"][image_id] = rec["cluster"]
    return store[image_id]


def set_cell(image_id: str, i: int, points) -> None:
    get_cells(image_id)[i] = [[float(x), float(y)] for x, y in np.asarray(points, float).reshape(-1, 2)]


def bump_nonce(widget_key: str) -> None:
    """Give a click widget a fresh identity (its stored value resets to None)."""
    st.session_state["nonce"][widget_key] = st.session_state["nonce"].get(widget_key, 0) + 1
    st.session_state["last_click"].pop(widget_key, None)


# ---------------------------------------------------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------------------------------------------------

def ruler_for(image_id: str) -> dict:
    """This image's ruler calibration ({"points": [...], "mm": float}), loaded from its saved record the first time."""
    rulers = st.session_state.setdefault("rulers", {})
    if image_id not in rulers:
        rec = session.load_record(image_id) or {}
        saved = rec.get("ruler") or {}
        rulers[image_id] = {"points": [list(p) for p in saved.get("points", [])][:2], "mm": float(saved.get("mm", 10.0))}
    return rulers[image_id]


def scale_for(image_id: str):
    """morphometry.Scale for the image, or None until two ruler points and a positive distance are set."""
    from hexengine import morphometry
    r = ruler_for(image_id)
    if len(r["points"]) == 2 and r["mm"] > 0 and r["points"][0] != r["points"][1]:
        return morphometry.Scale(tuple(r["points"][0]), tuple(r["points"][1]), float(r["mm"]))
    return None


def offset_for(image_id: str) -> int:
    """This image's rotation offset: the value chosen this session, else the saved record's, else 0."""
    offsets = st.session_state.setdefault("offsets", {})
    if image_id not in offsets:
        rec = session.load_record(image_id) or {}
        try:
            offsets[image_id] = int(rec.get("rotation_offset") or 0) % 6
        except (TypeError, ValueError):
            offsets[image_id] = 0
    return offsets[image_id]


def settings_for(image_id: str, settings: dict) -> dict:
    """The settings with this image's own rotation offset (the offset says which of ITS points is V0)."""
    return {**settings, "rotation_offset": offset_for(image_id)}


def settings_for_record(settings: dict) -> dict:
    """JSON-friendly copy of the settings dict."""
    out = dict(settings)
    out["state"] = settings["state"].name
    out["templates"] = list(settings["templates"])
    return out


def filename_of(image_id: str) -> str:
    for item in st.session_state.get("uploads", []):
        if item["image_id"] == image_id:
            return str(item["filename"])
    rec = session.load_record(image_id) or {}
    return rec.get("filename") or image_id


def build_record(image_id: str, settings: dict) -> dict:
    """Record for data/points/<id>.json: all cells, settings, centre-cell scores and the cluster summary."""
    settings = settings_for(image_id, settings)
    cells = get_cells(image_id)
    analysis, _ = compute.try_analyse(image_id, cells[0], settings)
    rec = {
        "image_id": image_id,
        "filename": filename_of(image_id),
        "cells": [c for c in cells],
        "rotation_offset": int(settings["rotation_offset"]),
        "state": settings["state"].name,
        "transform": settings["transform"],
        "settings": settings_for_record(settings),
        "scores": session.scores_to_dict(analysis.scores) if analysis else None,
    }
    if analysis is not None:
        rec["corners_used"] = analysis.points_used.tolist()
        rec["snapped"] = bool(analysis.snapped)
        rec["snap_note"] = analysis.snap_note
        rec["edge_margin_frac"] = float(analysis.margin_frac)
        rec["texture_lock"] = {"order": analysis.lock.order, "phase_deg": analysis.lock.phase_deg,
                               "offset_deg": analysis.lock.offset_deg, "p": analysis.lock.p_value,
                               "locked": analysis.lock.locked}
        rec["faces"] = [{"face": r.face.name, "theta_deg": float(np.degrees(r.dominant_theta)),
                         "coherence": float(r.coherence), "ternary": int(r.ternary)} for r in analysis.faces]
        if analysis.primary is not None:
            rec["primary_line"] = {"face": analysis.primary.face_name,
                                   "angle_vs_reference_deg": analysis.primary.angle_vs_reference_deg}
    cluster = st.session_state["cluster"].get(image_id)
    if cluster:
        rec["cluster"] = cluster
    ruler = ruler_for(image_id)
    if ruler["points"]:
        rec["ruler"] = ruler
    scale = scale_for(image_id)
    if scale is not None and analysis is not None:
        from hexengine import morphometry
        m = morphometry.measure_cell(analysis.points_used, scale, analysis.wall_width_px)
        rec["morphometry"] = {"px_per_mm": scale.px_per_mm, "width_mm": m.width_mm, "widths_mm": m.widths_mm,
                              "anisotropy_mm": m.anisotropy_mm, "inner_width_mm": m.inner_mm,
                              "size_class": m.size_class, "width_basis": "snapped" if analysis.snapped else "clicked"}
    return rec


def _fingerprint(rec: dict) -> str:
    return json.dumps({k: v for k, v in rec.items() if k != "updated"}, sort_keys=True, default=str)


def save(image_id: str, settings: dict, quiet: bool = False) -> dict:
    """Persist the current record for an image (and remember it, so autosave does not rewrite it)."""
    rec = build_record(image_id, settings)
    session.save_record(rec)
    st.session_state["saved_fp"][image_id] = _fingerprint(rec)
    if not quiet:
        flash(f"Saved points for {rec['filename']}.", "success")
    return rec


def autosave(image_id: str, settings: dict) -> None:
    """Save when the record content changed since the last save (called after a complete cell is scored)."""
    rec = build_record(image_id, settings)
    if st.session_state["saved_fp"].get(image_id) != _fingerprint(rec):
        session.save_record(rec)
        st.session_state["saved_fp"][image_id] = _fingerprint(rec)


# ---------------------------------------------------------------------------------------------------------------------
# Samples and ground truth
# ---------------------------------------------------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def ground_truth() -> dict:
    """samples/ground_truth.json ({} if absent)."""
    path = samples_dir() / "ground_truth.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def ground_truth_cells(filename: str) -> list | None:
    """Ground-truth cells for a sample file name: all listed cells (centre first) or just its vertices."""
    entry = ground_truth().get(Path(str(filename)).name)
    if not entry:
        return None
    if entry.get("cells"):
        return [c["vertices"] for c in entry["cells"]][:N_CELLS]
    return [entry["vertices"]]


def apply_ground_truth(image_id: str, settings: dict) -> bool:
    """Replace an image's cells with its ground truth and save. False if it has none."""
    gt = ground_truth_cells(filename_of(image_id))
    if not gt:
        return False
    st.session_state["cells"][image_id] = _pad(gt)
    st.session_state["cluster"].pop(image_id, None)
    save(image_id, settings, quiet=True)
    return True


def load_samples(settings: dict, use_ground_truth: bool) -> list[str]:
    """Copy samples/*.png into the uploads store (optionally with ground-truth points). Returns image ids."""
    ids = []
    for path in sorted(samples_dir().glob("*.png")):
        iid = session.save_upload(path.read_bytes(), path.name)
        ids.append(iid)
        if use_ground_truth:
            st.session_state.setdefault("uploads", []).append({"image_id": iid, "filename": path.name})
            apply_ground_truth(iid, settings)
    return ids
