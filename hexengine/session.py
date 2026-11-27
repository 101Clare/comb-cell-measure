"""Persisting clicked points / settings / scores, and exports (PNG sheets, CSV, ZIP).

OWNER: Agent F (render & export). Data root defaults to env HEX_DATA_DIR or ./data.
Layout:
  data/uploads/<image_id>.<ext>      original bytes
  data/uploads/_index.json           {image_id: original filename} (so un-marked uploads keep their names)
  data/points/<image_id>.json        {"image_id", "filename", "cells": [[[x,y]*6], ...], "rotation_offset",
                                       "state", "transform", "settings": {...}, "scores": {...} | null, "updated"}
  data/exports/                      sheets, CSV, ZIP
image_id = first 12 hex chars of sha1(file bytes).

Notes
-----
* The data root is resolved on every call (relative paths are relative to the current working directory), so tests
  and the app can redirect it simply by setting HEX_DATA_DIR. Sub-folders are created lazily, the first time
  something is written there.
* JSON writes go to a temporary file first and are then renamed, so a crash never leaves a half-written record.
* Records may contain numpy arrays / numpy scalars / enums / Scores dataclasses; they are converted to plain JSON
  types on save. Loading returns plain Python (lists, floats, str) — convert back with np.asarray where needed.
"""

from __future__ import annotations

import csv
import dataclasses
import enum
import hashlib
import json
import os
import re
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

from .types import Scores

_INDEX_NAME = "_index.json"
_ID_RE = re.compile(r"^[0-9a-f]{12}$")
_SCALAR_SCORE_FIELDS = ("affine_mean", "affine_max", "homography_mean", "homography_max",
                        "similarity_mean", "similarity_max", "central_symmetry_error")
# Calibrated-verdict inputs, logged next to the raw residuals so the absolute numbers stay inspectable.
_CALIBRATION_FIELDS = ("verdict_basis", "noise_floor", "excess", "radius_px", "corner_sigma_px", "sigma_source",
                       "threshold_verdict")


# ---------------------------------------------------------------------------------------------------------------------
# Paths and small helpers
# ---------------------------------------------------------------------------------------------------------------------

def data_dir() -> Path:
    """Root data folder: $HEX_DATA_DIR if set, else ./data (relative to the current working directory)."""
    root = os.environ.get("HEX_DATA_DIR") or "data"
    return Path(root).expanduser().resolve()


def _subdir(name: str) -> Path:
    """data/<name>, created on first use."""
    p = data_dir() / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_name(name: str, default: str = "file") -> str:
    """Sanitise a user-supplied name into a safe single path component (no folders, no odd characters)."""
    base = Path(str(name).replace("\\", "/")).name          # drop any directory part, incl. Windows-style
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    return base[:120] or default


def _check_id(image_id: str) -> str:
    """Reject anything that is not a 12-hex-char image id (guards against path tricks like '../')."""
    if not isinstance(image_id, str) or not _ID_RE.match(image_id):
        raise ValueError(f"invalid image_id: {image_id!r}")
    return image_id


def _timestamp() -> str:
    """Compact local timestamp for export file names, e.g. 20261124_153012."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _to_jsonable(obj):
    """Recursively convert numpy arrays/scalars, enums, dataclasses and paths into plain JSON types."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _to_jsonable(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(_to_jsonable(k)) if not isinstance(k, str) else k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, enum.Enum):
        return obj.name if not isinstance(obj, str) else obj.value
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None  # JSON has no NaN / inf
    return obj


def _write_json(path: Path, payload) -> None:
    """Atomic JSON write (temp file + rename)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path):
    """Read JSON, returning None if the file is missing or unreadable."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _upload_index() -> dict:
    p = data_dir() / "uploads" / _INDEX_NAME
    idx = _read_json(p) if p.exists() else None
    return idx if isinstance(idx, dict) else {}


def _upload_path(image_id: str) -> Path | None:
    """The stored upload file for an id, or None."""
    up = data_dir() / "uploads"
    if not up.is_dir():
        return None
    for p in sorted(up.glob(f"{_check_id(image_id)}.*")):
        if not p.name.endswith(".tmp"):
            return p
    return None


# ---------------------------------------------------------------------------------------------------------------------
# Uploads
# ---------------------------------------------------------------------------------------------------------------------

def image_id(data: bytes) -> str:
    """First 12 hex chars of sha1(file bytes) — identical images always get the same id."""
    return hashlib.sha1(data).hexdigest()[:12]


def save_upload(data: bytes, filename: str) -> str:
    """Store bytes, return image_id. Re-uploading the same file is harmless (bytes are not rewritten)."""
    iid = image_id(data)
    up = _subdir("uploads")
    ext = re.sub(r"[^a-z0-9]", "", Path(str(filename)).suffix.lower())[:8] or "img"
    if _upload_path(iid) is None:
        (up / f"{iid}.{ext}").write_bytes(data)
    idx = _upload_index()
    idx[iid] = Path(str(filename).replace("\\", "/")).name or f"{iid}.{ext}"
    _write_json(up / _INDEX_NAME, idx)
    return iid


def list_uploads() -> list[dict]:
    """[{'image_id', 'filename', 'path', 'has_points', 'scores'}] sorted by filename.

    has_points is True when a saved record has at least one cell with points; scores is the saved scores dict or None.
    """
    up = data_dir() / "uploads"
    if not up.is_dir():
        return []
    idx = _upload_index()
    items = []
    for p in up.iterdir():
        if not p.is_file() or p.name == _INDEX_NAME or p.name.endswith(".tmp") or not _ID_RE.match(p.stem):
            continue
        rec = load_record(p.stem) or {}
        cells = rec.get("cells") or []
        items.append({
            "image_id": p.stem,
            "filename": idx.get(p.stem) or rec.get("filename") or p.name,
            "path": p,
            "has_points": any(len(c) > 0 for c in cells),
            "scores": rec.get("scores"),
        })
    items.sort(key=lambda d: (str(d["filename"]).lower(), d["image_id"]))
    return items


def load_image(image_id: str) -> np.ndarray:
    """RGB uint8; applies EXIF orientation (phone photos are shown the right way up)."""
    from PIL import Image, ImageOps

    path = _upload_path(image_id)
    if path is None:
        raise FileNotFoundError(f"no upload stored for image_id {image_id}")
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return np.array(im.convert("RGB"), dtype=np.uint8)


# ---------------------------------------------------------------------------------------------------------------------
# Point / settings / score records
# ---------------------------------------------------------------------------------------------------------------------

def save_record(record: dict) -> None:
    """Write data/points/<image_id>.json. numpy arrays, enums and Scores are converted; 'updated' is set to now."""
    iid = _check_id(record.get("image_id"))
    payload = dict(record)
    if isinstance(payload.get("scores"), Scores):
        payload["scores"] = scores_to_dict(payload["scores"])
    payload["updated"] = datetime.now().isoformat(timespec="seconds")
    _write_json(_subdir("points") / f"{iid}.json", payload)


def load_record(image_id: str) -> dict | None:
    """The saved record for an image, or None if there is none (or it is unreadable)."""
    path = data_dir() / "points" / f"{_check_id(image_id)}.json"
    if not path.exists():
        return None
    rec = _read_json(path)
    return rec if isinstance(rec, dict) else None


def scores_to_dict(scores: Scores) -> dict:
    """Scores dataclass -> plain JSON-able dict (numpy values become Python floats/lists)."""
    return _to_jsonable(dataclasses.asdict(scores))


# ---------------------------------------------------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------------------------------------------------

def save_png(image: np.ndarray, name: str) -> Path:
    """Write RGB image to data/exports/<name>.png and return the path."""
    from PIL import Image

    stem = _safe_name(name, "image")
    if stem.lower().endswith(".png"):
        stem = stem[:-4]
    path = _subdir("exports") / f"{stem}.png"
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)
    return path


def _continuity_mean(record: dict):
    """Cluster continuity mean if the record carries one. Accepted shapes (first match wins):
    record['continuity_mean']; record['cluster']['continuity_mean']; mean of record['continuity'][i]['score']."""
    if record.get("continuity_mean") is not None:
        return record["continuity_mean"]
    cluster = record.get("cluster")
    if isinstance(cluster, dict) and cluster.get("continuity_mean") is not None:
        return cluster["continuity_mean"]
    cont = record.get("continuity")
    if isinstance(cont, list) and cont:
        vals = [c.get("score") if isinstance(c, dict) else getattr(c, "score", None) for c in cont]
        vals = [float(v) for v in vals if v is not None]
        if vals:
            return float(np.mean(vals))
    return None


def export_csv(records: list[dict]) -> Path:
    """One row per image: filename, image_id, state, transform, rotation_offset, every scalar score, verdict,
    null percentiles, cluster continuity mean (if present).

    Null-percentile columns are named null_<metric> (one per metric found in any record). Missing values are blank.
    Written as UTF-8 with BOM so Excel opens it cleanly. File: data/exports/hex_scores_<timestamp>.csv.
    """
    rows, null_keys = [], []
    for rec in records:
        rec = _to_jsonable(rec)
        sc = rec.get("scores") or {}
        nulls = sc.get("null_percentile") or {}
        for key in nulls:
            if key not in null_keys:
                null_keys.append(key)
        cells = rec.get("cells") or []
        row = {
            "filename": rec.get("filename", ""),
            "image_id": rec.get("image_id", ""),
            "state": rec.get("state", ""),
            "transform": rec.get("transform", ""),
            "rotation_offset": rec.get("rotation_offset", ""),
            "n_cells": len(cells),
        }
        for f in _SCALAR_SCORE_FIELDS:
            row[f] = sc.get(f, "")
        row["verdict"] = sc.get("verdict", "")
        for f in _CALIBRATION_FIELDS:
            row[f] = sc.get(f, "")
        lock = rec.get("texture_lock") or {}
        row["texture_lock_p"] = lock.get("p", "")
        row["texture_locked"] = lock.get("locked", "")
        row["snapped"] = rec.get("snapped", "")
        row["edge_margin_frac"] = rec.get("edge_margin_frac", "")
        for key, val in nulls.items():
            row[f"null_{key}"] = val
        cm = _continuity_mean(rec)
        row["continuity_mean"] = "" if cm is None else cm
        row["updated"] = rec.get("updated", "")
        rows.append(row)

    header = (["filename", "image_id", "state", "transform", "rotation_offset", "n_cells", *_SCALAR_SCORE_FIELDS,
               "verdict", *_CALIBRATION_FIELDS, "texture_lock_p", "texture_locked", "snapped", "edge_margin_frac"]
              + [f"null_{k}" for k in null_keys] + ["continuity_mean", "updated"])
    path = _subdir("exports") / f"hex_scores_{_timestamp()}.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, restval="", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
    return path


def export_zip(paths: list[Path], name: str = "hex_export") -> Path:
    """Bundle files (or whole folders) into data/exports/<name>_<timestamp>.zip and return its path.

    Files are stored flat by name; folders keep their internal structure under the folder name. Duplicate names get
    a numeric suffix. Missing paths are skipped.
    """
    out = _subdir("exports") / f"{_safe_name(name, 'hex_export')}_{_timestamp()}.zip"
    used: set[str] = set()

    def unique(arc: str) -> str:
        stem, dot, ext = arc.rpartition(".")
        if not dot:
            stem, ext = arc, ""
        cand, i = arc, 1
        while cand in used:
            cand = f"{stem}_{i}.{ext}" if ext else f"{stem}_{i}"
            i += 1
        used.add(cand)
        return cand

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            p = Path(p)
            if p.resolve() == out.resolve():
                continue
            if p.is_dir():
                for f in sorted(p.rglob("*")):
                    if f.is_file() and f.resolve() != out.resolve():
                        zf.write(f, unique(f"{p.name}/{f.relative_to(p).as_posix()}"))
            elif p.is_file():
                zf.write(p, unique(p.name))
    return out
