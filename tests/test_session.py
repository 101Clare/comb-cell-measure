"""Tests for hexengine.session: round trips through a temporary HEX_DATA_DIR."""

import csv
import io
import zipfile

import numpy as np
import pytest
from PIL import Image

from hexengine import session
from hexengine.types import CubeState, Scores, TransformKind


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setenv("HEX_DATA_DIR", str(tmp_path / "hexdata"))
    return tmp_path / "hexdata"


def _png_bytes(w=40, h=20, colour=(200, 10, 10)):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), colour).save(buf, format="PNG")
    return buf.getvalue()


def _scores(verdict="weak"):
    return Scores(0.01, 0.02, 0.003, 0.004, 0.05, 0.06, 0.07, per_vertex={"affine": [0.1] * 6},
                  null_percentile={"affine": 2.5, "homography": 7.0}, verdict=verdict)


def test_data_dir_env(data):
    assert session.data_dir() == data.resolve()
    assert not data.exists()  # created lazily


def test_image_id():
    a = session.image_id(b"abc")
    assert len(a) == 12 and a == session.image_id(b"abc") and a != session.image_id(b"abd")


def test_upload_list_load(data):
    b1, b2 = _png_bytes(), _png_bytes(colour=(0, 0, 255))
    id1 = session.save_upload(b1, "zebra.PNG")
    id2 = session.save_upload(b2, "../../evil/apple.png")
    assert session.save_upload(b1, "zebra.PNG") == id1  # idempotent
    ups = session.list_uploads()
    assert [u["filename"] for u in ups] == ["apple.png", "zebra.PNG"]
    assert all(u["path"].parent == data.resolve() / "uploads" for u in ups)
    assert not any(u["has_points"] for u in ups)
    img = session.load_image(id1)
    assert img.shape == (20, 40, 3) and img.dtype == np.uint8
    assert tuple(img[0, 0]) == (200, 10, 10)
    assert id2 != id1
    with pytest.raises(ValueError):
        session.load_image("../secret")


def test_load_image_exif_orientation(data):
    im = Image.new("RGB", (40, 20), (0, 255, 0))
    exif = im.getexif()
    exif[0x0112] = 6  # rotate 90 CW on display
    buf = io.BytesIO()
    im.save(buf, format="JPEG", exif=exif.tobytes())
    iid = session.save_upload(buf.getvalue(), "phone.jpg")
    assert session.load_image(iid).shape[:2] == (40, 20)


def test_record_round_trip(data):
    iid = session.save_upload(_png_bytes(), "a.png")
    assert session.load_record(iid) is None
    rec = {"image_id": iid, "filename": "a.png", "cells": [np.arange(12, dtype=float).reshape(6, 2)],
           "rotation_offset": np.int64(2), "state": CubeState.CONCAVE, "transform": TransformKind.AFFINE,
           "settings": {"opacity": np.float32(0.5)}, "scores": _scores()}
    session.save_record(rec)
    back = session.load_record(iid)
    assert back["cells"][0][5] == [10.0, 11.0]
    assert back["rotation_offset"] == 2 and back["state"] == "CONCAVE" and back["transform"] == "affine"
    assert back["scores"]["null_percentile"]["affine"] == 2.5
    assert "T" in back["updated"]
    ups = session.list_uploads()
    assert ups[0]["has_points"] and ups[0]["scores"]["verdict"] == "weak"
    assert isinstance(rec["cells"][0], np.ndarray)  # input untouched


def test_scores_to_dict():
    d = session.scores_to_dict(_scores())
    assert d["affine_mean"] == 0.01 and d["per_vertex"]["affine"] == [0.1] * 6


def test_export_csv_png_zip(data):
    recs = []
    for i, name in enumerate(["x.png", "y.png"]):
        iid = session.save_upload(_png_bytes(colour=(i, 0, 0)), name)
        rec = {"image_id": iid, "filename": name, "cells": [[[0, 0]] * 6], "rotation_offset": 0,
               "state": "CONVEX", "transform": "affine", "settings": {},
               "scores": session.scores_to_dict(_scores("strong" if i else "noise"))}
        if i:
            rec["continuity"] = [{"score": 0.8}, {"score": 1.0}]
        session.save_record(rec)
        recs.append(session.load_record(iid))
    recs.append({"image_id": "000000000000", "filename": "unscored.png", "scores": None})
    path = session.export_csv(recs)
    assert path.parent == data.resolve() / "exports" and path.suffix == ".csv"
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3
    assert rows[0]["verdict"] == "noise" and float(rows[1]["null_homography"]) == 7.0
    assert float(rows[1]["continuity_mean"]) == pytest.approx(0.9) and rows[0]["continuity_mean"] == ""
    assert rows[2]["affine_mean"] == ""

    png = session.save_png(np.zeros((10, 10, 3), np.uint8), "sheet ../bad name")
    assert png.exists() and png.parent == data.resolve() / "exports"

    z = session.export_zip([path, png, data / "points", data / "missing.txt"], name="bundle")
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist()
    assert path.name in names and png.name in names
    assert sum(n.startswith("points/") for n in names) == 2
