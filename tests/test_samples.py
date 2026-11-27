"""The synthetic sample generator runs, is deterministic, and its ground truth is well-formed."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from samples import generate_samples as gs

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples"
EXPECTED_FILES = [name for name, _ in gs.SAMPLES]


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    out = tmp_path_factory.mktemp("samples")
    truth = gs.generate(out)
    return out, truth


def _check_truth(truth, folder):
    assert set(truth) == set(EXPECTED_FILES)
    for name, entry in truth.items():
        img = cv2.imread(str(folder / name))
        assert img is not None, name
        h, w = img.shape[:2]
        assert entry["image_size"] == [w, h]
        assert entry["description"]
        assert "expected" in entry
        polys = [entry["vertices"]] + [c["vertices"] for c in entry.get("cells", [])]
        for verts in polys:
            pts = np.asarray(verts, float)
            assert pts.shape == (6, 2), name
            assert np.all(pts[:, 0] >= 0) and np.all(pts[:, 0] < w), name
            assert np.all(pts[:, 1] >= 0) and np.all(pts[:, 1] < h), name
            # clockwise on screen (y down) => positive shoelace area; vertex 0 is the topmost-angle vertex
            x, y = pts[:, 0], pts[:, 1]
            assert 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) > 0, name


def test_generator_runs_and_writes_everything(generated):
    out, truth = generated
    for name in EXPECTED_FILES:
        assert (out / name).is_file()
    assert (out / "ground_truth.json").is_file()
    _check_truth(truth, out)


def test_generator_is_deterministic(generated, tmp_path):
    out, truth = generated
    again = gs.generate(tmp_path)
    assert again == truth
    for name in EXPECTED_FILES:
        assert (out / name).read_bytes() == (tmp_path / name).read_bytes(), name


def test_committed_samples_exist_and_are_valid():
    gt = SAMPLES_DIR / "ground_truth.json"
    assert gt.is_file(), "run: python samples/generate_samples.py"
    truth = json.loads(gt.read_text(encoding="utf-8"))
    _check_truth(truth, SAMPLES_DIR)


def test_honeycomb_has_centre_plus_three_neighbours():
    truth = json.loads((SAMPLES_DIR / "ground_truth.json").read_text(encoding="utf-8"))
    hc = truth["honeycomb.png"]
    assert len(hc["cells"]) == 4
    assert hc["cells"][0]["vertices"] == hc["vertices"]
    assert all(len(e["edge_a"]) == 2 for e in hc["expected"]["shared_edges"])


def test_reference_scores_tell_the_samples_apart():
    truth = json.loads((SAMPLES_DIR / "ground_truth.json").read_text(encoding="utf-8"))
    ref = {k: v["expected"]["reference_scores"] for k, v in truth.items()}
    assert ref["regular_hexagon.png"]["affine_mean"] < 0.01
    assert ref["sheared_hexagon.png"]["affine_mean"] < 0.01 < ref["sheared_hexagon.png"]["similarity_mean"]
    assert ref["perspective_hexagon.png"]["homography_mean"] < 0.01 < ref["perspective_hexagon.png"]["affine_mean"]
    assert ref["cube_off_axis.png"]["central_symmetry_error"] < 0.01 < ref["cube_off_axis.png"]["affine_mean"]
    assert ref["irregular_hexagon.png"]["affine_mean"] > 0.05


def test_reference_scores_agree_with_the_engine():
    """The committed reference numbers use the same definitions as hexengine.metrics (regression: the
    central-symmetry reference used to be twice the engine's value)."""
    from hexengine import metrics, transforms

    truth = json.loads((SAMPLES_DIR / "ground_truth.json").read_text(encoding="utf-8"))
    for name, entry in truth.items():
        ref = entry["expected"]["reference_scores"]
        sc = metrics.compute_scores(transforms.fit_hexagon(np.asarray(entry["vertices"], float)), with_null=False)
        for key in ("similarity_mean", "affine_mean", "homography_mean", "central_symmetry_error"):
            assert getattr(sc, key) == pytest.approx(ref[key], abs=1e-3), (name, key)
