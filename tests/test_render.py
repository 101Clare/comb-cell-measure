"""Tests for hexengine.render: shapes/dtypes, inputs untouched, and basic drawing behaviour."""

import numpy as np
import pytest

from hexengine import canonical, render
from hexengine.types import (CubeState, EdgeContinuity, Face, FaceResult, HexFit, OrientationField, Polyline,
                             PrimaryLine, Scores, SharedEdge)

V = canonical.VERTICES
C = np.zeros(2)


def _frame(size=512, value=40):
    return np.full((size, size, 3), value, dtype=np.uint8)


def _scores():
    return Scores(0.0123, 0.03, 0.004, 0.01, 0.08, 0.15, 0.02,
                  per_vertex={"affine": [0.01] * 6},
                  null_percentile={"affine": 0.4, "homography": 1.2, "similarity": 3.0},
                  verdict="strong")


def _face_results():
    faces = [("TOP", (5, 0, 1)), ("RIGHT", (1, 2, 3)), ("LEFT", (3, 4, 5))]
    out = []
    for t, (name, ids) in zip((1, 0, -1), faces):
        poly = np.vstack([C, V[list(ids)]])
        face = Face(name, poly, ids, radial_theta=0.0, tangential_theta=np.pi / 2, axis_ids=(ids[0], ids[2]))
        out.append(FaceResult(face, dominant_theta=0.3, coherence=0.7, ternary=t,
                              radial_alignment=0.9, tangential_alignment=0.1))
    return out


def _hexfit(offset=(0.0, 0.0), r=60.0):
    pts = V * r + np.array([200.0, 200.0]) + np.asarray(offset)
    eye = np.eye(3)
    return HexFit(pts, np.arange(6), 0, eye, eye, eye)


def _check(out, inp, shape=None):
    assert out.dtype == np.uint8
    assert out.ndim == 3 and out.shape[2] == 3
    if shape is not None:
        assert out.shape == shape
    assert out is not inp


def test_draw_points_shape_and_no_mutation():
    img = _frame(300)
    before = img.copy()
    pts = V * 100 + 150
    out = render.draw_points(img, pts)
    _check(out, img, img.shape)
    assert np.array_equal(img, before)
    assert np.any(out != img)


def test_draw_points_handles_empty_and_large():
    img = _frame(200)
    assert np.array_equal(render.draw_points(img, np.zeros((0, 2))), img)
    big = np.zeros((3000, 4000, 3), np.uint8)
    out = render.draw_points(big, V * 1000 + np.array([2000, 1500]))
    assert out.shape == big.shape


def test_draw_polylines_changes_pixels_near_line_only():
    frame = _frame()
    before = frame.copy()
    line = Polyline("h", np.array([[-0.5, 0.0], [0.5, 0.0]]))
    out = render.draw_polylines(frame, [line])
    _check(out, frame, frame.shape)
    assert np.array_equal(frame, before)
    centre = frame.shape[0] // 2
    assert np.any(out[centre - 2:centre + 3, 200:310] != frame[centre - 2:centre + 3, 200:310])
    assert np.array_equal(out[:100], frame[:100])  # far from the line: untouched


def test_hidden_polylines_draw_fewer_pixels():
    frame = _frame()
    pts = np.array([[-0.8, 0.0], [0.8, 0.0], [0.8, 0.5]])
    solid = render.draw_polylines(frame, [Polyline("s", pts)])
    dashed = render.draw_polylines(frame, [Polyline("d", pts, hidden=True)])
    n_solid = np.count_nonzero(np.any(solid != frame, axis=2))
    n_dashed = np.count_nonzero(np.any(dashed != frame, axis=2))
    assert 0 < n_dashed < n_solid


def test_arrows_add_pixels():
    frame = _frame()
    pts = np.array([[-0.8, 0.0], [0.8, 0.0]])
    plain = render.draw_polylines(frame, [Polyline("p", pts)])
    arrows = render.draw_polylines(frame, [Polyline("a", pts, arrows=True)])
    assert np.count_nonzero(np.any(arrows != frame, axis=2)) > np.count_nonzero(np.any(plain != frame, axis=2))


def test_draw_construction():
    from hexengine import cube
    try:
        cube.visible_edges(CubeState.CONVEX)
    except NotImplementedError:
        pytest.skip("cube.py not implemented yet")
    frame = _frame()
    before = frame.copy()
    for state in CubeState:
        out = render.draw_construction(frame, state)
        _check(out, frame, frame.shape)
        assert np.any(out != frame)
    assert np.array_equal(frame, before)
    a = render.draw_construction(frame, CubeState.CONVEX)
    b = render.draw_construction(frame, CubeState.CONCAVE)
    assert np.any(a != b)


def test_draw_orientation():
    size = 256
    frame = _frame(size)
    before = frame.copy()
    yy, xx = np.mgrid[:size, :size]
    field = OrientationField(theta=np.full((size, size), 0.5), coherence=(xx / size).astype(float),
                             energy=np.ones((size, size)))
    out = render.draw_orientation(frame, field, step=16)
    _check(out, frame, frame.shape)
    assert np.array_equal(frame, before)
    assert np.any(out != frame)
    assert np.array_equal(out[:10, :10], frame[:10, :10])  # corner is outside the hexagon


def test_draw_ternary():
    frame = _frame()
    before = frame.copy()
    primary = PrimaryLine(C.copy(), V[1].copy(), theta=0.5, ternary=1, angle_vs_reference_deg=60.0, face_name="TOP")
    out = render.draw_ternary(frame, _face_results(), primary)
    _check(out, frame, frame.shape)
    assert np.array_equal(frame, before)
    assert np.any(out != frame)
    out2 = render.draw_ternary(frame, _face_results(), None)
    assert out2.shape == frame.shape


def test_draw_cluster():
    img = _frame(400)
    before = img.copy()
    fa, fb = _hexfit(), _hexfit(offset=(60 * np.sqrt(3), 0))
    seg = np.array([fa.image_points[1], fa.image_points[2]])
    edge = SharedEdge(0, 1, (1, 2), (5, 4), seg)
    cont = [EdgeContinuity(edge, 0.3, 0.35, 2.9, 0.95, 1, 1, "same")]
    out = render.draw_cluster(img, [fa, fb], cont, face_results=[_face_results(), _face_results()])
    _check(out, img, img.shape)
    assert np.array_equal(img, before)
    assert np.any(out != img)


def test_scores_panel():
    out = render.scores_panel(_scores(), 512, 400)
    assert out.shape == (400, 512, 3) and out.dtype == np.uint8
    assert out.std() > 0
    s = _scores()
    s.verdict = ""
    s.null_percentile = {}
    assert render.scores_panel(s).shape == (512, 512, 3)


def test_comparison_sheet_rows():
    panels = {f"p{i}": _frame(100 + 50 * i) for i in range(5)}
    panels["wide"] = np.zeros((50, 300, 3), np.uint8)
    del panels["p4"]
    before = {k: v.copy() for k, v in panels.items()}
    tile = 128
    sheet = render.comparison_sheet(panels, title="Test", tile=tile)
    assert sheet.dtype == np.uint8 and sheet.ndim == 3
    # 5 panels -> 3 columns, 2 rows: width holds 3 tiles, height holds 2 (but not 3).
    assert 3 * tile < sheet.shape[1] < 4 * tile
    assert 2 * tile < sheet.shape[0] < 3 * tile + 60
    for k in panels:
        assert np.array_equal(panels[k], before[k])
    one = render.comparison_sheet({"a": _frame(64)}, tile=tile)
    assert tile < one.shape[1] < 2 * tile


def test_comparison_sheet_accepts_grey_and_rgba():
    sheet = render.comparison_sheet({"grey": np.zeros((40, 60), np.uint8),
                                     "rgba": np.zeros((40, 60, 4), np.uint8)}, tile=96)
    assert sheet.ndim == 3 and sheet.shape[2] == 3
