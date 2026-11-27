"""n-gon snapping and cyclic-ring geometry on a synthetic 41 / 19 degree 12-gon with spiral chords."""

import cv2
import numpy as np
import pytest

from hexengine import polygon

C, R = np.array([300.0, 300.0]), 220.0


def ring_corners(long_step=41.0, start=-90.0):
    angs, a = [], start
    for k in range(12):
        angs.append(a)
        a += long_step if k % 2 == 0 else 60 - long_step
    return np.array([C + R * np.array([np.cos(np.radians(t)), np.sin(np.radians(t))]) for t in angs])


def drawing(corners, chords=True):
    img = np.full((600, 600), 235, np.uint8)
    cv2.polylines(img, [np.round(corners * 16).astype(np.int32)], True, 40, 2, cv2.LINE_AA, shift=4)
    if chords:                                      # lines running into every corner, as in hand-drafted constructions
        for p in corners:
            q = C + 0.45 * (p - C) @ np.array([[np.cos(0.5), -np.sin(0.5)], [np.sin(0.5), np.cos(0.5)]])
            cv2.line(img, tuple(np.round(p).astype(int)), tuple(np.round(q).astype(int)), 40, 2, cv2.LINE_AA)
    return cv2.cvtColor(cv2.GaussianBlur(img, (0, 0), 0.7), cv2.COLOR_GRAY2RGB)


def test_snap_recovers_corners_and_steps():
    true = ring_corners()
    img = drawing(true)
    rough = true + np.random.default_rng(1).uniform(-3, 3, true.shape)
    s = polygon.snap_polygon(img, rough)
    assert np.max(np.hypot(*(s.points - true).T)) < 0.6
    assert s.suspect == []
    ring = polygon.cyclic_ring(s.points)
    assert ring.radius_px == pytest.approx(R, abs=0.5)
    assert ring.radial_rms_frac < 0.002
    assert ring.long_step_deg == pytest.approx(41.0, abs=0.2)
    assert ring.short_step_deg == pytest.approx(19.0, abs=0.2)
    assert ring.long_short_side_ratio == pytest.approx(np.sin(np.radians(20.5)) / np.sin(np.radians(9.5)), rel=0.01)


def test_occluded_side_uses_its_visible_ends():
    true = ring_corners()
    img = drawing(true, chords=False)
    a, b = true[3], true[4]
    mid = (0.5 * (a + b)).astype(int)
    cv2.circle(img, tuple(mid), 25, (90, 90, 90), -1)        # a play-button-like blob over the middle of side 4
    s = polygon.snap_polygon(img, true, occluded={3: [(0.0, 0.2), (0.8, 1.0)]})
    assert np.max(np.hypot(*(s.points - true).T)) < 0.8


def test_wrong_rough_mark_is_flagged():
    true = ring_corners()
    img = drawing(true)
    rough = true.copy()
    rough[5] += (0, 12)                                       # 12 px off: the snap lands elsewhere
    s = polygon.snap_polygon(img, rough, search=14)
    assert s.suspect


def test_ring_geometry_and_exclusion():
    pts = ring_corners(30.0)                                  # regular 12-gon
    ring = polygon.cyclic_ring(pts)
    assert ring.long_step_deg == pytest.approx(30, abs=1e-6) and ring.long_short_side_ratio == pytest.approx(1)
    bad = pts.copy()
    bad[2] += (15, 0)
    assert polygon.cyclic_ring(bad, exclude=[2]).radial_rms_frac < 1e-9
    assert polygon.cyclic_ring(pts[::-1]).long_step_deg == pytest.approx(30, abs=1e-6)   # either winding


def test_step_candidates():
    names = [c["name"] for c in polygon.step_candidates(41.2, 0.3)]
    assert "arctan(sqrt3/2)" in names and "arccos(3/4)" in names and "30 (regular 12-gon)" not in names
    assert polygon.step_candidates(30.0, 0.1)[0]["name"] == "30 (regular 12-gon)"
