"""One-call comb measurement on synthetic comb of known cell width."""

import cv2
import numpy as np
import pytest

from hexengine import comb

PX_PER_MM = 10.0


def synthetic_comb(width_mm=5.2, size=(500, 700), angle_deg=7.0, seed=0):
    """Light wax walls on dark cells, pointy-top hexagons whose flat-to-flat width is width_mm."""
    h, w = size
    d = width_mm * PX_PER_MM                       # centre spacing = flat-to-flat width
    R = d / np.sqrt(3)                             # circumradius
    th = np.radians(angle_deg)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    img = np.full((h, w), 70, np.uint8)
    a1, a2 = rot @ np.array([d, 0.0]), rot @ np.array([d / 2, d * np.sqrt(3) / 2])
    for i in range(-30, 30):
        for j in range(-30, 30):
            c = np.array([w / 2, h / 2]) + i * a1 + j * a2
            if not (-d < c[0] < w + d and -d < c[1] < h + d):
                continue
            pts = [c + rot @ (R * np.array([np.cos(np.radians(60 * k + 30)), np.sin(np.radians(60 * k + 30))]))
                   for k in range(6)]
            cv2.polylines(img, [np.round(np.array(pts) * 16).astype(np.int32)], True, 215, 4, cv2.LINE_AA, shift=4)
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    img = np.clip(img + np.random.default_rng(seed).normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


@pytest.mark.parametrize("width", [4.9, 5.4])
def test_measures_known_width(width):
    img = synthetic_comb(width)
    r = comb.measure_comb(img, (100, 20), (600, 20), 500 / PX_PER_MM, (80, 60, 620, 460))
    assert r.px_per_mm == pytest.approx(PX_PER_MM)
    assert r.cells_passed >= 20
    assert r.median_width_mm == pytest.approx(width, abs=0.05)
    assert r.lattice_width_mm == pytest.approx(width, abs=0.05)
    assert r.overlay is not None and r.overlay.shape == img.shape


def test_size_class_and_box_order():
    img = synthetic_comb(6.4)
    r = comb.measure_comb(img, (100, 20), (600, 20), 50.0, (620, 460, 80, 60))   # box given corner-reversed
    assert r.size_class == "drone"


def test_no_comb_in_box_is_a_clear_error():
    img = np.full((400, 400, 3), 128, np.uint8)
    img += np.random.default_rng(0).integers(0, 30, img.shape, dtype=np.uint8)
    with pytest.raises(ValueError):
        comb.measure_comb(img, (10, 10), (300, 10), 30.0, (50, 50, 350, 350))


def test_bad_ruler_inputs():
    img = synthetic_comb()
    with pytest.raises(ValueError):
        comb.measure_comb(img, (100, 20), (105, 20), 5.0, (80, 60, 620, 460))
    with pytest.raises(ValueError):
        comb.measure_comb(img, (100, 20), (600, 20), 0.0, (80, 60, 620, 460))
