"""Tests for the secondary hexagon auto-detector (hexengine.detect)."""

import cv2
import numpy as np
import pytest

from hexengine import canonical, detect
from hexengine.transforms import order_points


def _hexagon_image(centre=(400, 340), radius=200, size=(800, 700), angle_deg=0.0, shear=0.0,
                   fg=(200, 150, 60), bg=(40, 40, 40), noise=0.0, seed=0):
    """Draw a filled (optionally rotated / sheared) hexagon; returns (image, true vertices (6, 2))."""
    w, h = size
    t = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    sh = np.array([[1.0, shear], [0.0, 1.0]])
    verts = (canonical.VERTICES * radius) @ (sh @ rot).T + np.asarray(centre, float)
    img = np.full((h, w, 3), bg, dtype=np.uint8)
    # Draw with 8x sub-pixel precision (shift=3) so the true vertices are exact.
    cv2.fillPoly(img, [np.round(verts * 8).astype(np.int32)], fg, cv2.LINE_AA, shift=3)
    if noise:
        rng = np.random.default_rng(seed)
        img = np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img, verts


def _max_error(found, truth):
    ordered_truth, _ = order_points(truth)
    return float(np.max(np.hypot(*(found - ordered_truth).T)))


def test_detects_regular_hexagon_within_a_few_px():
    img, verts = _hexagon_image()
    found = detect.detect_hexagon(img)
    assert found is not None and found.shape == (6, 2)
    assert _max_error(found, verts) < 3.0


@pytest.mark.parametrize("angle, shear", [(13.0, 0.0), (0.0, 0.3), (-20.0, 0.15)])
def test_detects_rotated_and_sheared(angle, shear):
    img, verts = _hexagon_image(angle_deg=angle, shear=shear, radius=180)
    found = detect.detect_hexagon(img)
    assert found is not None
    assert _max_error(found, verts) < 3.0


def test_detects_with_noise_near_click():
    img, verts = _hexagon_image(noise=12.0, centre=(300, 300), radius=150)
    found = detect.detect_hexagon(img, near=(310, 290))
    assert found is not None
    assert _max_error(found, verts) < 4.0


def test_output_is_ordered_like_manual_clicks():
    img, verts = _hexagon_image()
    found = detect.detect_hexagon(img)
    reordered, order = order_points(found)
    assert np.array_equal(order, np.arange(6))          # already clockwise, topmost first
    assert np.allclose(reordered, found)


def test_returns_none_on_blank_image():
    assert detect.detect_hexagon(np.full((300, 400, 3), 128, np.uint8)) is None


def test_returns_none_for_a_circle():
    img = np.zeros((400, 400, 3), np.uint8)
    cv2.circle(img, (200, 200), 120, (255, 255, 255), -1, cv2.LINE_AA)
    assert detect.detect_hexagon(img) is None
