"""Tests for hexengine.refine: corners snap onto drawn walls, with a sensible uncertainty and wall width."""

import cv2
import numpy as np
import pytest

from hexengine import canonical, refine

TRUE = canonical.VERTICES * 120.0 + np.array([200.0, 180.0])


def drawn_hexagon(wall_px: int = 5, dark_on_light: bool = True) -> np.ndarray:
    img = np.full((360, 400, 3), 210 if dark_on_light else 40, np.uint8)
    cv2.polylines(img, [np.round(TRUE).astype(np.int32)], True, (40, 40, 40) if dark_on_light else (220, 220, 220),
                  wall_px, cv2.LINE_AA)
    return cv2.GaussianBlur(img, (0, 0), 1.0)


def rough(seed: int = 0, err: float = 7.0) -> np.ndarray:
    return TRUE + np.random.default_rng(seed).uniform(-err, err, TRUE.shape)


def test_snaps_onto_wall_centreline():
    r = refine.refine_hexagon(drawn_hexagon(), rough(), "dark_ridge", scale=2.5)
    assert np.hypot(*(r.points - TRUE).T).max() < 1.0
    assert r.qc_pass and r.sigma_px < 0.5
    assert r.radius_px == pytest.approx(120.0, abs=1.0)


def test_wall_width_estimate_tracks_thickness():
    thin = refine.refine_hexagon(drawn_hexagon(3), rough(1), "dark_ridge", scale=2.0).wall_width_px
    thick = refine.refine_hexagon(drawn_hexagon(9), rough(1), "dark_ridge", scale=4.0).wall_width_px
    assert thin < thick
    assert 2.0 < thin < 10.0


def test_auto_picks_a_passing_kind():
    r = refine.refine_auto(drawn_hexagon(5, dark_on_light=False), rough(2))
    assert r.qc_pass and r.kind in ("bright_ridge", "edge")
    assert np.hypot(*(r.points - TRUE).T).max() < 1.5


def test_blank_image_fails_cleanly():
    blank = np.full((360, 400, 3), 128, np.uint8)
    with pytest.raises(ValueError):
        refine.refine_auto(blank, rough(3))
