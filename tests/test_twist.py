"""Twist / chirality: calibration on synthetic logarithmic spirals of known pitch, and the controls."""

import cv2
import numpy as np
import pytest

from hexengine import twist

CENTRE, R_IN, R_OUT = (300, 300), 60, 260


@pytest.mark.parametrize("pitch", [5, 15, 30, 45, 60, 75])
def test_pitch_and_index_recovered(pitch):
    t = twist.measure(twist.synthetic_spiral(pitch), CENTRE, R_IN, R_OUT, n_boot=300)
    assert t.C == pytest.approx(np.sin(np.radians(2 * pitch)), abs=0.01)
    assert t.peaks[0]["pitch_deg"] == pytest.approx(pitch, abs=2)
    assert t.peaks[0]["share"] > 0.9
    assert t.handedness == "clockwise-out" and t.sectors_same_sign == 12


@pytest.mark.parametrize("pitch", [0, 90])
def test_radial_and_concentric_have_no_twist(pitch):
    t = twist.measure(twist.synthetic_spiral(pitch), CENTRE, R_IN, R_OUT, n_boot=300)
    assert abs(t.C) < 0.02 and t.handedness == "none"
    assert abs(abs(t.peaks[0]["pitch_deg"]) - pitch) < 2


def test_index_alone_cannot_tell_15_from_75_but_peaks_can():
    a = twist.measure(twist.synthetic_spiral(15), CENTRE, R_IN, R_OUT, n_boot=300)
    b = twist.measure(twist.synthetic_spiral(75), CENTRE, R_IN, R_OUT, n_boot=300)
    assert a.C == pytest.approx(b.C, abs=0.02)
    assert a.peaks[0]["pitch_deg"] == pytest.approx(15, abs=2)
    assert b.peaks[0]["pitch_deg"] == pytest.approx(75, abs=2)


def test_mirror_flips_the_sign():
    a = twist.measure(twist.synthetic_spiral(30), CENTRE, R_IN, R_OUT, n_boot=300)
    b = twist.measure(twist.synthetic_spiral(30, mirror=True), (599 - CENTRE[0], CENTRE[1]), R_IN, R_OUT,
                      n_boot=300)
    assert b.C == pytest.approx(-a.C, abs=0.01)
    assert b.handedness == "anticlockwise-out"
    assert b.peaks[0]["pitch_deg"] == pytest.approx(-30, abs=2)


def test_centred_regular_hexagons_are_achiral():
    """Nested, centred, unrotated hexagon outlines: each side leans both ways about the tangent equally."""
    img = np.full((600, 600), 235, np.uint8)
    for r in range(70, 260, 30):
        pts = [(300 + r * np.cos(np.radians(60 * k)), 300 + r * np.sin(np.radians(60 * k))) for k in range(6)]
        cv2.polylines(img, [np.round(np.array(pts)).astype(np.int32)], True, 40, 2, cv2.LINE_AA)
    t = twist.measure(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), CENTRE, R_IN, R_OUT, n_boot=300)
    assert abs(t.C) < 0.03 and t.handedness == "none"


def test_mask_and_empty_band():
    img = twist.synthetic_spiral(20)
    mask = np.ones(img.shape[:2], bool)
    mask[:, :300] = False                                 # only the right half
    t = twist.measure(img, CENTRE, R_IN, R_OUT, mask=mask, n_boot=300)
    assert t.peaks[0]["pitch_deg"] == pytest.approx(20, abs=2)
    with pytest.raises(ValueError):
        twist.measure(img, CENTRE, 10, 5)
