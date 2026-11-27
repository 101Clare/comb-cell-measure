"""Camera-tilt estimate and correction in comb.measure_comb."""

import numpy as np
import pytest

from hexengine import comb
from tilt_helpers import tilted_comb


def test_straight_on_photo_is_left_alone():
    img, p1, p2, mm, box = tilted_comb(5.2, 0.0, 0.0)
    r = comb.measure_comb(img, p1, p2, mm, box, max_cells=40, draw=False)
    assert r.tilt_deg < 2.0 and not r.tilt_corrected
    assert r.median_width_mm == pytest.approx(5.2, abs=0.01)
    assert r.lattice_width_mm == pytest.approx(5.2, abs=0.02)


@pytest.mark.parametrize("tilt,axis", [(10.0, 100.0), (25.0, 45.0), (30.0, 120.0)])
def test_tilt_is_found_and_corrected(tilt, axis):
    img, p1, p2, mm, box = tilted_comb(5.2, tilt, axis)
    r = comb.measure_comb(img, p1, p2, mm, box, max_cells=40, draw=False)
    assert r.tilt_deg == pytest.approx(tilt, abs=1.0)
    assert r.tilt_corrected
    assert r.median_width_mm == pytest.approx(5.2, abs=0.01)
    assert r.lattice_width_mm == pytest.approx(5.2, abs=0.01)
    assert abs(r.median_uncorrected_mm - 5.2) > abs(r.median_width_mm - 5.2)
    assert any("tilted" in w for w in r.warnings) == (tilt > comb.TILT_WARN_DEG)


def test_regularising_stretch_of_a_regular_lattice_is_identity():
    th = np.radians(17)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    basis = np.array([rot @ [50.0, 0.0], rot @ [25.0, 25.0 * np.sqrt(3)]])
    S, tilt = comb.regularising_stretch(basis)
    assert tilt == pytest.approx(0.0, abs=1e-3)
    assert np.allclose(S, np.eye(2), atol=1e-9)
