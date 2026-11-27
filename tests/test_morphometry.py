"""Tests for hexengine.morphometry: scale, flat-to-flat widths, classes, ruler tick period."""

import numpy as np
import pytest

from hexengine import canonical, morphometry


def test_scale_and_flat_to_flat():
    s = morphometry.Scale((0, 0), (300, 400), 50.0)          # 500 px = 50 mm
    assert s.px_per_mm == pytest.approx(10.0) and s.mm_per_px == pytest.approx(0.1)
    R = 30.0
    hexagon = canonical.VERTICES * R + 200.0
    w = morphometry.flat_to_flat_px(hexagon)
    assert np.allclose(w, np.sqrt(3) * R)
    m = morphometry.measure_cell(hexagon, s, wall_width_px=4.0)
    assert m.width_mm == pytest.approx(np.sqrt(3) * R / 10) and m.anisotropy_mm == pytest.approx(0, abs=1e-9)
    assert m.inner_mm == pytest.approx(m.width_mm - 0.4)
    stretched = hexagon.copy(); stretched[:, 0] = (stretched[:, 0] - 200) * 1.1 + 200
    assert morphometry.measure_cell(stretched, s).anisotropy_mm > 0.1


@pytest.mark.parametrize("w,label", [(4.9, "natural / small-cell worker"), (5.3, "standard worker"),
                                     (5.8, "intermediate / storage"), (6.4, "drone"), (5.17, "between classes"),
                                     (4.5, "below natural worker range"), (7.0, "above drone range")])
def test_classify(w, label):
    assert morphometry.classify(w) == label


def test_ruler_tick_period_on_synthetic_ruler():
    img = np.full((60, 800, 3), 220, np.uint8)
    px_per_mm = 7.3
    for k in range(0, 105):
        x = 20 + k * px_per_mm
        h = 30 if k % 10 == 0 else (22 if k % 5 == 0 else 14)
        img[10:10 + h, int(round(x)):int(round(x)) + 1] = 30
    got = morphometry.ruler_tick_period(img, (20, 18), (780, 18), expect_px_per_mm=7.0)
    assert got == pytest.approx(px_per_mm, rel=0.01)
    with pytest.raises(ValueError):
        morphometry.Scale((5, 5), (5, 5), 10.0).mm_per_px
