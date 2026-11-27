"""Synthetic comb photographed at a tilt (weak perspective: a squash along one direction)."""

import cv2
import numpy as np

from test_comb import PX_PER_MM, synthetic_comb


def tilted_comb(width_mm=5.2, tilt_deg=15.0, axis_deg=30.0, angle_deg=7.0, size=(560, 760), seed=0):
    """Returns (image, ruler_p1, ruler_p2, ruler_mm, box). The ruler is 50 mm in the comb plane."""
    img = synthetic_comb(width_mm, size=size, angle_deg=angle_deg, seed=seed)
    h, w = size
    c = np.array([w / 2, h / 2])
    a = np.radians(axis_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    M = R @ np.diag([1.0, np.cos(np.radians(tilt_deg))]) @ R.T
    A = np.hstack([M, (c - M @ c)[:, None]])
    out = cv2.warpAffine(img, A, (w, h), flags=cv2.INTER_LINEAR, borderValue=(70, 70, 70))
    p1, p2 = c + np.array([-250.0, -230.0]), c + np.array([250.0, -230.0])
    q1, q2 = M @ (p1 - c) + c, M @ (p2 - c) + c
    box = (c[0] - 240, c[1] - 170, c[0] + 240, c[1] + 200)
    return out, tuple(q1), tuple(q2), 500 / PX_PER_MM, box
