"""Twist (chirality) of line texture around a centre: which way a spiral turns, and at what pitch.

The structure tensor (inference.orientation_field) gives a line's direction but not its handedness. Here each
pixel's line direction is compared with the radial direction from a chosen centre. The signed offset
delta = line angle - radial angle, folded into (-90, 90] degrees, is 0 for radial lines, +-90 for concentric
rings, and a consistent +psi for a spiral whose arms cross every radius at the pitch angle psi.

Two read-outs, in a ring-shaped band r_in < r < r_out:

* chirality index  C = weighted mean of sin(2 delta)  (weights: coherence x gradient strength).
  Sign = handedness, size = how strongly and consistently the texture twists. Its 95 % interval comes from a
  bootstrap over angular sectors, and `sectors_same_sign` counts how many sectors agree.
  C is NOT a pitch: sin(2 delta) is the same for psi and 90 - psi (a 15 deg spiral and a 75 deg spiral both give
  C = 0.5), and a drawing that mixes line families (chords and near-tangential polygon sides) averages them.
* pitch peaks: the weighted histogram of delta, smoothed; its peaks are the pitch angles of the line families
  present. Calibrated on synthetic logarithmic spirals of known pitch (tests/test_twist.py): a single-family
  spiral is recovered to within about 2 degrees.

Sign convention (image coordinates, y down): delta > 0 means a line leans clockwise of the radial direction ON
SCREEN, so following it outward turns clockwise (inward, anticlockwise). The mirror image gives exactly -C.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Twist:
    C: float
    ci95: tuple[float, float]
    sectors_same_sign: int
    n_sectors: int
    peaks: list[dict] = field(default_factory=list)   # [{"pitch_deg", "share"}], strongest first
    handedness: str = ""                              # "clockwise-out", "anticlockwise-out" or "none"


def line_field(grey: np.ndarray, sigma_d: float = 1.0, sigma_i: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel line direction (radians, along the lines) and weight (coherence x gradient strength)."""
    g = cv2.GaussianBlur(grey.astype(np.float64), (0, 0), sigma_d)
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    jxx, jyy, jxy = (cv2.GaussianBlur(a, (0, 0), sigma_i) for a in (gx * gx, gy * gy, gx * gy))
    angle = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2
    tr = jxx + jyy
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (tr + 1e-9)
    return angle, coh * np.sqrt(tr)


def _fold(delta: np.ndarray) -> np.ndarray:
    """Axial angle difference folded into (-pi/2, pi/2]."""
    return (delta + np.pi / 2) % np.pi - np.pi / 2


def measure(image: np.ndarray, centre, r_in: float, r_out: float, mask: np.ndarray | None = None,
            n_sectors: int = 12, n_boot: int = 5000, sigma_d: float = 1.0, sigma_i: float = 3.0,
            bin_deg: float = 2.0, smooth_deg: float = 3.0, min_share: float = 0.1, seed: int = 0) -> Twist:
    """Twist of the line texture in the band r_in < r < r_out around `centre` (x, y). `mask` (bool, image-sized)
    excludes pixels where it is False, e.g. an overlay or a hand."""
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    ang, w = line_field(grey, sigma_d, sigma_i)
    yy, xx = np.mgrid[: grey.shape[0], : grey.shape[1]]
    dx, dy = xx - centre[0], yy - centre[1]
    r, phi = np.hypot(dx, dy), np.arctan2(dy, dx)
    m = (r > r_in) & (r < r_out)
    if mask is not None:
        m &= mask
    if not m.any():
        raise ValueError("the band is empty")
    delta, ww, ph = _fold(ang[m] - phi[m]), w[m], phi[m]
    s = np.sin(2 * delta) * ww
    sector = ((ph + np.pi) / (2 * np.pi) * n_sectors).astype(int) % n_sectors
    num, den = np.bincount(sector, s, n_sectors), np.bincount(sector, ww, n_sectors)
    C = float(num.sum() / den.sum())
    idx = np.random.default_rng(seed).integers(0, n_sectors, (n_boot, n_sectors))
    boot = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-12)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    per = num / np.maximum(den, 1e-12)
    same = int(np.sum(np.sign(per) == np.sign(C))) if C != 0 else 0

    # Pitch histogram, circular over the 180-degree axial range.
    nb = int(round(180 / bin_deg))
    edges = np.linspace(-90, 90, nb + 1)
    hist, _ = np.histogram(np.degrees(delta), edges, weights=ww)
    k = max(1, int(round(smooth_deg / bin_deg)))
    kern = np.exp(-0.5 * (np.arange(-3 * k, 3 * k + 1) / k) ** 2)
    padded = np.r_[hist[-3 * k:], hist, hist[: 3 * k]]
    sm = np.convolve(padded, kern / kern.sum(), mode="same")[3 * k: 3 * k + nb]
    centres = 0.5 * (edges[:-1] + edges[1:])
    total = sm.sum()
    tops = [i for i in range(nb) if sm[i] >= sm[i - 1] and sm[i] > sm[(i + 1) % nb]]
    peaks = []
    if tops:
        # share: every bin goes to its nearest peak (circular), so the shares add up to 1
        dist = np.abs(((centres[:, None] - centres[tops][None, :]) + 90) % 180 - 90)
        owner = np.argmin(dist, axis=1)
        for j, i in enumerate(tops):
            y0, y1, y2 = sm[i - 1], sm[i], sm[(i + 1) % nb]          # parabolic refinement of the position
            den_p = y0 - 2 * y1 + y2
            off = 0.5 * (y0 - y2) / den_p if den_p < 0 else 0.0
            pos = ((centres[i] + off * bin_deg) + 90) % 180 - 90
            peaks.append({"pitch_deg": round(float(pos), 1),
                          "share": round(float(sm[owner == j].sum() / total), 3)})
    peaks = sorted([p for p in peaks if p["share"] >= min_share], key=lambda p: -p["share"])
    hand = "none" if lo <= 0 <= hi else ("clockwise-out" if C > 0 else "anticlockwise-out")
    return Twist(C=round(C, 4), ci95=(round(float(lo), 4), round(float(hi), 4)), sectors_same_sign=same,
                 n_sectors=n_sectors, peaks=peaks, handedness=hand)


def synthetic_spiral(pitch_deg: float, size: int = 600, arms: int = 12, r0: float = 30.0, r1: float = 280.0,
                     thickness: int = 2, mirror: bool = False) -> np.ndarray:
    """Dark logarithmic-spiral arms on a light ground, crossing every radius at `pitch_deg` (0 = radial lines,
    90 = concentric circles). Positive pitch turns clockwise on screen going outward. For calibration and tests."""
    img = np.full((size, size), 235, np.uint8)
    c = size / 2
    psi = np.radians(pitch_deg)
    for a in range(arms):
        th0 = 2 * np.pi * a / arms
        if abs(pitch_deg) >= 89.9:
            r = r0 + (r1 - r0) * a / max(arms - 1, 1)
            cv2.circle(img, (int(c), int(c)), int(r), 40, thickness, cv2.LINE_AA)
            continue
        if abs(pitch_deg) < 0.1:
            th = np.full(200, th0)
            rr = np.linspace(r0, r1, 200)
        else:
            k = 1 / np.tan(psi)                 # r = r0 * exp(k * (theta - th0))
            span = np.log(r1 / r0) / k
            th = th0 + np.linspace(0, span, 2000)
            rr = r0 * np.exp(k * (th - th0))
        pts = np.c_[c + rr * np.cos(th), c + rr * np.sin(th)]
        cv2.polylines(img, [np.round(pts * 16).astype(np.int32)], False, 40, thickness, cv2.LINE_AA, shift=4)
    img = cv2.GaussianBlur(img, (0, 0), 0.7)
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return rgb[:, ::-1].copy() if mirror else rgb
