"""Final pilot: honeycomb cell morphometry in millimetres (results_v4).

Run:  .venv\\Scripts\\python.exe experiments\\honeycomb_pilot.py <pilot_mm folder> <output folder>

Pipeline per image (engine unchanged: snap-to-walls, 2.5 % quality gate, lattice placement):
1. Scale: two points on the ruler and the known distance between them (CONFIG "ruler") -> mm per pixel.
   Cross-check: the ruler's millimetre-tick period measured along the same line (morphometry.ruler_tick_period).
2. Cells: the comb's repeat lattice is measured inside a comb-only region (CONFIG "comb", so ruler ticks can't
   dominate), cells are laid on it from an automatically found anchor, then each is snapped to the walls with the
   engine's automatic wall-type choice and must pass the quality gate.
3. Per cell: mean flat-to-flat width (wall centre to wall centre, the quantity a "10 cells = 48 mm" ruler count
   measures), the three opposite-wall widths, anisotropy, approximate inner width, residual vs noise floor, class.
4. Per image: median cell width from snapped cells, and the lattice spacing (mean of the six neighbour distances
   over the whole comb region) as a second, independent estimate. Both are compared with the ground truth.

The Necker / ternary layers are not computed here; this pilot measures size and shape only.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from hexengine import metrics, morphometry, refine, render, transforms  # noqa: E402
import batch2_test as b2  # noqa: E402  (anchor placement)
import lattice  # noqa: E402

# ruler: (x1, y1), (x2, y2) in image px on two labelled centimetre marks, and the distance between them in mm.
# ground_truth_mm: printed on the image (controls), or read from the image's own cell count against its ruler.
CONFIG = {
    "1-control-natural-comb-4p8mm.jpg": {
        # Two scales are in frame. The lower one (10-14) is the centimetre ruler on the comb: the printed "48 mm"
        # arrow runs from 9.6 to 14.4 on it. The upper scale (21-31) is a different ruler, ~0.45 of the comb scale.
        "label": "Control 1: natural dark comb", "ruler": ((67, 128), (397, 128), 40.0), "tick_strip": ((67, 160), (397, 160)),
        "comb": (0, 175, 432, 347), "ground_truth_mm": 4.80, "gt_source": "printed: 48 mm / 10 cells"},
    "2-control-foundation-5p3mm.jpg": {
        "label": "Control 2: wax foundation", "ruler": ((70, 185), (409.5, 185), 50.0), "tick_strip": ((70, 140), (409.5, 140)),
        "comb": (0, 0, 427, 112), "blank": [(112, 22, 338, 70)],      # the printed "106/20=5.3" text
        # Foundation has embossed cell bases, not walls: if the 2-D lattice is rejected, the cell spacing is
        # measured along the comb rows above and below the text (11-12 cells per row).
        "rows": [(0, 2, 427, 24), (0, 72, 427, 96)],
        "ground_truth_mm": 5.30, "gt_source": "printed: 106 mm / 20 cells"},
    "3-field-brood-steel-scale.jpg": {
        "label": "Field 3: natural brood, steel scale", "ruler": ((130, 58), (130, 207), 30.0), "tick_strip": ((118, 20), (118, 230)),
        "comb": (0, 0, 98, 240), "ground_truth_mm": None,
        "gt_source": "the image's own cell numbers 3-10 run beside the ruler (read by eye: about 5.1 mm)"},
    "4-field-brood-yellow-tape.jpg": {
        "label": "Field 4: brood comb, yellow tape", "ruler": ((52.5, 180), (461, 180), 60.0), "tick_strip": ((52.5, 182), (461, 182)),
        "comb": (0, 205, 468, 375), "ground_truth_mm": None,
        "gt_source": "the image's own count of 10 cells between two marks (read by eye: about 4.8 mm)"},
    "5-field-golden-comb.jpg": {
        "label": "Field 5: golden comb", "ruler": ((115, 19), (115, 131), 20.0), "tick_strip": ((90, 5), (90, 160)),
        "comb": (0, 0, 76, 164), "ground_truth_mm": None, "gt_source": "none"},
}
SCALES = (1.0, 1.5, 2.5)
MAX_CELLS = 30


LATTICE_SPREAD_MAX = 0.15   # six neighbour spacings must agree within 15 %, or the lattice is rejected


def comb_lattice(img, box, blank=None):
    """Comb lattice in the comb region (text boxes blanked first); raises ValueError if absent or not hexagonal."""
    x0, y0, x1, y1 = box
    roi = np.ascontiguousarray(img[y0:y1, x0:x1]).copy()
    for bx0, by0, bx1, by1 in blank or []:
        roi[by0 - y0:by1 - y0, bx0 - x0:bx1 - x0] = roi.reshape(-1, 3).mean(axis=0)
    basis = lattice.lattice_vectors(roi, min_period_px=8.0, max_period_px=max(roi.shape[:2]) / 2.2)
    d = np.hypot(*lattice.neighbour_vectors(basis).T)
    if (d.max() - d.min()) / d.mean() > LATTICE_SPREAD_MAX:
        raise ValueError(f"lattice not hexagonal (neighbour spacings {np.round(d, 1).tolist()} px)")
    return basis


def row_period(img, rows, lo_px=15.0, hi_px=60.0):
    """Mean repeat (px) along horizontal comb rows: autocorrelation of each row strip's intensity profile,
    strongest peak between lo_px and hi_px, refined over the furthest clear multiple (more cells = less error)."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    periods = []
    for x0, y0, x1, y1 in rows:
        prof = g[y0:y1, x0:x1].mean(axis=0)
        prof = prof - cv2.GaussianBlur(prof.reshape(1, -1), (0, 0), 30).ravel()
        ac = np.correlate(prof, prof, mode="full")[len(prof) - 1:]
        ac /= ac[0] + 1e-9
        k = int(lo_px) + int(np.argmax(ac[int(lo_px):int(hi_px)]))
        best = float(k)
        for mult in range(2, 12):                          # furthest multiple still clearly present
            j0 = int(round(k * mult))
            if j0 + 3 >= len(ac):
                break
            j = j0 - 3 + int(np.argmax(ac[j0 - 3:j0 + 4]))
            if ac[j] < 0.15:
                break
            y0_, y1_, y2_ = ac[j - 1], ac[j], ac[j + 1]
            den = y0_ - 2 * y1_ + y2_
            best = (j + (0.5 * (y0_ - y2_) / den if den < 0 else 0.0)) / mult
        periods.append(best)
    return float(np.mean(periods)), periods


def find_anchor(img, basis, box):
    """Anchor anywhere in the comb region: search a whole cell's worth of positions for the best wall fit."""
    x0, y0, x1, y1 = box
    guess = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    fmap = refine.feature_map(img, "bright_ridge", 1.0)
    rep = float(min(np.hypot(*basis[0]), np.hypot(*basis[1])))
    best, best_s = guess, -1.0
    for dx in np.linspace(-0.6 * rep, 0.6 * rep, 25):
        for dy in np.linspace(-0.6 * rep, 0.6 * rep, 25):
            c = guess + [dx, dy]
            cells = [lattice.cell_corners(c + i * basis[0] + j * basis[1], basis) for i in (-1, 0, 1) for j in (-1, 0, 1)]
            s = b2.edge_score(fmap, cells)
            if s > best_s:
                best, best_s = c, s
    return best


def cells_in_box(anchor, basis, box, img_shape):
    x0, y0, x1, y1 = box
    out = []
    for corners in lattice.lattice_cells(anchor, basis, img_shape, margin_frac=0.05, max_cells=400):
        if (corners[:, 0] >= x0).all() and (corners[:, 0] <= x1).all() and \
           (corners[:, 1] >= y0).all() and (corners[:, 1] <= y1).all():
            out.append(corners)
    return out[:MAX_CELLS]


def main(img_dir: Path, out_dir: Path):
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)
    rows, per_image = [], {}
    for name, cfg in CONFIG.items():
        img = cv2.cvtColor(cv2.imread(str(img_dir / name)), cv2.COLOR_BGR2RGB)
        p1, p2, mm = cfg["ruler"]
        scale = morphometry.Scale(p1, p2, mm)
        ticks = morphometry.ruler_tick_period(img, *cfg["tick_strip"], expect_px_per_mm=scale.px_per_mm)
        method, note = "2-D lattice", ""
        try:
            basis = comb_lattice(img, cfg["comb"], cfg.get("blank"))
            lattice_px = float(np.mean(np.hypot(*lattice.neighbour_vectors(basis).T)))
            anchor = find_anchor(img, basis, cfg["comb"])
            cands = cells_in_box(anchor, basis, cfg["comb"], img.shape)
            if cfg.get("blank"):
                cands = [c for c in cands if not any(
                    (bx0 - 5 <= c[:, 0].mean() <= bx1 + 5) and (by0 - 5 <= c[:, 1].mean() <= by1 + 5)
                    for bx0, by0, bx1, by1 in cfg["blank"])]
        except ValueError as exc:
            if not cfg.get("rows"):
                raise
            method, note = "row repeat", f"2-D lattice rejected: {exc}"
            lattice_px, row_px = row_period(img, cfg["rows"])
            cands = []
        cells = []
        shown = img.copy()
        cv2.line(shown, tuple(int(v) for v in p1), tuple(int(v) for v in p2), (255, 0, 255), 1, cv2.LINE_AA)
        for k, rough in enumerate(cands):
            try:
                r = refine.refine_auto(img, rough, scales=SCALES)
            except ValueError:
                continue
            fit = transforms.fit_hexagon(r.points)
            s = metrics.compute_scores(fit, with_null=False, corner_sigma_px=r.sigma_px)
            m = morphometry.measure_cell(r.points, scale, r.wall_width_px)
            row = {"image": name, "cell": k, "qc_pass": r.qc_pass, "qc_reasons": "; ".join(r.qc_reasons),
                   "radius_px": round(r.radius_px, 2), "corner_sigma_px": round(r.sigma_px, 3),
                   "wall_width_px": round(r.wall_width_px, 2), "wall_type": r.kind,
                   "width_mm": round(m.width_mm, 3), "width1_mm": round(m.widths_mm[0], 3),
                   "width2_mm": round(m.widths_mm[1], 3), "width3_mm": round(m.widths_mm[2], 3),
                   "anisotropy_mm": round(m.anisotropy_mm, 3), "inner_width_mm": round(m.inner_mm, 3),
                   "affine_residual": round(s.affine_mean, 4), "noise_floor": round(s.noise_floor, 4),
                   "excess": round(s.excess, 4), "shape": s.verdict, "size_class": m.size_class,
                   "points": np.round(r.points, 2).tolist()}
            rows.append(row)
            if r.qc_pass:
                cells.append(row)
                shown = render.draw_points(shown, r.points, labels=False)
                c = r.points.mean(axis=0)
                cv2.putText(shown, f"{m.width_mm:.2f}", (int(c[0]) - 11, int(c[1]) + 3), cv2.FONT_HERSHEY_SIMPLEX,
                            0.27, (255, 255, 0), 1, cv2.LINE_AA)
            else:
                shown = render.draw_points(shown, r.points, labels=False, color=(255, 60, 60))
        big = cv2.resize(shown, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(out_dir / "overlays" / name), cv2.cvtColor(big, cv2.COLOR_RGB2BGR))
        w = np.array([c["width_mm"] for c in cells])
        gt = cfg["ground_truth_mm"]
        res = {
            "label": cfg["label"], "ground_truth_mm": gt, "gt_source": cfg["gt_source"],
            "ruler_px_per_mm": round(scale.px_per_mm, 3), "ruler_mm_per_px": round(scale.mm_per_px, 4),
            "tick_px_per_mm": None if ticks is None else round(ticks, 3),
            "scale_disagreement_pct": None if ticks is None else round(100 * (ticks / scale.px_per_mm - 1), 2),
            "spacing_method": method, "spacing_note": note,
            "lattice_spacing_px": round(lattice_px, 2), "lattice_width_mm": round(lattice_px * scale.mm_per_px, 3),
            "cells_tried": len(cands), "cells_passed": len(cells),
            "median_width_mm": None if not len(w) else round(float(np.median(w)), 3),
            "iqr_width_mm": None if len(w) < 4 else [round(float(q), 3) for q in np.percentile(w, [25, 75])],
            "median_anisotropy_mm": None if not cells else round(float(np.median([c["anisotropy_mm"] for c in cells])), 3),
            "regular_cells": sum(c["shape"] == "regular" for c in cells),
            "classes": {k: sum(c["size_class"] == k for c in cells) for k in sorted({c["size_class"] for c in cells})},
            "median_cell_radius_px": None if not cells else round(float(np.median([c["radius_px"] for c in cells])), 1),
        }
        if gt:
            res["error_cells_mm"] = None if not len(w) else round(float(np.median(w)) - gt, 3)
            res["error_lattice_mm"] = round(lattice_px * scale.mm_per_px - gt, 3)
        per_image[name] = res
        print(f"{name:36s} px/mm ruler {scale.px_per_mm:6.3f} ticks {res['tick_px_per_mm']} | lattice {res['lattice_width_mm']} mm"
              f" | cells {len(cells)}/{len(cands)} median {res['median_width_mm']} mm | gt {gt}")

    (out_dir / "results.json").write_text(json.dumps({"per_image": per_image,
                                                      "benchmarks": morphometry.BENCHMARKS}, indent=1))
    cols = ["image", "cell", "qc_pass", "qc_reasons", "radius_px", "corner_sigma_px", "wall_width_px", "wall_type",
            "width_mm", "width1_mm", "width2_mm", "width3_mm", "anisotropy_mm", "inner_width_mm",
            "affine_residual", "noise_floor", "excess", "shape", "size_class"]
    with open(out_dir / "honeycomb_morphometry.csv", "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    print(json.dumps(per_image, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
