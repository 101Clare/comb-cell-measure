"""Baseline test of the Hexagonal Geometric Test Engine on real photographs.

Run:  .venv\\Scripts\\python.exe experiments\\baseline_test.py <image folder> <output folder>

For every image in CONFIG it gets hexagons (seeded watershed cells or hand-placed rough corners), refines their
corners onto the walls/edges (hexengine.refine) and then measures, per hexagon:

1. Geometry:  affine / homography / similarity mean residual, central-symmetry error, random-hexagon percentile.
2. Noise floor: the affine residual a PERFECT regular hexagon of the same pixel size would show if its corners
   had the measured corner uncertainty (Monte Carlo, 95th percentile). A residual below the floor is
   indistinguishable from a perfect hexagon; only the excess above it is real shape information.
3. Necker: ternary states and primary line under both readings.
4. Six-fold texture lock: the orientation field inside the cell (outline margin excluded) gives a hexatic order
   Z = sum w * exp(i 6 (theta - 90 deg)) / sum w. Its phase psi = arg(Z)/6 says where the texture's six-fold
   pattern sits relative to the hexagon's own axes: psi ~ 0 = along the cube axes (radial), psi ~ +-30 = across
   them (tangential). Per image group, a rotation-shuffle test asks whether cells lock to their own outline
   more than randomly rotated cells would.
5. Continuity: for neighbouring hexagons, the engine's edge continuity score, compared with the same score for
   random NON-neighbouring pairs (a null: if neighbours are not more continuous than strangers, there is no
   cross-boundary signal).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hexengine import canonical, cells, inference, metrics, neighbours, refine, render, transforms  # noqa: E402
from hexengine.types import CubeState  # noqa: E402

RNG = np.random.default_rng(2026)

def lattice(centres, half_w, half_h):
    """Rough pointy-top hexagons around given centres (for a regular comb seen straight on)."""
    return [[(cx, cy - half_h), (cx + half_w, cy - half_h / 2), (cx + half_w, cy + half_h / 2), (cx, cy + half_h),
             (cx - half_w, cy + half_h / 2), (cx - half_w, cy - half_h / 2)] for cx, cy in centres]


# Every hexagon starts from rough corners placed by eye (image px, in order around the hexagon); refinement then
# snaps them onto the walls/edges. kind/scale set the wall detector (scale ~ half the wall width, px).
# "components": cells are found automatically as bright regions between thin dark walls (clean froth only).
CONFIG = {
    # group 1: control
    "1-tiles-c-dark-pattern.jpg": {"group": "1 Tiles (control)", "kind": "edge", "scale": 1.2,
                                   "manual": [[(240, 115), (280, 137), (280, 184), (240, 207), (200, 184), (200, 139)]]},
    "1-tiles-b-floor-photo.jpg": {"group": "1 Tiles (control)", "kind": "edge", "scale": 1.5,
                                  "manual": [[(273, 49), (327, 82), (329, 175), (270, 205), (206, 173), (215, 79)]]},
    # group 2: honeycomb
    "2-honeycomb-b-straight-on.jpg": {"group": "2 Honeycomb", "kind": "bright_ridge", "scale": 3.0,
        "manual": lattice([(230, 109), (340, 109), (450, 109), (175, 192), (285, 192), (395, 192), (505, 192),
                           (230, 275), (340, 275), (450, 275), (175, 358), (285, 358), (395, 358)], 55, 55)},
    "2-honeycomb-a-backlit.jpg": {"group": "2 Honeycomb", "kind": "dark_ridge", "scale": 7.0, "manual": [
        [(285, 115), (378, 130), (380, 290), (275, 325), (185, 295), (185, 150)],
        [(475, 105), (580, 140), (580, 300), (480, 325), (380, 290), (378, 130)],
        [(380, 290), (480, 325), (475, 470), (370, 505), (265, 470), (270, 325)]]},
    # group 3: basalt
    "3-basalt-a-large-columns.jpg": {"group": "3 Basalt", "kind": "dark_ridge", "scale": 2.5, "manual": [
        [(115, 150), (235, 160), (245, 270), (200, 345), (110, 350), (55, 240)],
        [(245, 150), (400, 140), (420, 230), (390, 295), (260, 290), (245, 200)]]},
    "3-basalt-b-overhead.jpg": {"group": "3 Basalt", "kind": "dark_ridge", "scale": 2.0, "manual": [
        [(197, 137), (253, 137), (268, 168), (247, 217), (165, 215), (158, 165)],
        [(263, 180), (320, 180), (347, 233), (333, 273), (253, 270), (240, 223)]]},
    # group 4: bubbles
    "4-bubbles-a-blue-single-layer.jpg": {"group": "4 Bubbles", "kind": "bright_ridge", "scale": 2.0, "manual": [
        [(173, 205), (258, 294), (350, 282), (372, 231), (327, 151), (263, 139)]]},
    "4-bubbles-b-black-white-froth.jpg": {"group": "4 Bubbles", "kind": "dark_ridge", "scale": 1.2,
                                          "components": {"min_area": 250}},
    # group 5: turtle
    "5-turtle-a-domed-scutes.jpg": {"group": "5 Turtle", "kind": "dark_ridge", "scale": 4.0, "manual": [
        [(185, 125), (290, 100), (345, 160), (330, 300), (230, 330), (160, 250)]]},
    "5-turtle-b-close-scutes.jpg": {"group": "5 Turtle", "kind": "dark_ridge", "scale": 4.0, "manual": [
        [(150, 150), (240, 135), (320, 145), (325, 250), (230, 285), (145, 270)],
        [(330, 145), (420, 135), (470, 160), (480, 270), (410, 290), (330, 250)]]},
    # group 6: snowflakes (the inner hexagonal plate outline)
    "6-snowflake-a-stellar-plate.jpg": {"group": "6 Snowflake", "kind": "bright_ridge", "scale": 1.2,
                                        "manual": [[(222, 125), (290, 144), (305, 209), (257, 257), (189, 240), (172, 171)]]},
    "6-snowflake-b-dark-background.jpg": {"group": "6 Snowflake", "kind": "bright_ridge", "scale": 1.5,
                                          "manual": [[(202, 72), (272, 99), (280, 182), (215, 222), (140, 192), (132, 114)]]},
}


def froth_cells(img, min_area):
    """Clean froth: cells are the bright connected regions between thin dark walls."""
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (0, 0), 1.0)
    _, binary = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary)
    h, w = g.shape
    out, sides = [], {}
    for k in range(1, n):
        x, y, bw, bh, area = stats[k]
        if area < min_area or x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
            continue
        cnt, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        poly = cv2.approxPolyDP(cnt[0], 0.04 * cv2.arcLength(cnt[0], True), True).reshape(-1, 2).astype(float)
        sides[len(poly)] = sides.get(len(poly), 0) + 1
        if len(poly) == 6:
            out.append(poly)
    return out, sides


# ---------------------------------------------------------------------------------------------------------------------


def analyse_hexagon(img, corners, kind, scale):
    """One hexagon through the engine's calibrated pipeline: snap (refine, with its quality gate), noise-floor
    scores, source-pixel edge margin, "edges" ternary convention (plus the original "faces" one for comparison),
    and texture lock with both the source-pixel and the old fixed margin."""
    r = refine.refine_hexagon(img, corners, kind, scale=scale)
    # The starting corners here were placed deliberately roughly (and the froth ones come from the cells' inner
    # outlines), so the "moved far from the clicks" limit is relaxed; every outline was also checked by eye.
    r.qc_reasons = refine.quality(r, max_shift_frac=0.35)
    fit = transforms.fit_hexagon(r.points)
    s = metrics.compute_scores(fit, corner_sigma_px=r.sigma_px)
    margin = inference.source_margin_frac(r.wall_width_px, r.radius_px)
    warped = transforms.warp_to_canonical(img, fit, "affine")
    field = inference.orientation_field(warped)
    nk = {}
    for conv in inference.CONVENTIONS:
        for st in CubeState:
            faces = inference.analyse_faces(field, st, edge_margin_frac=margin, convention=conv)
            prim = inference.primary_line(faces, st)
            nk[f"{conv}:{st.name}"] = {"ternary": [f.ternary for f in faces],
                                      "coherence": [round(f.coherence, 3) for f in faces],
                                      "primary": None if prim is None else prim.face_name}
    lock = inference.texture_lock(field, margin)
    lock_fixed = inference.texture_lock(field, inference.DEFAULT_EDGE_MARGIN_FRAC)
    return {
        "qc_pass": r.qc_pass, "qc_reasons": r.qc_reasons,
        "points": np.round(r.points, 2).tolist(), "radius_px": round(r.radius_px, 1),
        "corner_sigma_px": round(r.sigma_px, 2), "side_rms_px": round(float(np.median(r.side_rms)), 2),
        "inlier_frac": round(float(r.inlier_frac.min()), 2), "wall_width_px": round(r.wall_width_px, 2),
        "edge_margin_frac": round(margin, 3),
        "affine": s.affine_mean, "homography": s.homography_mean, "similarity": s.similarity_mean,
        "cse": s.central_symmetry_error, "null_pct_affine": s.null_percentile["affine_mean"],
        "null_pct_cse": s.null_percentile["central_symmetry_error"],
        "noise_floor": s.noise_floor, "excess": s.excess, "verdict": s.verdict,
        "threshold_verdict": s.threshold_verdict,
        "necker": nk, "lock_p": lock.p_value, "lock_order": lock.order,
        "_lock": lock, "_lock_fixed": lock_fixed,
        "_fit": fit, "_faces": inference.analyse_faces(field, CubeState.CONVEX, edge_margin_frac=margin),
        "_warped": warped,
    }


def lock_shuffle_test(rows, n_shuffle: int = 5000):
    """Pooled rotation-shuffle texture-lock test, with the calibrated margin and, for comparison, the old one."""
    new = inference.pooled_lock_test([r["_lock"] for r in rows], n_shuffle)
    old = inference.pooled_lock_test([r["_lock_fixed"] for r in rows], n_shuffle)
    return {"source_margin": new, "fixed_margin": old}


def continuity_test(entries):
    """Edge continuity for neighbouring hexagons vs random non-neighbouring pairs in the same image."""
    fits = [e["_fit"] for e in entries]
    faces = [e["_faces"] for e in entries]
    edges = neighbours.find_shared_edges(fits, tol_px=8.0)
    if not edges:
        return None, []
    cont = neighbours.edge_continuity(fits, faces, CubeState.CONVEX, edges)
    nb = {(e.cell_a, e.cell_b) for e in edges}
    null = []
    for a in range(len(fits)):
        for b in range(a + 1, len(fits)):
            if (a, b) in nb:
                continue
            # pretend edge 0 of a meets edge 3 of b: compare the faces' image-space orientations the same way
            ta = neighbours._face_theta_in_image(fits[a], faces[a][0], "affine")
            tb = neighbours._face_theta_in_image(fits[b], faces[b][0], "affine")
            d = abs(ta - tb) % np.pi
            null.append(abs(np.cos(min(d, np.pi - d))))
    return {"n_edges": len(cont), "mean_score": float(np.mean([c.score for c in cont])),
            "null_mean": float(np.mean(null)) if null else None, "n_null": len(null),
            "relations": {k: sum(c.relation == k for c in cont) for k in ("same", "flipped", "unrelated")}}, cont


def main(img_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "overlays").mkdir(exist_ok=True)
    rows, per_image = [], {}
    for name, cfg in CONFIG.items():
        img = cv2.cvtColor(cv2.imread(str(img_dir / name)), cv2.COLOR_BGR2RGB)
        info = {"group": cfg["group"], "size": [img.shape[1], img.shape[0]], "n_cells": 0, "sides": {},
                "refine_failed": 0}
        candidates = []
        if "components" in cfg:
            candidates, info["sides"] = froth_cells(img, cfg["components"]["min_area"])
            info["n_cells"] = sum(info["sides"].values())
        else:
            candidates = [np.asarray(m, float) for m in cfg["manual"]]
            info["n_cells"] = len(candidates)
            info["sides"] = {6: len(candidates)}
        entries = []
        shown = img.copy()
        for k, corners in enumerate(candidates):
            try:
                e = analyse_hexagon(img, corners, cfg["kind"], cfg["scale"])
            except ValueError:
                info["refine_failed"] += 1
                continue
            e.update(image=name, group=cfg["group"], cell=k)
            entries.append(e)
            colour = (0, 230, 255) if e["qc_pass"] else (255, 60, 60)
            shown = render.draw_points(shown, np.array(e["points"]), labels=False, color=colour)
            c = np.array(e["points"]).mean(axis=0)
            cv2.putText(shown, f"{e['affine']:.3f}" if e["qc_pass"] else "QC", (int(c[0]) - 18, int(c[1]) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / "overlays" / name), cv2.cvtColor(shown, cv2.COLOR_RGB2BGR))
        passed = [e for e in entries if e["qc_pass"]]
        cont, _ = continuity_test(passed) if len(passed) >= 2 else (None, [])
        info["continuity"] = cont
        info["n_hexagons"] = len(entries)
        info["n_qc_pass"] = len(passed)
        per_image[name] = info
        rows.extend(entries)
        print(f"{name:38s} cells {info['n_cells']:3d} sides {dict(sorted(info['sides'].items()))} "
              f"hexagons {len(entries)} QC-pass {len(passed)}")

    rows_all, rows = rows, [r for r in rows if r["qc_pass"]]
    groups = sorted({r["group"] for r in rows})
    per_group = {g: lock_shuffle_test([r for r in rows if r["group"] == g]) for g in groups}

    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows_all]
    (out_dir / "results.json").write_text(json.dumps({"per_image": per_image, "per_group_lock": per_group,
                                                      "hexagons": clean}, indent=1, default=float))
    with open(out_dir / "hexagons.csv", "w", newline="", encoding="utf-8-sig") as fh:
        cols = ["group", "image", "cell", "qc_pass", "radius_px", "corner_sigma_px", "side_rms_px", "wall_width_px",
                "edge_margin_frac", "affine", "homography", "similarity", "cse", "noise_floor", "excess", "verdict",
                "threshold_verdict", "null_pct_affine", "null_pct_cse", "lock_order", "lock_p"]
        keys = [f"{c}:{s.name}" for c in inference.CONVENTIONS for s in CubeState]
        wr = csv.writer(fh)
        wr.writerow(cols + [f"{k} ternary" for k in keys] + [f"{k} primary" for k in keys])
        for r in clean:
            wr.writerow([r[c] if not isinstance(r[c], float) else round(r[c], 4) for c in cols] +
                        [r["necker"][k]["ternary"] for k in keys] + [r["necker"][k]["primary"] for k in keys])
    print(json.dumps(per_group, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
