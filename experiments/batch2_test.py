"""Batch 2: targeted replication of the two leads from results_v2, plus the calibration controls.

Run:  .venv\\Scripts\\python.exe experiments\\batch2_test.py <batch2 image folder> <output folder> <results_v2 folder>

Engine settings are the calibrated ones from results_v2 (8 px source-pixel margin, 0.25 px minimum corner uncertainty,
2.5 % quality gate, "edges" ternary convention); the per-hexagon pipeline is baseline_test.analyse_hexagon unchanged.

Cells
-----
Regular patterns (honeycomb, tumbling blocks) are placed on their repeat lattice (lattice.py) from one anchor
picked by eye, which is first nudged to the position where the rough cell outlines sit best on the walls. Every
cell is then snapped onto the walls by hexengine.refine and must pass the quality gate. Snowflakes are marked by
hand (rough corners below). Snowflake g is a fern-like dendrite with no hexagonal outline and is not mapped.

Hypothesis A (honeycomb Concave lead): per cell, does a primary line appear under Concave, under Convex, both or
neither? Test 1 is an exact sign test on the cells that differ (Concave-only vs Convex-only). Test 2 is a rotated
control: each cell's normalised content is rotated by angles that are not multiples of 60 deg, which breaks any
alignment with the cell's own geometry; the observed (Concave - Convex) primary difference is compared with that
null distribution. Both must reach p < 0.05, and the lean must appear in more than one image, to count.

Hypothesis B (concentric "across-edges" signature): per crystal hexagon, signature = at least 2 of 3 regions at -1
under Convex AND at least 2 of 3 at +1 under Concave. Texture lock: per-cell rotation-shuffle p, and the pooled
shuffle test across independent crystals (Batch 2 alone, and Batch 1 + Batch 2). Lock needs p < 0.01.

Calibration: tumbling-block cubes should read +1, +1, +1 under Convex and sit at or near their noise floor.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from hexengine import canonical, inference, render, refine, transforms  # noqa: E402
from hexengine.types import CubeState  # noqa: E402
import baseline_test as bt  # noqa: E402
import lattice  # noqa: E402

RNG = np.random.default_rng(2611)

LATTICE = {  # name: group, wall kind, refine scale, anchor (a cell centre picked by eye), cube pattern?
    "2-honeycomb-e-dreamstime-457945902.jpg": ("Honeycomb", "bright_ridge", 3.0, (325, 225), False),
    "2-honeycomb-f-dreamstime-26991231.jpg": ("Honeycomb", "bright_ridge", 3.0, (310, 240), False),
    "2-honeycomb-g-dreamstime-368597532.jpg": ("Honeycomb", "bright_ridge", 3.0, (308, 270), False),
    "2-honeycomb-h-shutterstock-1377104081.jpg": ("Honeycomb", "bright_ridge", 3.0, (735, 470), False),
    "1-tiles-h-etsy-end-grain-board.jpg": ("Tiles (control)", "edge", 2.0, (1337, 548), True),
    "1-tiles-f-dreamstime-vector-471749722.jpg": ("Tiles (control)", "edge", 1.0, (370, 320), True),
    "1-tiles-e-wooden-top-chopping-board.jpg": ("Tiles (control)", "edge", 1.5, (750, 600), True),
    "1-tiles-g-teresa-quilt-banner.jpg": ("Tiles (control)", "edge", 1.5, (500, 240), True),
}
# Snowflakes use kind "auto": refine.refine_auto picks the wall type and scale (best trusted fit, closest to the
# marks), so no per-image choice is made by hand. Plate outlines are thin lines, unlike comb walls.
MANUAL = {  # name: group, kind, scale, [rough hexagons]
    "6-snowflake-e-kljatov-quiet-hexagon.jpg": ("Snowflake", "auto", None, [
        [(597, 618), (970, 693), (1066, 970), (821, 1290), (469, 1237), (384, 885)],      # inner plate
        [(565, 416), (1098, 512), (1280, 1066), (906, 1503), (309, 1375), (139, 842)]]),  # outer plate
    "6-snowflake-f-fortify-stellar-plate.jpg": ("Snowflake", "auto", None, [
        [(265, 207), (327, 243), (327, 327), (265, 363), (203, 327), (203, 243)]]),
    "6-snowflake-h-kljatov-facebook.jpg": ("Snowflake", "auto", None, [
        [(305, 362), (345, 385), (345, 430), (305, 452), (265, 430), (265, 385)]]),
}
NOT_MAPPABLE = {"6-snowflake-g-kljatov-built-by-laws.jpg": "fern-like dendrite: no hexagonal plate outline"}
MAX_CELLS = 25
ROTATIONS_DEG = [7.5, 15, 22.5, 30, 37.5, 45, 52.5]   # not multiples of 60: geometry-breaking control rotations


# ---------------------------------------------------------------------------------------------------- placement

def edge_score(fmap, corners_list) -> float:
    """Mean wall/edge response sampled along the sides of rough hexagons."""
    vals = []
    for c in corners_list:
        for k in range(6):
            a, b = c[k], c[(k + 1) % 6]
            ts = np.linspace(0.15, 0.85, 12)[:, None]
            pts = (a + ts * (b - a)).astype(np.float32)
            vals.append(cv2.remap(fmap, pts[:, :1], pts[:, 1:], cv2.INTER_LINEAR).ravel())
    return float(np.mean(np.concatenate(vals))) if vals else 0.0


def place_anchor(img, basis, guess, kind, scale, cube: bool):
    """Nudge the anchor to where the outlines of the nearest cells sit best on the walls.

    Search radius is 0.15 of the repeat length for cube patterns (to stay on the same kind of junction: a cube's
    own near corner, not the corner where three cubes meet) and 0.35 for honeycomb (any cell centre will do)."""
    fmap = refine.feature_map(img, kind, scale)
    rep = float(min(np.hypot(*basis[0]), np.hypot(*basis[1])))
    rad = (0.15 if cube else 0.35) * rep
    best, best_s = np.asarray(guess, float), -1.0
    for dx in np.linspace(-rad, rad, 15):
        for dy in np.linspace(-rad, rad, 15):
            c = np.asarray(guess, float) + [dx, dy]
            cells = [lattice.cell_corners(c + i * basis[0] + j * basis[1], basis)
                     for i in (-1, 0, 1) for j in (-1, 0, 1)]
            s = edge_score(fmap, cells)
            if s > best_s:
                best, best_s = c, s
    return best


def top_face_contrast(grey, corners) -> float:
    """Brightness of the rhombus above the centre (C, V5, V0, V1) minus the two rhombi below it."""
    corners, _ = transforms.order_points(corners)          # V0 = top, clockwise, as the engine labels them
    c = corners.mean(axis=0)
    regions = [[c, corners[5], corners[0], corners[1]], [c, corners[1], corners[2], corners[3]],
               [c, corners[3], corners[4], corners[5]]]
    means = []
    for poly in regions:
        m = np.zeros(grey.shape, np.uint8)
        cv2.fillPoly(m, [np.round(np.array(poly)).astype(np.int32)], 1)
        m = cv2.erode(m, np.ones((5, 5), np.uint8))
        means.append(float(grey[m > 0].mean()) if m.any() else 0.0)
    return means[0] - 0.5 * (means[1] + means[2])


def cube_anchor(img, basis, anchor):
    """Of the two kinds of three-way junction in a tumbling-block pattern, keep the cube's own near corner.

    Both kinds sit on edges equally well (that is the Necker ambiguity of the pattern itself), so the lattice
    alone cannot choose. The convention here, as in the original tile control, is a cube seen from above: its
    lit top face is the diamond directly above the near corner. Candidates are the anchor and the six corners of
    its cell; the one whose upper diamond is brightest relative to the two below, summed over nearby cells, wins.
    """
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    cands = [np.asarray(anchor, float)] + list(lattice.cell_corners(anchor, basis))
    def score(c):
        cells = [lattice.cell_corners(c + i * basis[0] + j * basis[1], basis) for i in (-1, 0, 1) for j in (-1, 0, 1)]
        h, w = grey.shape
        cells = [x for x in cells if (x >= 0).all() and (x[:, 0] < w).all() and (x[:, 1] < h).all()]
        return np.mean([top_face_contrast(grey, x) for x in cells]) if cells else -1e9
    return max(cands, key=score)


def candidates(name, img):
    if name in MANUAL:
        group, kind, scale, hexes = MANUAL[name]
        return group, kind, scale, [np.asarray(h, float) for h in hexes], None
    group, kind, scale, anchor, cube = LATTICE[name]
    max_period = max(img.shape[:2]) / 3 if "quilt" in name else None
    basis = lattice.lattice_vectors(img, max_period_px=max_period)
    if anchor is None:
        return group, kind, scale, [], basis
    anchor = place_anchor(img, basis, anchor, kind, scale, cube)
    if cube:
        anchor = place_anchor(img, basis, cube_anchor(img, basis, anchor), kind, scale, cube)
    return group, kind, scale, lattice.lattice_cells(anchor, basis, img.shape, max_cells=MAX_CELLS), basis


# ---------------------------------------------------------------------------------------------------- controls

def rotated_primaries(warped, margin_frac):
    """Primary line present (concave, convex) for the cell content rotated by each control angle."""
    size = warped.shape[0]
    mask = canonical.hexagon_mask(size)
    out = []
    for deg in ROTATIONS_DEG:
        m = cv2.getRotationMatrix2D((size / 2, size / 2), deg, 1.0)
        rot = cv2.warpAffine(np.ascontiguousarray(warped), m, (size, size), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
        rot[~mask] = 30
        fld = inference.orientation_field(rot)
        res = {}
        for st in CubeState:
            faces = inference.analyse_faces(fld, st, edge_margin_frac=margin_frac)
            res[st.name] = inference.primary_line(faces, st) is not None
        out.append((res["CONCAVE"], res["CONVEX"]))
    return out


def hypothesis_a(cells, n_null: int = 5000):
    con = np.array([c["necker"]["edges:CONCAVE"]["primary"] is not None for c in cells])
    cvx = np.array([c["necker"]["edges:CONVEX"]["primary"] is not None for c in cells])
    only_con, only_cvx = int(np.sum(con & ~cvx)), int(np.sum(cvx & ~con))
    sign_p = binomtest(only_con, only_con + only_cvx, 0.5, alternative="greater").pvalue \
        if only_con + only_cvx else 1.0
    obs = float(np.mean(con) - np.mean(cvx))
    rot = [c["_rot"] for c in cells]                     # per cell: list of (concave, convex) under rotation
    null = np.empty(n_null)
    for k in range(n_null):
        pick = [r[RNG.integers(len(r))] for r in rot]
        null[k] = np.mean([p[0] for p in pick]) - np.mean([p[1] for p in pick])
    rot_p = float((np.sum(null >= obs) + 1) / (n_null + 1))
    per_image = {}
    for c in cells:
        d = per_image.setdefault(c["image"], [0, 0, 0])
        d[0] += 1
        d[1] += c["necker"]["edges:CONCAVE"]["primary"] is not None
        d[2] += c["necker"]["edges:CONVEX"]["primary"] is not None
    images_leaning = sum(1 for n, a, b in per_image.values() if a > b)
    return {"n_cells": len(cells), "concave_primary": int(con.sum()), "convex_primary": int(cvx.sum()),
            "concave_only": only_con, "convex_only": only_cvx, "sign_test_p": float(sign_p),
            "difference": obs, "rotated_null_mean": float(null.mean()), "rotated_control_p": rot_p,
            "per_image": {k: {"cells": v[0], "concave": v[1], "convex": v[2]} for k, v in per_image.items()},
            "images_leaning_concave": images_leaning, "images": len(per_image),
            "replicates": bool(sign_p < 0.05 and rot_p < 0.05 and images_leaning >= 2)}


def signature(c) -> bool:
    return (c["necker"]["edges:CONVEX"]["ternary"].count(-1) >= 2
            and c["necker"]["edges:CONCAVE"]["ternary"].count(1) >= 2)


# ---------------------------------------------------------------------------------------------------- main

def main(img_dir: Path, out_dir: Path, v2_dir: Path):
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)
    rows, per_image = [], {}
    for name in sorted(list(LATTICE) + list(MANUAL) + list(NOT_MAPPABLE)):
        if name in NOT_MAPPABLE:
            per_image[name] = {"group": "Snowflake", "n_tried": 0, "n_pass": 0, "note": NOT_MAPPABLE[name]}
            print(f"{name:46s} not mapped: {NOT_MAPPABLE[name]}")
            continue
        img = cv2.cvtColor(cv2.imread(str(img_dir / name)), cv2.COLOR_BGR2RGB)
        try:
            group, kind, scale, cands, basis = candidates(name, img)
        except ValueError as exc:
            per_image[name] = {"group": LATTICE[name][0], "n_tried": 0, "n_pass": 0, "note": str(exc)}
            print(f"{name:46s} lattice not found: {exc}")
            continue
        if not cands:
            per_image[name] = {"group": group, "n_tried": 0, "n_pass": 0,
                               "note": "no anchor (photographed at a steep angle / no reliable cube centre)"}
            print(f"{name:46s} skipped: {per_image[name]['note']}")
            continue
        shown, entries = img.copy(), []
        for k, corners in enumerate(cands):
            try:
                k_kind, k_scale = kind, scale
                if kind == "auto":
                    pick = refine.refine_auto(img, corners, scales=(1.5, 3.0, 5.0))
                    k_kind, k_scale = pick.kind, pick.scale
                e = bt.analyse_hexagon(img, corners, k_kind, k_scale)
            except ValueError:
                continue
            e.update(image=name, group=group, cell=k)
            if e["qc_pass"] and group == "Honeycomb":
                e["_rot"] = rotated_primaries(e["_warped"], e["edge_margin_frac"])
            entries.append(e)
            colour = (0, 230, 255) if e["qc_pass"] else (255, 60, 60)
            shown = render.draw_points(shown, np.array(e["points"]), labels=False, color=colour)
        cv2.imwrite(str(out_dir / "overlays" / name), cv2.cvtColor(shown, cv2.COLOR_RGB2BGR))
        passed = [e for e in entries if e["qc_pass"]]
        per_image[name] = {"group": group, "n_tried": len(cands), "n_pass": len(passed),
                           "lattice": None if basis is None else np.round(basis, 1).tolist()}
        rows.extend(entries)
        print(f"{name:46s} tried {len(cands):3d}  QC-pass {len(passed):3d}")

    ok = [r for r in rows if r["qc_pass"]]
    honey = [r for r in ok if r["group"] == "Honeycomb"]
    crystals = [r for r in ok if r["group"] == "Snowflake"]
    tiles = [r for r in ok if r["group"] == "Tiles (control)"]

    hyp_a = hypothesis_a(honey) if honey else None

    v2 = json.loads((v2_dir / "results.json").read_text())
    v2_ok = [h for h in v2["hexagons"] if h["qc_pass"]]
    v2_crystal_locks = [bt.inference.TextureLock(order=h["lock_order"], phase_deg=0.0, offset_deg=0.0,
                                                 p_value=h["lock_p"], locked=False)
                        for h in v2_ok if h["group"] in ("6 Snowflake", "5 Turtle")]
    # Batch-1 phases are not stored, so the combined pooled test re-derives them from the saved per-cell p:
    # p = offset/15 deg and offset = |12 psi| / 12, which fixes cos(12 psi) = cos(pi * p).
    for lk, h in zip(v2_crystal_locks, [h for h in v2_ok if h["group"] in ("6 Snowflake", "5 Turtle")]):
        lk.phase_deg = float(np.degrees(np.pi * h["lock_p"]) / 12.0)
    b2_locks = [r["_lock"] for r in crystals]
    hyp_b = {
        "n_hexagons": len(crystals), "n_crystals": len({r["image"] for r in crystals}),
        "signature": [{"image": r["image"], "cell": r["cell"], "convex": r["necker"]["edges:CONVEX"]["ternary"],
                       "concave": r["necker"]["edges:CONCAVE"]["ternary"], "signature": signature(r),
                       "lock_order": r["lock_order"], "lock_p": r["lock_p"]} for r in crystals],
        "pooled_lock_batch2": inference.pooled_lock_test(b2_locks, 5000),
        "pooled_lock_one_per_crystal": inference.pooled_lock_test(
            [max((r for r in crystals if r["image"] == im), key=lambda r: r["lock_order"])["_lock"]
             for im in sorted({r["image"] for r in crystals})], 5000),
        "pooled_lock_batch1_plus_2": inference.pooled_lock_test(b2_locks + v2_crystal_locks, 5000),
    }
    calib = {"n": len(tiles),
             "convex_all_plus_one": sum(r["necker"]["edges:CONVEX"]["ternary"] == [1, 1, 1] for r in tiles),
             "regular": sum(r["verdict"] == "regular" for r in tiles),
             "per_image": {im: {"cells": sum(r["image"] == im for r in tiles),
                                "convex_+1+1+1": sum(r["image"] == im and r["necker"]["edges:CONVEX"]["ternary"]
                                                     == [1, 1, 1] for r in tiles),
                                "regular": sum(r["image"] == im and r["verdict"] == "regular" for r in tiles)}
                           for im in sorted({r["image"] for r in tiles})}}

    def group_summary(rs, all_rs):
        if not rs:
            return {"mapped": 0, "tried": len(all_rs)}
        f = lambda k: float(np.median([r[k] for r in rs]))
        lock = inference.pooled_lock_test([r["_lock"] if "_lock" in r else
                                           bt.inference.TextureLock(r["lock_order"], 0, 0, r["lock_p"], False)
                                           for r in rs], 5000)
        return {"images": len({r["image"] for r in all_rs}), "mapped": len(rs), "tried": len(all_rs),
                "affine": f("affine"), "homography": f("homography"), "noise_floor": f("noise_floor"),
                "regular": sum(r["verdict"] == "regular" for r in rs),
                "lock_L": None if lock is None else lock["L"], "lock_p": None if lock is None else lock["p"],
                "convex_primary": sum(r["necker"]["edges:CONVEX"]["primary"] is not None for r in rs),
                "concave_primary": sum(r["necker"]["edges:CONCAVE"]["primary"] is not None for r in rs)}

    summary = {g: group_summary([r for r in ok if r["group"] == g], [r for r in rows if r["group"] == g])
               for g in ("Honeycomb", "Snowflake", "Tiles (control)")}
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    (out_dir / "results.json").write_text(json.dumps({"per_image": per_image, "summary": summary,
                                                      "hypothesis_a": hyp_a, "hypothesis_b": hyp_b,
                                                      "calibration": calib, "hexagons": clean},
                                                     indent=1, default=float))
    with open(out_dir / "hexagons_batch2.csv", "w", newline="", encoding="utf-8-sig") as fh:
        cols = ["group", "image", "cell", "qc_pass", "radius_px", "corner_sigma_px", "wall_width_px",
                "edge_margin_frac", "affine", "homography", "similarity", "cse", "noise_floor", "excess",
                "verdict", "threshold_verdict", "null_pct_affine", "lock_order", "lock_p"]
        keys = [f"edges:{s.name}" for s in CubeState]
        wr = csv.writer(fh)
        wr.writerow(cols + [f"{k} ternary" for k in keys] + [f"{k} primary" for k in keys])
        for r in clean:
            wr.writerow([r[c] if not isinstance(r[c], float) else round(r[c], 4) for c in cols] +
                        [r["necker"][k]["ternary"] for k in keys] + [r["necker"][k]["primary"] for k in keys])
    print(json.dumps({"summary": summary, "hypothesis_a": hyp_a, "calibration": calib,
                      "hypothesis_b": {k: v for k, v in hyp_b.items() if k != "signature"}}, indent=1, default=float))
    for s in hyp_b["signature"]:
        print("  crystal", s)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
