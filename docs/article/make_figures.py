"""Figures for the closing article. Run from hex-engine:  .venv\\Scripts\\python.exe docs\\article\\make_figures.py

Charts use only the project's own results (results_v2, results_v3, results_v4) and synthetic samples, so every
figure in figures/ is the project's own work. Photos of third-party images are NOT generated here.
"""

import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hexengine import inference, refine, render, transforms  # noqa: E402
from hexengine.types import CubeState  # noqa: E402

RES = ROOT.parent / "hex-test-images"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

# Reference palette (light): slots 1-2 are a validated adjacent pair; chrome per palette.md.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10.5, "axes.edgecolor": AXIS,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.titleweight": "semibold", "axes.titlesize": 12.5, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def finish(ax, grid_axis="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


v2 = json.loads((RES / "results_v2" / "results.json").read_text())
v3 = json.loads((RES / "results_v3" / "results.json").read_text())
v4 = json.loads((RES / "results_v4" / "results.json").read_text())

# ---------------------------------------------------------------- 1. residual vs noise floor (results_v2)
groups = [("1 Tiles (control)", "Tumbling-block tile (control)"), ("6 Snowflake", "Snowflake plates"),
          ("2 Honeycomb", "Honeycomb"), ("3 Basalt", "Basalt columns"), ("5 Turtle", "Turtle scutes"),
          ("4 Bubbles", "Soap froth")]
fig, ax = plt.subplots(figsize=(8.2, 4.2))
for i, (g, label) in enumerate(groups):
    hs = [h for h in v2["hexagons"] if h["group"] == g and h["qc_pass"]]
    res, floor = np.median([h["affine"] for h in hs]), np.median([h["noise_floor"] for h in hs])
    y = len(groups) - 1 - i
    ax.plot([floor, res], [y, y], color=AXIS, linewidth=2, zorder=1)
    ax.scatter([floor], [y], s=70, color=MUTED, zorder=2, edgecolor=SURFACE, linewidth=2)
    ax.scatter([res], [y], s=80, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=2)
    ax.text(max(res, floor) + 0.006, y, f"{res:.3f}  (n = {len(hs)})", va="center", color=INK2, fontsize=9.5)
ax.set_yticks(range(len(groups)))
ax.set_yticklabels([l for _, l in groups][::-1])
ax.set_xlim(0, 0.24)
ax.set_xlabel("Median affine residual (fraction of the cell's radius)")
ax.set_title("Natural hexagons are real but irregular: far above measurement error", loc="left")
ax.scatter([], [], s=70, color=MUTED, label="Noise floor (marking error alone)")
ax.scatter([], [], s=80, color=BLUE, label="Measured residual")
ax.legend(loc="upper right", frameon=False)
finish(ax)
fig.tight_layout()
fig.savefig(OUT / "fig2-residual-vs-noise-floor.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- 2. honeycomb lead: batch 1 vs batch 2
b1 = [h for h in v2["hexagons"] if h["group"] == "2 Honeycomb" and h["qc_pass"]]
b2 = [h for h in v3["hexagons"] if h["group"] == "Honeycomb" and h["qc_pass"]]
def counts(hs):
    cvx = sum(h["necker"]["edges:CONVEX"]["primary"] is not None for h in hs)
    ccv = sum(h["necker"]["edges:CONCAVE"]["primary"] is not None for h in hs)
    return 100 * cvx / len(hs), 100 * ccv / len(hs), cvx, ccv, len(hs)
fig, ax = plt.subplots(figsize=(7.4, 4.0))
labels = []
for i, (name, hs) in enumerate([("Batch 1\n1 photo", b1), ("Batch 2\n4 photos", b2)]):
    pc, pk, c, k, n = counts(hs)
    ax.bar(i - 0.19, pc, 0.36, color=BLUE, label="Convex primary line" if i == 0 else None)
    ax.bar(i + 0.19, pk, 0.36, color=ORANGE, label="Concave primary line" if i == 0 else None)
    ax.text(i - 0.19, pc + 2, f"{c}/{n}", ha="center", color=INK2, fontsize=9.5)
    ax.text(i + 0.19, pk + 2, f"{k}/{n}", ha="center", color=INK2, fontsize=9.5)
    labels.append(name)
ax.set_xticks([0, 1])
ax.set_xticklabels(labels)
ax.set_ylabel("Share of cells (%)")
ax.set_ylim(0, 112)
ax.set_yticks(range(0, 101, 20))
a = v3["hypothesis_a"]
ax.set_title("The honeycomb 'Concave' lead did not replicate", loc="left")
ax.text(1.0, 70, f"Batch 2: sign test p = {a['sign_test_p']:.2f}\nrotated control p = {a['rotated_control_p']:.2f}",
        ha="center", va="top", color=INK2, fontsize=9.5)
ax.legend(loc="upper left", ncol=2, frameon=False)
finish(ax, "y")
fig.tight_layout()
fig.savefig(OUT / "fig3-honeycomb-lead-replication.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- 3. outline artefact removed (texture lock)
rows = [("Soap froth", v2["per_group_lock"]["4 Bubbles"]),
        ("Honeycomb", v2["per_group_lock"]["2 Honeycomb"]),
        ("Snowflake plates", v2["per_group_lock"]["6 Snowflake"])]
fig, ax = plt.subplots(figsize=(7.8, 3.4))
for i, (label, d) in enumerate(rows):
    y = len(rows) - 1 - i
    pf, ps = d["fixed_margin"]["p"], d["source_margin"]["p"]
    ax.plot([pf, ps], [y, y], color=AXIS, linewidth=2, zorder=1)
    ax.scatter([pf], [y], s=75, color=ORANGE, zorder=3, edgecolor=SURFACE, linewidth=2)
    ax.scatter([ps], [y], s=75, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=2)
    if abs(pf - ps) < 1e-9:
        ax.text(ps / 1.4, y, "unchanged: a real lock", va="center", ha="right", color=INK2, fontsize=9)
ax.axvline(0.01, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
ax.text(0.0105, len(rows) - 0.55, "locked if p < 0.01", color=INK2, fontsize=9)
ax.set_xscale("log")
ax.set_xlim(1e-4, 1.5)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows][::-1])
ax.set_ylim(-0.6, len(rows) - 0.3)
ax.set_xlabel("Texture-lock shuffle p-value (log scale)")
ax.set_title("A false signal removed: the cell's own outline was being read as texture", loc="left")
ax.scatter([], [], s=75, color=ORANGE, label="Original margin (fixed share of frame)")
ax.scatter([], [], s=75, color=BLUE, label="Calibrated margin (wall + 8 photo px)")
ax.legend(loc="lower right", frameon=False, fontsize=9)
finish(ax)
fig.tight_layout()
fig.savefig(OUT / "fig4-artefact-removed.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- 4. pilot accuracy (results_v4)
order = ["1-control-natural-comb-4p8mm.jpg", "2-control-foundation-5p3mm.jpg", "3-field-brood-steel-scale.jpg",
         "4-field-brood-yellow-tape.jpg", "5-field-golden-comb.jpg"]
names = ["Control 1: natural comb", "Control 2: wax foundation", "Field 3: brood comb",
         "Field 4: brood comb", "Field 5: golden comb"]
fig, ax = plt.subplots(figsize=(8.2, 4.0))
ax.axvspan(5.20, 5.40, color="#e1e0d9", alpha=0.6, zorder=0)
ax.text(5.30, 4.62, "typical commercial\nworker cell 5.2–5.4 mm", ha="center", va="top", color=INK2, fontsize=8.5)
for i, (k, label) in enumerate(zip(order, names)):
    d = v4["per_image"][k]
    y = len(order) - 1 - i
    if d["median_width_mm"] is not None:
        ax.scatter([d["median_width_mm"]], [y + 0.12], s=75, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=2)
    ax.scatter([d["lattice_width_mm"]], [y - 0.12], s=75, color=ORANGE, zorder=3, edgecolor=SURFACE, linewidth=2)
    if d["ground_truth_mm"]:
        ax.plot([d["ground_truth_mm"]] * 2, [y - 0.32, y + 0.32], color=INK, linewidth=2.2, zorder=4)
        err = d.get("error_cells_mm") if d.get("error_cells_mm") is not None else d.get("error_lattice_mm")
        ax.text(d["ground_truth_mm"] + 0.03, y + 0.30, f"printed {d['ground_truth_mm']:.2f} mm (error {err:+.3f})",
                color=INK, fontsize=8.5, va="center")
ax.set_yticks(range(len(order)))
ax.set_yticklabels(names[::-1])
ax.set_xlim(4.6, 5.65)
ax.set_ylim(-0.6, 4.7)
ax.set_xlabel("Cell width, wall centre to wall centre (mm)")
ax.set_title("Pilot: honeycomb cell width measured from a ruler in the photo", loc="left")
ax.scatter([], [], s=75, color=BLUE, label="Median of snapped cells")
ax.scatter([], [], s=75, color=ORANGE, label="Lattice spacing (whole comb)")
ax.plot([], [], color=INK, linewidth=2.2, label="Ground truth printed on the photo")
ax.legend(loc="lower right", frameon=False, fontsize=9)
finish(ax)
fig.tight_layout()
fig.savefig(OUT / "fig5-pilot-accuracy.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- 1. method, on the project's own synthetic sample
gt = json.loads((ROOT / "samples" / "ground_truth.json").read_text())
img = cv2.cvtColor(cv2.imread(str(ROOT / "samples" / "shaded_cube.png")), cv2.COLOR_BGR2RGB)
true = np.array(gt["shaded_cube.png"]["vertices"], float)
rough = true + np.random.default_rng(4).uniform(-14, 14, true.shape)
r = refine.refine_auto(img, rough)
fit = transforms.fit_hexagon(r.points)
w = transforms.warp_to_canonical(img, fit)
fld = inference.orientation_field(w)
faces = inference.analyse_faces(fld, CubeState.CONCAVE)
panels = {"1. rough clicks (cyan)": render.draw_points(img, rough, color=(0, 230, 255)),
          "2. snapped to the walls": render.draw_points(img, r.points, color=(255, 200, 40)),
          "3. normalised + cube edges": render.draw_construction(w, CubeState.CONVEX),
          "4. ternary reading": render.draw_ternary(w, faces, inference.primary_line(faces, CubeState.CONCAVE))}
sheet = render.comparison_sheet(panels, title="How the engine reads one cell (synthetic test image)", tile=320)
cv2.imwrite(str(OUT / "fig1-method.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
print(sorted(p.name for p in OUT.glob("*.png")))
