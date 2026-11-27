"""Comb Cell Measure: a free tool for beekeepers to measure honeycomb cell size from a photo with a ruler.

Run locally:  streamlit run app/bee_app.py
Photos are processed in memory and never stored.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from streamlit_image_coordinates import streamlit_image_coordinates

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hexengine import comb, morphometry  # noqa: E402

MAX_SIDE = 2000          # photos are reduced to this long side: cells stay large, measuring stays quick
STEPS = ["ruler mark 1", "ruler mark 2", "one corner of the comb area", "the opposite corner of the comb area"]
COLOURS = [(255, 0, 255), (255, 0, 255), (60, 200, 60), (60, 200, 60)]

st.set_page_config(page_title="Comb Cell Measure", page_icon="🐝", layout="wide")
st.title("🐝 Comb Cell Measure")
st.write("Measure honeycomb cell size (mm, wall centre to wall centre) from one photo with a ruler lying on the "
         "comb. Free and open source. Your photo is processed in memory and is not stored.")

with st.expander("How to take a good photo", expanded=False):
    st.markdown(
        "1. **Ruler flat on the comb**, metric side up, across the cells you want to measure.\n"
        "2. **Straight on:** phone parallel to the comb, about 15–25 cm away (each cell at least ~50 pixels across).\n"
        "3. **Full-size photo:** not a copy sent through a messaging app, which compresses it.\n"
        "4. **Even light**, no harsh shadow across the ruler. Empty or backlit comb is easiest; capped brood works.\n"
        "5. Got a hand count (\"10 cells = 49 mm\")? Compare it with the result.")

up = st.file_uploader("Comb photo (JPG or PNG)", type=["jpg", "jpeg", "png"])
if up is None:
    st.info("Upload a photo to start.")
    st.stop()

key = f"{up.name}-{up.size}"
if st.session_state.get("photo_key") != key:
    pil = ImageOps.exif_transpose(Image.open(up)).convert("RGB")
    f = min(1.0, MAX_SIDE / max(pil.size))
    if f < 1:
        pil = pil.resize((round(pil.size[0] * f), round(pil.size[1] * f)), Image.LANCZOS)
    st.session_state.update(photo_key=key, img=np.array(pil), clicks=[], last=None, report=None, nonce=0)
img = st.session_state["img"]
clicks = st.session_state["clicks"]


def reset(n_keep: int = 0) -> None:
    del st.session_state["clicks"][n_keep:]
    st.session_state["report"] = None
    st.session_state["nonce"] += 1


left, right = st.columns([3, 2])
with right:
    st.subheader("1. Mark the photo")
    if len(clicks) < 4:
        st.info(f"Click **{STEPS[len(clicks)]}** on the photo.")
    else:
        st.success("Ruler and comb area marked.")
    st.caption("Ruler: click two marks far apart (e.g. 1 cm and 6 cm). Comb area: drag-free, just click two "
               "opposite corners of a box over clean cells, away from the ruler.")
    mm = st.number_input("Distance between your two ruler marks (mm)", min_value=1.0, max_value=1000.0,
                         value=50.0, step=1.0)
    walls = st.radio("Cell walls look…", ["lighter than the cells (most comb)", "darker than the cells (backlit)"])
    b1, b2 = st.columns(2)
    b1.button("Undo last click", on_click=reset, args=(max(0, len(clicks) - 1),), disabled=not clicks)
    b2.button("Start again", on_click=reset, disabled=not clicks)
    go = st.button("📏 Measure cells", type="primary", disabled=len(clicks) < 4, use_container_width=True)

with left:
    shown = img.copy()
    t = max(2, img.shape[1] // 500)
    for i, (x, y) in enumerate(clicks):
        cv2.circle(shown, (int(x), int(y)), 4 * t, COLOURS[i], -1, cv2.LINE_AA)
    if len(clicks) >= 2:
        cv2.line(shown, tuple(map(int, clicks[0])), tuple(map(int, clicks[1])), COLOURS[0], t, cv2.LINE_AA)
    if len(clicks) == 4:
        cv2.rectangle(shown, tuple(map(int, clicks[2])), tuple(map(int, clicks[3])), COLOURS[2], t)
    if st.session_state["report"] is not None and st.session_state["report"].overlay is not None:
        shown = st.session_state["report"].overlay
    v = streamlit_image_coordinates(shown, use_column_width="always", key=f"photo{st.session_state['nonce']}")
    if v and "x" in v and len(clicks) < 4:
        sig = (v["x"], v["y"], v.get("width"), v.get("height"))
        if st.session_state["last"] != sig:
            st.session_state["last"] = sig
            clicks.append((v["x"] * img.shape[1] / float(v["width"] or img.shape[1]),
                           v["y"] * img.shape[0] / float(v["height"] or img.shape[0])))
            st.rerun()

if go:
    (p1, p2, c1, c2) = clicks
    with st.spinner("Finding the comb pattern and measuring each cell (about 10–30 seconds)…"):
        try:
            st.session_state["report"] = comb.measure_comb(
                img, p1, p2, mm, (*c1, *c2), tick_strip=(p1, p2),
                wall_kind="bright_ridge" if walls.startswith("lighter") else "dark_ridge", max_cells=40)
        except ValueError as exc:
            st.session_state["report"] = None
            st.error(f"Couldn't measure this photo: {exc}")
    st.rerun()

rep = st.session_state["report"]
if rep is None:
    st.stop()

st.subheader("2. Results")
m = st.columns(3)
m[0].metric("Median cell width", "—" if rep.median_width_mm is None else f"{rep.median_width_mm:.2f} mm",
            help="Median over the cells that passed the quality check (blue outlines). Use this number.")
m[1].metric("Whole-comb spacing", f"{rep.lattice_width_mm:.2f} mm",
            help="The repeat distance of the whole comb pattern in your box: an independent second estimate.")
m[2].metric("Cells measured", f"{rep.cells_passed} of {rep.cells_tried}",
            help="Red outlines failed the quality check (blurred, damaged or covered walls) and are not counted.")
if rep.median_width_mm is not None:
    cls = rep.size_class
    if cls == "between classes":                 # in a gap between two ranges: name the nearest one
        lo_hi = [(min(abs(rep.median_width_mm - lo), abs(rep.median_width_mm - hi)), lab)
                 for lo, hi, lab in morphometry.BENCHMARKS]
        cls = "close to " + min(lo_hi)[1]
    st.markdown(f"**Size class:** {cls}")
    st.caption("Indicative Apis mellifera ranges: " + "; ".join(
        f"{lo:.2f}–{hi:.2f} mm {lab}" for lo, hi, lab in morphometry.BENCHMARKS))
if rep.quartiles_mm:
    st.caption(f"Middle half of the cells: {rep.quartiles_mm[0]:.2f}–{rep.quartiles_mm[1]:.2f} mm. Scale "
               f"{rep.px_per_mm:.2f} px/mm" + ("" if rep.scale_disagreement_pct is None else
                                              f"; ruler ticks agree within {abs(rep.scale_disagreement_pct):.1f} %")
               + ".")
for w in rep.warnings:
    st.warning(w)

ok = [c for c in rep.cells if c.qc_pass]
if len(ok) >= 3:
    fig, ax = plt.subplots(figsize=(6, 2.6))
    w = np.array([c.width_mm for c in ok])
    x0, x1 = np.floor((w.min() - 0.3) * 20) / 20, np.ceil((w.max() + 0.3) * 20) / 20
    for lo, hi, _ in morphometry.BENCHMARKS:
        ax.axvspan(lo, hi, color="#e1e0d9", alpha=0.6, zorder=0)
    ax.hist(w, bins=np.arange(x0, x1 + 0.025, 0.05), color="#2a78d6", edgecolor="#fcfcfb", zorder=2)
    ax.set_xlim(x0, x1)
    ax.set_xlabel("Cell width (mm)")
    ax.set_ylabel("Cells")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    st.caption("Grey bands: indicative size classes (natural/small-cell worker, standard worker, intermediate, drone).")

rows = [{"cell": i + 1, "passed_quality_check": c.qc_pass, "width_mm": round(c.width_mm, 3),
         "width_a_mm": round(c.widths_mm[0], 3), "width_b_mm": round(c.widths_mm[1], 3),
         "width_c_mm": round(c.widths_mm[2], 3), "anisotropy_mm": round(c.anisotropy_mm, 3),
         "size_class": c.size_class, "note": c.qc_reasons} for i, c in enumerate(rep.cells)]
d1, d2 = st.columns(2)
d1.download_button("Download cell table (CSV)", pd.DataFrame(rows).to_csv(index=False).encode("utf-8"),
                   file_name="comb_cells.csv", mime="text/csv")
buf = io.BytesIO()
Image.fromarray(rep.overlay).save(buf, format="PNG")
d2.download_button("Download marked-up photo (PNG)", buf.getvalue(), file_name="comb_cells.png", mime="image/png")
st.caption("Accuracy so far: within 0.06 mm of printed ground truth on two test photos (a pilot, not yet a full "
           "validation). Single cells scatter by about ±0.1 mm on small photos; the median is the reliable number.")
