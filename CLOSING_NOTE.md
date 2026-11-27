# Closing note: Hexagonal Geometric Test Engine

Clare Rudling, with Ben · engine built with Claude (Anthropic) · 24 November 2026
Code: `hex-engine/` (git history from 24 Nov 2026). Data and results: `hex-test-images/results*`.

## Part 1. The cube / ternary hypothesis: tested and not supported

**Hypothesis.** A roughly hexagonal structure in an image can be normalised onto a perfect hexagon and read as an
isometric cube. Its internal texture then carries a Necker orientation (convex or concave), ternary states
(+1 / 0 / −1) along the cube's edges, and directional lines that continue across neighbouring cells.

**Method.** Six-point normalisation (affine, homography and similarity fits); corners snapped to the walls to
sub-pixel precision; a per-cell noise floor (the residual a perfect hexagon of the same size would show from
corner error alone); a quality gate; ternary states calibrated on a tumbling-block control; and shuffle and
rotation controls. Every lead had to replicate in an independent second image batch.

**Results.**

| Claim | Evidence | Verdict |
|---|---|---|
| Natural hexagons are projected regular hexagons (cube silhouettes) | Every natural group sits far above its noise floor; central symmetry fails as well (honeycomb, basalt, foam, scutes) | Not supported: they are irregular six-sided cells |
| A preferred Necker orientation exists | Batch 1: 8 of 9 honeycomb cells Concave (one photo). Batch 2: 10 Concave vs 16 Convex in 48 cells from 4 photos; sign test p = 0.93, rotated control p = 0.91 | Not replicated |
| Concentric "across-edges" ternary signature | Batch 1: 4 of 4 cells. Batch 2: 0 of 2 plates | Not replicated (the sign is also a fragile proxy for concentric rings) |
| Directional lines continue across shared walls | Honeycomb p = 0.58, foam p = 0.56 (after removing an outline-leakage artefact) | Not supported |
| Internal texture locks to the hexagon's own axes | Snowflake plates: L = 1.00, p = 0.0006 (Batch 2); all crystal cells L = 0.81, p = 0.0016 | Replicated for crystal plates only, as crystal growth predicts |

**Conclusion.** For natural hexagons the cube / ternary reading is falsified on these data. The only surviving
signal is snowflake plates' internal ridges sitting on the plate's own six axes, which is expected from crystal
physics. The cube layers are kept in the software behind an "unconfirmed" switch, for reference.

**Method lessons worth keeping.** (1) A fixed edge margin let small cells read their own outline as texture: the
margin must be set in the photo's pixels (wall width + 8 px). (2) The original ternary rule credited a real cube's
edges to the opposite Necker reading; calibrating on a known control caught it. (3) A single-photo lean (8 of 9)
vanished at 48 cells: replicate before interpreting.

## Part 2. What the pipeline is good for: calibrated honeycomb cell measurement

**Pipeline.** Ruler calibration (two points + known distance, cross-checked against the ruler's millimetre-tick
period) → the comb's repeat lattice → cells snapped to the wall centrelines → per-cell flat-to-flat width (mean of the
three opposite-wall distances, wall centre to wall centre, the quantity a "10 cells = 48 mm" ruler count
measures), anisotropy, irregularity above the noise floor and size class. A second, independent estimate is the
lattice spacing over the whole comb.

**Pilot (results_v4), five ruler photographs:**

| Image | Ground truth | Cells passed | Median cell width (IQR) | Lattice spacing | Error |
|---|---|---|---|---|---|
| Control 1: natural dark comb | 4.80 mm (printed, 48 mm / 10 cells) | 26 / 30 | 4.806 mm (4.73–4.87) | 4.782 mm | +0.006 mm (cells), −0.018 mm (lattice) |
| Control 2: wax foundation | 5.30 mm (printed, 106 mm / 20 cells) | no walls to snap | not measured per cell | 5.242 mm (row repeat) | −0.058 mm (−1.1 %) |
| Field 3: brood comb, steel scale | none | 20 / 25 | 4.927 mm (4.84–5.03) | 4.844 mm | none |
| Field 4: brood comb, yellow tape | none | 30 / 30 | 5.005 mm (4.91–5.06) | 4.940 mm | none |
| Field 5: golden comb | none | 5 / 10 | 4.865 mm (4.80–4.98) | 4.733 mm | none |

The two-point ruler scales agree with the measured tick period within 0.2–3 %. All three field combs measure
4.7–5.0 mm, in the natural / small-cell worker range.

**Limitations.** (1) All five images are web thumbnails (207–468 px wide), so cells are only 14–23 px in radius.
Single-cell widths scatter by about ±0.1 mm, and anisotropy (median 0.3–0.55 mm) is mostly pixel noise; the
medians and lattice spacings are the reliable numbers. (2) Calibrate on the ruler lying on the comb, and check it against something in the
image. In Control 1 the upper scale belongs to a different ruler (or uses different markings) and gives cells
2.15× too large; only the lower scale, confirmed by the photo's own "48 mm" arrow, gives 4.8 mm. (A ruler resting
slightly above the comb adds a small perspective error too, but nothing like 2×.) (3) Foundation has embossed bases, not walls, so only its repeat can be measured. (4) Two controls is a
pilot, not a validation study. (5) The size classes are indicative ranges supplied for this pilot; cells near a
boundary (Control 1 sits at 4.80 mm) split between neighbouring classes.

**Recommendation.** As a morphometry tool it is promising: two independent methods agree with the printed ground
truths to within 0.06 mm. A proper validation would use 10–20 full-resolution photographs (cells at least 50 px
across) with the ruler on the comb and hand-counted ground truth for each, including drone and foundation comb.

*Reproduce:* `experiments/baseline_test.py` (results, results_v2), `experiments/batch2_test.py` (results_v3),
`experiments/honeycomb_pilot.py` (results_v4). The findings report covers the full analysis.

## Status (25 November 2026)

Closing article published on Substack and other media: [https://clare539454.substack.com/p/what-a-hexagon-can-and-cant-tell](https://clare539454.substack.com/p/what-a-hexagon-can-and-cant-tell)
(source: `docs/article/article.md`). Reviewed by Grok and Gemini. Project paused until a new
use case or test comes up.
