"""'Help' tab: what each control does and what the pipeline computes, including where definitions are provisional."""

from __future__ import annotations

import streamlit as st

HELP = r"""
### What this is
A test bed for one question: **how close is a marked hexagon to a projected regular hexagon (an isometric cube
silhouette), and does its internal structure read as a cube?** Every number comes from the `hexengine` package, so
the same values can be reproduced in a notebook.

### The pipeline
1. **Six points.** Click the six vertices in order around the hexagon. You can start anywhere and go either way: the
   engine makes the order clockwise and puts the topmost point first. *Rotation offset* then chooses which point
   becomes V0 (each step turns the normalised image by 60°). It is remembered per image and saved with its points.
2. **Snap to walls** (on by default). A straight line is fitted along each wall, using every pixel of the side, and
   the corners become the intersections, to a fraction of a pixel. Your clicks only need to be roughly right. It
   also measures each corner's uncertainty and the wall's width. If the walls can't be found cleanly (blurred,
   curved, or the fit wanders off to another line) your clicks are used and the app says why. *Wall type* Auto tries
   dark lines, bright lines and tone edges.
3. **Normalisation.** Three least-squares fits map the corners onto the perfect hexagon (circumradius 1):
   *similarity* (rotation + scale, 4 dof), *affine* (+ shear, 6 dof) and *homography* (+ perspective, 8 dof).
   The *Warp transform* setting chooses which one warps the image into the 512 px canonical frame.
4. **Scores** (canonical units, so independent of image size):
   * **Mean vertex residual**: for the selected transform. 0 = an exact projection of a regular hexagon.
   * **Noise floor**: what a *perfect* hexagon of the same size would score with corners off by the measured (or,
     without snapping, an assumed ±1.5 px) error. Only the *excess* above it is real shape information.
   * **Central-symmetry error**: 0 = opposite sides parallel and equal. Every parallel view of a cube has this, even
     when it is not an affine image of a regular hexagon.
   * **Null percentile**: the share of 1,000 random hexagons that score *better*.
   * **Shape verdict**: *regular* (within the noise floor), *irregular* (above it: a real but distorted hexagon) or
     *random-like* (no better than 5 % of random hexagons). The raw residuals and the old fixed-threshold verdict
     (*Legacy verdict*, thresholds under *Advanced*) are still shown and saved.
5. **Cube reading (Necker toggle).** *Convex / vertex-out (+1)*: the centre is the near corner and the cube's edges
   run to V1, V3, V5. *Concave / vertex-in (−1)*: the centre is the far corner and the edges run to V0, V2, V4.
6. **Orientation field.** A structure tensor (smoothed products of image gradients) gives each pixel a dominant line
   orientation and a *coherence* (0 = no preferred direction, 1 = one clear direction). σ gradient / σ tensor
   (Advanced) control the smoothing.
7. **Ternary regions and sign convention.** Three rhombi are read, each centred on one of the reading's own cube
   edges (EDGE_Vk = the edge from the centre to Vk). **+1 = structure along the convex cube's edges; −1 = along the
   concave corner's edges.** Calibrated on a tumbling-block tile: its real edges score +1, +1, +1 under Convex.
   Within each region (minus the *edge margin*) the orientations are averaged. Coherence below the minimum gives
   **0**; alignment with the region's long diagonal (the edge itself) gives the reading's sign; alignment across it
   gives the opposite sign; anything else **0**. *Ternary regions → Faces (original)* restores the first rule, which
   read the faces instead and credited a real cube's edges to the opposite reading.
8. **Edge margin.** By default it is measured in the photo's own pixels: wall width + 8 px (*Margin buffer*). Small cells are enlarged
   a lot when normalised, so a fixed share of the frame let their blurred outline count as texture.
9. **Primary line.** The best primary-signed region's long diagonal (a cube edge), with its angle from the neutral
   vertical reference V3→V0.
10. **Texture lock.** Does the texture sit on the hexagon's own directions? *p* is a rotation-shuffle result: the
    chance a randomly rotated cell lines up as closely. Locked = p < 0.01. One cell rarely gets there; Batch review
    and Neighbour cluster pool several cells, which is where the test has power.

### Manifold overlays (working definitions)
The spec names four templates but does not define them. These are paths on a 3D cube, projected isometrically.
Dashed segments are hidden (behind the cube) for the current Necker state.
* **Around the cube**: the belt through the six edge midpoints that avoid the two diagonal corners.
* **Around a vertex**: a loop through the three faces at the fixed corner (1,1,1). Solid when convex, dashed when concave.
* **Reverse around a vertex**: the same loop at the opposite corner, traversed the other way.
* **Across six faces**: a closed path through all six face centres, crossing each shared edge at its midpoint.

### Neighbour cluster
Mark *Cell 1 (centre)* and up to three neighbours. Vertices of different cells closer than the *snap tolerance*
are merged. For each shared edge, the orientation of the region touching it on each side is mapped back to the
photo and compared: score = |cos Δ| (1 = continuous). *Relation* compares the two regions' ternary values:
same / flipped / unrelated (either is 0). The pooled texture-lock test across the marked cells is shown too.

### Controls
* **Undo / Clear**: remove the last point or all points of the current cell.
* **Auto-detect**: a secondary helper (edges → contours → six-corner convex polygon) that suggests points near your
  last click or the image centre. Always check it.
* **Save points**: write the points, settings and scores to `data/points/`. Complete cells are also saved
  automatically when they are scored.
* **Use ground-truth points**: for the bundled samples only, load the exact vertices.
* **Batch review**: every image re-scored with the current settings; *Export all* writes a comparison sheet per image,
  a CSV and a ZIP to `data/exports/`.

### Honest caveats
The manifold paths, the ternary rule and sign convention, the edge margin, the snapping quality gate and the verdict
rules are **working definitions** chosen for this test bed (the convention, margin and verdicts were calibrated on
the 24 Nov 2026 baseline test). They are parameters, not established results. The null baseline uses one particular family of
random hexagons (random angles, radii 0.5–1.5), and a different family would give different percentages.
"""


def render() -> None:
    st.markdown(HELP)
