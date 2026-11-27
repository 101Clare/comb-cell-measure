# Interface contracts (shared by all build agents)

These conventions are binding. Every module in `hexengine/` must follow them, so the parts fit together
without guesswork. If you need to change a signature, change it here *and* in the stub, and say so in your report.

## Coordinate conventions

* **Image space**: pixel coordinates `(x, y)`, x right, **y down**, as loaded by OpenCV/PIL (RGB uint8 arrays, shape `(H, W, 3)`).
* **Canonical units**: the perfect hexagon has circumradius **1**, centre at the origin, also **y down**.
  Vertex `k` (k = 0..5) sits at angle `theta_k = -90° + 60°·k` measured with `atan2(y, x)` in y-down coords, i.e.
  ```
  V0 = ( 0.000, -1.0)  top
  V1 = ( 0.866, -0.5)  upper right
  V2 = ( 0.866,  0.5)  lower right
  V3 = ( 0.000,  1.0)  bottom
  V4 = (-0.866,  0.5)  lower left
  V5 = (-0.866, -0.5)  upper left
  ```
  So the vertices run **clockwise on screen** (pointy-top hexagon, i.e. the isometric cube silhouette).
* **Canonical frame**: a square raster of `FRAME_SIZE = 512` px. Canonical unit `u` maps to frame pixel
  `FRAME_SIZE/2 + u * FRAME_SCALE`, with `FRAME_SCALE = 0.45 * FRAME_SIZE`. Use `canonical.canonical_to_frame_matrix()`.
* **Angles / orientations**: radians, `atan2(dy, dx)` in the relevant y-down space. *Orientations* (undirected lines)
  live in `[0, π)`. *Directions* live in `(-π, π]`.
* **Transforms** are always 3×3 numpy float64 matrices acting on homogeneous column vectors `[x, y, 1]ᵀ`.
  `HexFit.affine/similarity/homography` map **image px → canonical units**.

## Vertex ordering (important subtlety)

Every cyclic relabelling or reflection of a regular hexagon is itself a similarity transform, so the residual scores are
**identical for every starting vertex and both windings**. The order therefore only affects which way "up" the
normalised image lands. Rule: `transforms.order_points()` makes the winding clockwise-on-screen and makes the point
whose angle from the centroid is closest to −90° (topmost) become index 0. Then the user's `rotation_offset` (0..5) shifts it
further, so index `(0 + offset) % 6` of the ordered list maps to V0. An odd offset swaps which interior "Y" the cube reading uses.

## Necker / cube states

`cube.CubeState.CONVEX` (sign +1, "vertex-out"): the near vertex is at the centre and the three visible cube edges are the
spokes centre→V1, V3, V5. Visible faces are TOP `(C,V5,V0,V1)`, RIGHT `(C,V1,V2,V3)` and LEFT `(C,V3,V4,V5)`.

`cube.CubeState.CONCAVE` (sign −1, "vertex-in"): the centre is the far (interior) vertex, and the spokes run centre→V0, V2, V4.
Faces are UPPER_RIGHT `(C,V0,V1,V2)`, BOTTOM `(C,V2,V3,V4)` and UPPER_LEFT `(C,V4,V5,V0)`.

## Ternary rule (working definition, all thresholds are parameters)

**Regions read (sign convention, calibrated 2026-11-24).** By default (`convention="edges"`) the three rhombi read for a
reading are the ones centred on *that reading's own cube edges*: for CONVEX the rhombi whose long diagonals are the
spokes to V1, V3, V5 (named `EDGE_V1`, `EDGE_V3`, `EDGE_V5`), for CONCAVE those along V0, V2, V4. So:
**+1 = structure along the convex cube's edges; −1 = structure along the concave corner's edges.** Calibrated on the
tumbling-block tile control, whose real edges score +1, +1, +1 under CONVEX and 0, 0, 0 under CONCAVE.
`convention="faces"` keeps the original rule (each reading's own faces). It credited that same tile to CONCAVE
(−1, −1, −1), because a reading's edges are its faces' boundaries, which the edge margin hides.

Per region: the structure-tensor dominant orientation `θ_f` and coherence `c_f`.
* **radial line** of a region = its long diagonal (centre → the outer vertex opposite the centre). This is the line running
  outward from the primary (centre) vertex.
* **tangential line** = its short diagonal (perpendicular to the radial line), i.e. wrapping *around* the vertex.
* If `c_f < coherence_min` → **0**.
* If `|θ_f − radial| ≤ angle_tol` → `state.sign` (convex gives +1, concave gives −1: "primary").
* If `|θ_f − tangential| ≤ angle_tol` → `−state.sign`.
* Otherwise → **0**.
The **neutral reference line (0)** is the canonical vertical axis V3→V0. The **primary line** is the radial line of the
face with the strongest primary-signed alignment (coherence × alignment).

## Edge margin (calibrated 2026-11-24)

Regions are read through masks that stay clear of every region boundary. By default the margin is set in the
ORIGINAL photo's pixels: estimated wall width + a buffer of 8 px (`inference.source_margin_frac`; the patch asked for 4 px, but 4 px left
a false texture lock in soap froth on the baseline photos and 8 px removed it), converted to a fraction of the
hexagon's radius, and never smaller than the blur reach (4·σ_tensor + 4·σ_grad + 2 frame px). A fixed fraction of the
normalised frame (the original behaviour) let small, heavily enlarged cells read their own outline as texture.

## Verdicts (calibrated 2026-11-24)

* **Shape** (`Scores.verdict`): residual of the selected transform vs that cell's **noise floor** (`metrics.noise_floor`:
  95th percentile residual of a perfect hexagon of the same pixel size with the measured corner uncertainty).
  `regular` = at or below the floor; `random-like` = no better than 5 % of random hexagons; otherwise `irregular`.
  The original fixed-threshold verdict is kept as `Scores.threshold_verdict`, and all raw residuals are logged.
* **Texture lock** (`inference.texture_lock` / `pooled_lock_test`): rotation-shuffle p < 0.01. Per cell the shuffle is
  exact (p = offset / 15°); pooling several cells is what gives the test power.
