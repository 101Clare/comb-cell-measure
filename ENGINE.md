# Hexagonal Geometric Test Engine

A small, local **research test bed** for one question: *how close is a hexagon in a photo to a projected
regular hexagon - and, read as an isometric cube, what does its texture say?*

You load an image (honeycomb, basalt column, turtle shell, tiles...), click the six corners of a hexagon, and
the engine maps it onto a mathematically perfect reference hexagon. It then reports how much distortion is
left over, overlays the cube "manifold" paths, reads the texture direction on each cube face as a
+1 / 0 / −1 ternary state, and checks continuity across neighbouring cells.

This is a tool for **testing hypotheses honestly**, not a finished product. Every score comes with a random-hexagon
null baseline, and all the maths lives in plain Python (`hexengine/`) so you can check it, change it, or
call it from a notebook and get exactly the numbers the app shows.

## Quick start

1. **Install Docker Desktop** (free for personal use): <https://www.docker.com/products/docker-desktop/>.
   It sets up WSL2 for you; restart if asked, then start Docker Desktop and wait for *Engine running*.
2. **Start the engine.** In PowerShell, in this folder:

   ```powershell
   .\start.ps1
   ```

   or, doing the same by hand:

   ```powershell
   docker compose up -d --build
   ```

   The first build downloads about 1 GB and takes a few minutes. Later starts take seconds.
3. **Open <http://localhost:8501>** (`start.ps1` opens it for you once the app is healthy).

Other launcher options:

| Command | What it does |
|---|---|
| `.\start.ps1 -Logs` | Follow the app's log output (Ctrl+C stops watching; the app keeps running) |
| `.\start.ps1 -Stop` | Stop and remove the container (your `data\` folder is kept) |

If Windows refuses to run the script, allow local scripts once with
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### Running without Docker

The same app runs straight from the project's Python environment, with no container:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

Then open <http://localhost:8501>. It uses the same `data\` folder as the container. Don't run both at once,
because they share port 8501.

## Where your data goes

Everything the app saves lands in the **`data\`** folder next to this README, which is mounted into the
container at `/app/data`:

* uploaded images and the six points you clicked for each one (so you can come back later),
* `data\exports\` - PNG comparison sheets, a CSV of every score across all images, settings/points JSON
  and a ZIP of everything.

Deleting or rebuilding the container never touches `data\`.

**Live editing:** `hexengine\`, `app\` and `samples\` are mounted into the container too, and Streamlit is
set to re-run on save. Edit a formula in `hexengine\metrics.py`, save, and the app picks it up straight
away - no rebuild. Only changes to `requirements.txt` or the `Dockerfile` need `docker compose up -d --build`.

## Sample images

`samples\` holds synthetic images with **known answers**, made by `samples\generate_samples.py` (seeded, so
always identical). `samples\ground_truth.json` has the exact six vertices for each, what the engine should
report, and reference scores. Use them to check the engine before trusting it with real photos.

| File | What it tests |
|---|---|
| `regular_hexagon.png` | Perfect hexagon: every residual ≈ 0 |
| `sheared_hexagon.png` | Affine shear + rotation: affine ≈ 0, similarity > 0 |
| `perspective_hexagon.png` | Camera at an angle: homography ≈ 0, affine > 0 |
| `shaded_cube.png` | Isometric cube with radial line texture: ternary +1/+1/+1 when read convex; the Necker toggle changes it |
| `cube_off_axis.png` | Real cube from a non-diagonal view: central-symmetry error ≈ 0 but affine > 0 |
| `irregular_hexagon.png` | Random hexagon: the "noise" control, everything should score badly |
| `honeycomb.png` | Tiled honeycomb: centre cell + 3 neighbours for continuity mode |
| `rotated_flat_top.png` | Flat-top hexagon: tests automatic vertex ordering and the rotation offset |

To regenerate them: `python samples/generate_samples.py` (inside the container:
`docker compose exec hex-engine python samples/generate_samples.py`).

## Running the tests

Inside the container (nothing to install on your PC):

```powershell
docker compose exec hex-engine pytest -q
```

Or locally, with the project's virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

(To create that environment from scratch: `python -m venv .venv` then
`.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.)

## Usage

The app has a **sidebar** (images and settings) and four tabs: **Single image**, **Neighbour cluster**,
**Batch review** and **Help**.

**First test, about 2 minutes**

1. In the sidebar, leave *Use ground-truth points* ticked and click **Load sample images**. The eight synthetic
   samples are copied into `data/uploads/` with their exact vertices, and each one shows as *scored* in the image
   list.
2. Pick `sheared_hexagon.png`. On *Single image* the original with numbered points sits next to the normalised
   (un-sheared) hexagon. The tiles show the **mean vertex residual**, which is about 0 here, next to this cell's
   **noise floor** (what a perfect hexagon of the same size would score from corner error alone) and the
   central-symmetry error. Each tile says what share of 1,000 random hexagons score better (lower). The **shape
   verdict** is *regular* when the residual is within the noise floor, *irregular* above it, and *random-like* when
   it is no better than 5 % of random hexagons. It is judged on the transform selected under *Warp transform*.
   Pick `perspective_hexagon.png` to see why that matters: it is irregular under affine but regular under
   homography, because it is a camera view at an angle.
3. Pick `irregular_hexagon.png` for comparison: every residual is high and the shape verdict is not *regular*.
4. Pick `shaded_cube.png`, tick *Ternary colouring* and switch the **Necker state** between Convex (+1) and
   Concave (−1). The regions, signs and manifold solid/dashed segments all flip. Its line texture runs along the
   concave corner's edges, so it reads 0, 0, 0 under Convex and −1, −1, −1 under Concave.

**Marking your own images**

* Upload one or more images (jpg/png/bmp/tif/webp). They are stored in `data/uploads/` and stay there after a
  restart.
* Click the six vertices in order **around** the hexagon. You can start anywhere and go either way. Numbered
  markers appear as you click, and the caption shows the winding (CW/CCW) once there are three points.
  **Undo** removes the last point and **Clear** removes them all.
* **Auto-detect** is an optional helper. It suggests six points from the image edges, near your last click or the
  image centre. Check them, and Undo/re-click any that are off.
* Once six points are marked they are **snapped to the walls**: a line is fitted along each wall and the corners
  become the intersections, to a fraction of a pixel. A green note says how far they moved and how precisely
  they are located; an amber note says why snapping was refused and your clicks were used instead. The expander
  under the image shows your clicks (cyan) against the snapped corners (gold).
* Results appear as soon as six points are marked, and the record (points, snapped corners, settings, scores) is
  saved to `data/points/` automatically. **Save points** saves at any time, including partial work.
* *Click image width* (sidebar → Advanced) changes the on-screen size only. Points are always stored in the
  original image's full-resolution pixels.

**Settings worth knowing**

* *Warp transform*: whether the affine or the homography fit warps the image. The scores always show all three.
* *Rotation offset* / *Rotate 60°*: chooses which vertex becomes V0. An odd offset swaps the cube's interior "Y".
  The offset belongs to the image: it is saved with its points and restored when you come back to it.
* *Overlays*: construction lines, each of the four manifold templates, the orientation field and the ternary
  colouring, with opacity and line width.
* *Corners*: snap on/off, wall type (Auto, dark line, bright line, edge between tones) and wall width (0 = measure).
* *Ternary inference*: angle tolerance, coherence minimum, the **edge margin** (default: wall width + 8 px in the
  photo's own pixels; or a fixed fraction) and the **ternary regions** (Edges, calibrated, or Faces, the original
  rule). Under *Advanced*: the legacy fixed thresholds (logged only) and the structure-tensor smoothing (σ).

**Neighbour cluster**: choose *Cell 1 (centre)* and click its six vertices, then do the same for Cell 2–4 on the
neighbouring hexagons (each cell has its own colour). Vertices closer than the *snap tolerance* are merged. With
two or more complete cells, each shared edge is drawn green/amber/red by its continuity score, with a table of
Δ angle, score, the ternary value on each side and the relation (same / flipped / unrelated). The honeycomb
sample's ground truth fills all four cells.

**Batch review**: a sortable table of every image with points, re-scored with the *current* settings, and a
gallery of normalised thumbnails. **Export all** writes a comparison sheet per image, a CSV of all scores and a
ZIP of everything to `data/exports/`, and offers the ZIP and CSV as downloads. On *Single image*, **Build
comparison sheet** and **Download record (JSON)** export one image.

## Geometry explained

The pipeline, step by step. Every step is a separate function in `hexengine/`, and the docstrings there give the
exact formulas.

1. **Canonical hexagon** (`canonical.py`). It is a regular hexagon with circumradius 1, centred on the origin,
   with the y axis pointing **down** (as in images). V0 is at the top and V0…V5 run clockwise on screen. This is
   exactly the outline of a unit cube seen straight down its body diagonal: the isometric view.

2. **Ordering the clicks** (`transforms.order_points`). The six clicks can start anywhere and go either way. The
   engine makes them clockwise and labels the top-most one V0. Because every relabelling of a regular hexagon is
   itself a rotation or reflection, **the residual scores do not depend on the order at all**. The order only
   decides which way up the normalised image lands. *Rotation offset* moves V0 round by 60° steps.

3. **Fitting** (`transforms.fit_hexagon`). Six points give 12 numbers, which is more than any of the three
   transforms needs, so each is fitted by least squares:
   * **similarity** (rotation + uniform scale + shift, 4 parameters) fits exactly only a *regular* hexagon;
   * **affine** (adds shear and stretch, 6 parameters) fits exactly any *parallel projection* of a regular
     hexagon;
   * **homography** (adds perspective, 8 parameters) fits exactly any *camera view* of a flat regular hexagon.

4. **Residuals** (`metrics.py`). After fitting, each of the six points lands some distance from its canonical
   vertex. The **mean vertex residual** is the average of those six distances, in units of the circumradius
   (0.05 means "5 % of the hexagon's radius off"). It measures how much of the observed shape that transform
   *cannot* explain. The **central-symmetry error** checks whether opposite sides are parallel and equal. Any
   parallel view of a real cube is centrally symmetric even when it is not an affine image of a *regular*
   hexagon, so this score stops genuine cube-like shapes being penalised by the affine residual alone
   (`cube_off_axis.png` is that case).

5. **Null baseline.** 1,000 random hexagons (fixed seed) are scored the same way. "x % of random hexagons score
   better" puts each number in context: near 0 % means the observed shape beats almost every random one.
   Note that the homography, with 8 parameters, fits random shapes better than the affine does, so compare
   each score with its own baseline.

6. **Normalising** (`transforms.warp_to_canonical`). The image is warped with the chosen transform into a
   512×512 frame where the hexagon is exactly the canonical one. Everything after this happens in that frame.

7. **Necker cube** (`cube.py`). The flat hexagon with an interior "Y" can be read two ways:
   * **convex**: the centre is the near corner, pointing at you, with edges to V1, V3, V5 and faces TOP / RIGHT /
     LEFT;
   * **concave**: the centre is the far, inside corner of a box, with edges to V0, V2, V4 and faces UPPER_RIGHT /
     BOTTOM / UPPER_LEFT.

   The toggle changes which three rhombi count as faces, which manifold segments are solid or dashed, and the
   sign convention.

8. **Manifold overlays** (`manifolds.py`). The four paths are built on a real 3D cube and projected:
   * *Around the cube* is the belt through the midpoints of the six side edges.
   * *Around a vertex* and *Reverse around a vertex* are the loops through the three faces round the corner
     (1,1,1) and its opposite corner, run in opposite directions.
   * *Across six faces* is a closed tour of all six face centres.

   Hidden parts are dashed.

9. **Orientation field** (`inference.py`). A structure tensor measures, at every pixel, the dominant direction
   of lines (perpendicular to the brightness gradient) and a **coherence** from 0 to 1. Coherence is 0 for
   flat or random texture and 1 for perfectly parallel lines. Each face's direction is the coherence-weighted
   average using the "doubled-angle" trick, so 5° and 175° average to 0°, not 90°. A band along every region
   boundary (*edge margin*) is left out so that the hexagon outline and the spokes are not mistaken for texture.
   By default the band is the wall's width + 8 px **in the photo's own pixels**, so a small cell that is enlarged
   many times by normalising still has its outline excluded.

10. **Ternary state** per region (*working definition*, calibrated). The three regions read are the rhombi centred
    on the reading's own cube edges (EDGE_Vk = the edge from the centre to Vk). If a region's line direction runs
    along its **long diagonal** (that edge), it gets the reading's sign; across it, the opposite sign; weak
    coherence or any other angle, **0**. So **+1 = structure along the convex cube's edges, −1 = along the
    concave corner's edges**. On the tumbling-block tile control the real edges score +1, +1, +1 under Convex.
    (The original rule read the faces instead; a cube's edges are its faces' boundaries, so it credited the tile
    to the concave reading. It is still available as *Ternary regions → Faces*.) The **primary line** is the edge
    that carries the primary sign most strongly, measured from the neutral vertical reference V3→V0.

11. **Continuity** (`neighbours.py`). For two marked cells that share an edge, the region next to that edge on
    each side has its direction mapped *back into the photo*. The two are then compared: score = |cos Δ|, where 1
    means the line carries straight across the edge. The relation is *same* or *flipped* depending on the
    ternary values on either side.

12. **Corner snapping and verdicts** (`refine.py`, `metrics.py`). Snapping fits a line along each wall and
    measures each corner's uncertainty. From that, the **noise floor** is the residual a perfect hexagon of the
    same pixel size would show from corner error alone. The shape verdict compares the residual with it.
    **Texture lock** asks whether the texture sits on the hexagon's own directions more closely than randomly
    rotated cells would (shuffle p < 0.01). Pooling several cells (Batch review, Neighbour cluster) gives it power.

**What the numbers can and cannot tell you.**
* A low residual says the *outline* is consistent with a projected regular hexagon or cube silhouette. On its
  own it says nothing about the 3D structure inside.
* The ternary and continuity readings depend on there being oriented texture inside the faces. A cell with
  plain interiors (like `honeycomb.png` once its walls are masked out) gives mostly 0 and "unrelated", which is
  the honest answer.
* The convex and concave readings share the same three line directions. For texture that runs the same way
  everywhere, flipping mostly inverts the signs; what changes is which part of the hexagon counts as which face.
* The manifold paths, the ternary rule and sign convention, the edge margin, the snapping quality gate and the
  verdict rules are working definitions. Each lives in one clearly marked place in the code so you and Ben can
  change them.

## Drawings and spirals (polygon.py, twist.py)

Added for analysing hand-drafted spiral constructions. Python only for now; they are not in the app yet.

* `polygon.snap_polygon(image, rough_corners, occluded={side: [(t0, t1), ...]})` snaps any n-gon's drawn sides to
  sub-pixel corners with a bootstrap error for each. The search is narrow (5 px), so mark the corners to within
  about 3 px. Corners that moved far or stayed uncertain are listed in `suspect`.
* `polygon.cyclic_ring(corners)` gives the circle fit, side lengths and angular steps. For an even n it also gives
  the long/short step (the drawings tested: 41 / 19 degrees, not 30 / 30).
* `polygon.step_candidates(step, se)` lists the constructible angles that fit a measured step.
* `twist.measure(image, centre, r_in, r_out)` returns the chirality index C, its sector-bootstrap 95 % interval,
  the handedness, and the **pitch peaks** (the angles, from radial, of the line families present). It was
  calibrated on synthetic logarithmic spirals (`twist.synthetic_spiral`): C = sin(2 x pitch) to within 0.01 and
  the pitch peak within 2 degrees. C alone can't tell 15 degrees from 75 degrees, so read the peaks for pitch.

## Project layout

```
hex-engine/
├── start.ps1               # Windows launcher: start / -Stop / -Logs
├── docker-compose.yml      # one service on port 8501, data\ and source folders mounted
├── Dockerfile              # python:3.12-slim + pinned requirements, runs as a non-root user
├── requirements.txt        # pinned Python packages
├── .streamlit/config.toml  # dark theme, gold accent, live reload on save
├── CONTRACTS.md            # coordinate, ordering and Necker conventions shared by all modules
├── app/                    # the Streamlit user interface (thin; calls hexengine)
├── hexengine/              # all the maths, UI-free and importable from a notebook
│   ├── canonical.py        #   the perfect reference hexagon and construction lines
│   ├── transforms.py       #   affine / homography / similarity fitting, vertex ordering, warping
│   ├── metrics.py          #   residual and symmetry scores, random-hexagon null baseline
│   ├── cube.py             #   isometric cube faces and the convex / concave (Necker) states
│   ├── manifolds.py        #   the four geodesic overlay paths
│   ├── inference.py        #   orientation field, per-face directions, ternary states
│   ├── neighbours.py       #   multi-cell shared edges and continuity scores
│   ├── polygon.py          #   n-sided polygon snapping (occluded sides), cyclic rings, construction angles
│   ├── twist.py            #   spiral handedness (chirality index) and pitch angles, synthetic spirals
│   ├── render.py           #   overlays and comparison sheets
│   └── session.py          #   saving / loading points and results
├── samples/                # synthetic test images + ground_truth.json + their generator
├── tests/                  # pytest suite
└── data/                   # your uploads, saved points and exports (created on first run)
```
