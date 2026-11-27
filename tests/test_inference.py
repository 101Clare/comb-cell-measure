"""Tests for hexengine.inference on synthetic canonical-frame images."""

import numpy as np
import pytest

from hexengine import canonical, cube, inference
from hexengine.types import CubeState, Face, FaceResult

SIZE = canonical.FRAME_SIZE
WAVELENGTH = 12.0  # px


def stripes(theta: float, size: int = SIZE, wavelength: float = WAVELENGTH) -> np.ndarray:
    """Grey float image of sinusoidal stripes whose LINES run at orientation theta (y-down pixel coords)."""
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    normal = np.array([-np.sin(theta), np.cos(theta)])  # perpendicular to the line direction
    return 0.5 + 0.5 * np.sin(2 * np.pi * (xx * normal[0] + yy * normal[1]) / wavelength)


def paint_faces(state: CubeState, attr: str) -> np.ndarray:
    """Canonical frame where each face of `state` carries stripes along its `attr` line (radial/tangential)."""
    img = np.full((SIZE, SIZE), 0.5)
    for face, mask in zip(cube.faces(state), inference.face_masks(state, SIZE, edge_margin_frac=0, convention="faces")):
        img[mask] = stripes(getattr(face, attr))[mask]
    return (img * 255).astype(np.uint8)


def shaded_cube_frame(levels=(0.9, 0.6, 0.3), background=0.1) -> np.ndarray:
    """Untextured convex cube: three flat faces of different brightness on a dark background."""
    img = np.full((SIZE, SIZE), background)
    for level, mask in zip(levels, inference.face_masks(CubeState.CONVEX, SIZE, edge_margin_frac=0, convention="faces")):
        img[mask] = level
    return (img * 255).astype(np.uint8)


def ang_err_deg(a: float, b: float) -> float:
    return np.degrees(inference.undirected_diff(a, b))


# ----------------------------------------------------------------------------- orientation field

@pytest.mark.parametrize("deg", [0.0, 30.0, 75.0, 90.0, 135.0, 170.0])
def test_stripes_orientation_recovered(deg):
    theta = np.deg2rad(deg)
    field = inference.orientation_field(stripes(theta))
    inner = (slice(40, -40), slice(40, -40))
    assert np.all(field.theta >= 0) and np.all(field.theta < np.pi)
    assert np.median(field.coherence[inner]) > 0.9
    mask = np.zeros((SIZE, SIZE), bool)
    mask[inner] = True
    t, c = inference.dominant_orientation(field, mask)
    assert ang_err_deg(t, theta) < 2.0
    assert c > 0.95


def test_rgb_input_accepted():
    grey = (stripes(np.deg2rad(40)) * 255).astype(np.uint8)
    rgb = np.stack([grey] * 3, axis=2)
    f = inference.orientation_field(rgb)
    mask = canonical.hexagon_mask()
    t, c = inference.dominant_orientation(f, mask)
    assert ang_err_deg(t, np.deg2rad(40)) < 2.0 and c > 0.9


def test_noise_low_coherence():
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 1, (SIZE, SIZE))
    f = inference.orientation_field(img)
    assert f.coherence.mean() < 0.3
    _, c = inference.dominant_orientation(f, canonical.hexagon_mask())
    assert c < 0.15


def test_flat_image_zero_coherence_no_nans():
    f = inference.orientation_field(np.full((SIZE, SIZE, 3), 128, np.uint8))
    for arr in (f.theta, f.coherence, f.energy):
        assert np.all(np.isfinite(arr))
    assert np.all(f.coherence == 0)
    assert np.all(f.energy == 0)
    assert inference.dominant_orientation(f, canonical.hexagon_mask()) == (0.0, 0.0)


def test_dominant_orientation_doubled_angle_wraps():
    # 5 deg and 175 deg are 10 deg apart; their mean must be ~0 deg, not 90 deg.
    theta = np.zeros((2, 2))
    theta[0, :] = np.deg2rad(5)
    theta[1, :] = np.deg2rad(175)
    f = inference.OrientationField(theta=theta, coherence=np.ones((2, 2)), energy=np.ones((2, 2)))
    t, c = inference.dominant_orientation(f, np.ones((2, 2), bool))
    assert ang_err_deg(t, 0.0) < 1e-6
    assert c == pytest.approx(np.cos(np.deg2rad(10)))


# ----------------------------------------------------------------------------- ternary rule (pure)

def test_assign_ternary_rule():
    r, tg = np.pi / 2, 0.0
    cv, cc = CubeState.CONVEX, CubeState.CONCAVE
    assert inference.assign_ternary(np.pi / 2 + 0.1, 0.8, r, tg, cv) == 1
    assert inference.assign_ternary(np.pi / 2 + 0.1, 0.8, r, tg, cc) == -1
    assert inference.assign_ternary(np.pi - 0.05, 0.8, r, tg, cv) == -1   # wraps mod pi to tangential
    assert inference.assign_ternary(np.pi - 0.05, 0.8, r, tg, cc) == 1
    assert inference.assign_ternary(np.pi / 4, 0.8, r, tg, cv) == 0       # oblique
    assert inference.assign_ternary(np.pi / 2, 0.1, r, tg, cv) == 0       # incoherent
    assert inference.assign_ternary(np.deg2rad(60), 0.8, r, tg, cv, angle_tol_deg=35) == 1


# ----------------------------------------------------------------------------- face analysis

def test_face_masks_partition_hexagon():
    for state in CubeState:
        masks = inference.face_masks(state, edge_margin_frac=0, convention="faces")
        assert len(masks) == 3
        total = np.sum(masks, axis=0)
        hexm = canonical.hexagon_mask()
        # faces cover the hexagon, overlapping only on their shared boundary lines
        assert (total[hexm] >= 1).mean() > 0.99
        assert (total > 1).mean() < 0.01


def test_face_masks_keep_clear_of_boundaries():
    import cv2
    frac = 0.12
    margin = frac * canonical.FRAME_SCALE
    full = inference.face_masks(CubeState.CONVEX, edge_margin_frac=0, convention="faces")
    eroded = inference.face_masks(CubeState.CONVEX, convention="faces")            # default margin 0.12
    assert inference.face_masks(CubeState.CONVEX, SIZE, convention="faces") is not None  # positional signature unchanged
    for f, e in zip(full, eroded):
        assert e.sum() > 0 and np.all(f[e])                     # subset, not empty
        dist = cv2.distanceTransform(f.astype(np.uint8), cv2.DIST_L2, 5)
        assert dist[e].min() > margin - 1.0                     # nothing within the margin of any face edge
    floored = inference.face_masks(CubeState.CONVEX, edge_margin_frac=0.0, min_margin_px=15.0, convention="faces")
    assert all(fl.sum() < f.sum() for fl, f in zip(floored, full))
    assert inference.edge_margin_px(SIZE, 0.01, sigma_tensor=6.0, sigma_grad=1.5) == pytest.approx(32.0)
    assert inference.edge_margin_px(SIZE, 0.12) == pytest.approx(margin)


def test_uniform_hexagon_warped_reads_zero():
    from hexengine import transforms
    h, w = 480, 520
    img = np.full((h, w, 3), 15, np.uint8)
    ang = np.deg2rad(-90 + 60 * np.arange(6) + 7)
    pts = np.array([260, 240]) + 190 * np.stack([np.cos(ang), 0.8 * np.sin(ang)], axis=1)
    import cv2
    cv2.fillPoly(img, [np.round(pts).astype(np.int32)], (200, 180, 150))
    fit = transforms.fit_hexagon(pts)
    warped = transforms.warp_to_canonical(img, fit, mask_outside=True)
    field = inference.orientation_field(warped)
    for state in CubeState:
        res = inference.analyse_faces(field, state, convention="faces")
        assert [r.ternary for r in res] == [0, 0, 0], (state, [(r.ternary, r.coherence) for r in res])


def test_untextured_shaded_cube_reads_zero():
    # Flat faces of different brightness: the only gradients are the outline and the convex spokes, which are
    # face boundaries under CONVEX and are excluded by the margin.
    field = inference.orientation_field(shaded_cube_frame())
    res = inference.analyse_faces(field, CubeState.CONVEX, convention="faces")
    assert [r.ternary for r in res] == [0, 0, 0], [(r.ternary, r.coherence) for r in res]


def test_convex_radial_stripes_all_plus_one():
    img = paint_faces(CubeState.CONVEX, "radial_theta")
    field = inference.orientation_field(img)
    res = inference.analyse_faces(field, CubeState.CONVEX, convention="faces")
    assert [r.ternary for r in res] == [1, 1, 1]
    for r in res:
        assert ang_err_deg(r.dominant_theta, r.face.radial_theta) < 5
        assert r.radial_alignment > 0.95 and r.tangential_alignment < 0.3
    pl = inference.primary_line(res, CubeState.CONVEX)
    assert pl is not None and pl.ternary == 1
    assert pl.face_name in {f.name for f in cube.faces(CubeState.CONVEX)}


def test_convex_tangential_stripes_all_minus_one():
    img = paint_faces(CubeState.CONVEX, "tangential_theta")
    res = inference.analyse_faces(inference.orientation_field(img), CubeState.CONVEX, convention="faces")
    assert [r.ternary for r in res] == [-1, -1, -1]
    assert inference.primary_line(res, CubeState.CONVEX) is None


def test_concave_radial_stripes_all_minus_one():
    img = paint_faces(CubeState.CONCAVE, "radial_theta")
    res = inference.analyse_faces(inference.orientation_field(img), CubeState.CONCAVE, convention="faces")
    assert [r.ternary for r in res] == [-1, -1, -1]
    pl = inference.primary_line(res, CubeState.CONCAVE)
    assert pl is not None and pl.ternary == -1


def test_same_image_different_state_differs():
    # Convex-radial image read as CONCAVE: the concave faces straddle two convex faces, and no face
    # should carry the concave primary sign (-1), so there is no concave primary line.
    img = paint_faces(CubeState.CONVEX, "radial_theta")
    field = inference.orientation_field(img)
    convex = [r.ternary for r in inference.analyse_faces(field, CubeState.CONVEX, convention="faces")]
    concave_res = inference.analyse_faces(field, CubeState.CONCAVE, convention="faces")
    assert -1 not in [r.ternary for r in concave_res]
    assert inference.primary_line(concave_res, CubeState.CONCAVE) is None
    assert inference.primary_line(inference.analyse_faces(field, CubeState.CONVEX, convention="faces"), CubeState.CONVEX) is not None

    # Uniform vertical stripes: only the face whose radial line is vertical fires, with the state's sign.
    field_v = inference.orientation_field(stripes(np.pi / 2))
    a = inference.analyse_faces(field_v, CubeState.CONVEX, convention="faces")
    b = inference.analyse_faces(field_v, CubeState.CONCAVE, convention="faces")
    ta, tb = [r.ternary for r in a], [r.ternary for r in b]
    assert sorted(ta) == [0, 0, 1]
    assert sorted(tb) == [-1, 0, 0]
    assert ta != tb and convex == [1, 1, 1]


def test_primary_line_angle_vs_reference():
    faces = cube.faces(CubeState.CONVEX)
    results = [FaceResult(face=f, dominant_theta=f.radial_theta, coherence=0.5 + 0.1 * k, ternary=1,
                          radial_alignment=1.0, tangential_alignment=0.0) for k, f in enumerate(faces)]
    pl = inference.primary_line(results, CubeState.CONVEX)
    best = faces[2]
    assert pl.face_name == best.name
    np.testing.assert_allclose(pl.start, [0, 0], atol=1e-12)
    d = pl.end - pl.start
    expected = np.degrees(np.arctan2(d[0], -d[1]))  # clockwise-positive angle from "up" in y-down coords
    assert pl.angle_vs_reference_deg == pytest.approx(expected)
    assert ang_err_deg(np.arctan2(d[1], d[0]), pl.theta) < 1e-9


# ----------------------------------------------------------------------------- "edges" convention (calibrated)

def test_edges_convention_default_and_regions_are_dual_faces():
    regs = inference.regions(CubeState.CONVEX)
    assert [r.name for r in regs] == ["EDGE_V1", "EDGE_V3", "EDGE_V5"]
    # the long diagonal of every region is a convex spoke
    assert sorted(r.outer_vertex_ids[1] for r in regs) == list(cube.spoke_ids(CubeState.CONVEX))
    assert sorted(r.outer_vertex_ids[1] for r in inference.regions(CubeState.CONCAVE)) == [0, 2, 4]


def test_shaded_convex_cube_scores_plus_one_under_convex_edges():
    """Calibration on the control: a cube's real edges (brightness steps on the convex spokes) are +1 under CONVEX."""
    field = inference.orientation_field(shaded_cube_frame())
    convex = inference.analyse_faces(field, CubeState.CONVEX)
    assert [r.ternary for r in convex] == [1, 1, 1]
    pl = inference.primary_line(convex, CubeState.CONVEX)
    assert pl is not None and pl.ternary == 1 and pl.face_name.startswith("EDGE_V")
    concave = inference.analyse_faces(field, CubeState.CONCAVE)
    assert inference.primary_line(concave, CubeState.CONCAVE) is None


def test_old_faces_convention_credited_the_other_reading():
    """Documents why the convention changed: the original rule gives the convex cube's edges to CONCAVE."""
    field = inference.orientation_field(shaded_cube_frame())
    old = inference.analyse_faces(field, CubeState.CONCAVE, convention="faces")
    assert [r.ternary for r in old] == [-1, -1, -1]


def test_unknown_convention_rejected():
    with pytest.raises(ValueError):
        inference.regions(CubeState.CONVEX, "sideways")


# ----------------------------------------------------------------------------- source-pixel margin

def test_source_margin_frac_scales_with_cell_size():
    small = inference.source_margin_frac(wall_width_px=3.0, radius_px=25.0, buffer_px=4.0)
    large = inference.source_margin_frac(wall_width_px=3.0, radius_px=200.0, buffer_px=4.0)
    assert small == pytest.approx(7.0 / 25.0)
    assert large == pytest.approx(7.0 / 200.0)
    assert inference.source_margin_frac(3.0, 25.0) == pytest.approx((3.0 + inference.SOURCE_MARGIN_BUFFER_PX) / 25.0)
    assert small > inference.DEFAULT_EDGE_MARGIN_FRAC > large
    assert inference.source_margin_frac(3.0, 0.0) == inference.DEFAULT_EDGE_MARGIN_FRAC


# ----------------------------------------------------------------------------- texture lock

def test_texture_lock_on_axis_stripes_vs_oblique():
    on_axis = inference.texture_lock(inference.orientation_field(stripes(np.deg2rad(90.0))))
    oblique = inference.texture_lock(inference.orientation_field(stripes(np.deg2rad(97.5))))
    assert on_axis.order > 0.9 and on_axis.offset_deg < 0.5
    assert oblique.offset_deg == pytest.approx(7.5, abs=0.5) and oblique.p_value == pytest.approx(0.5, abs=0.05)
    assert not oblique.locked
    flat = inference.texture_lock(inference.orientation_field(np.full((SIZE, SIZE), 128, np.uint8)))
    assert flat.order == 0.0 and flat.p_value == 1.0 and not flat.locked


def test_pooled_lock_test():
    from hexengine.types import TextureLock
    aligned = [TextureLock(order=0.8, phase_deg=float(d), offset_deg=abs(d), p_value=abs(d) / 15, locked=False)
               for d in (0.5, -0.8, 0.3, 1.0, -0.2, 0.6)]
    res = inference.pooled_lock_test(aligned, n_shuffle=2000)
    assert res["n"] == 6 and res["L"] > 0.9 and res["p"] < 0.01 and res["locked"]
    rng = np.random.default_rng(3)
    scattered = [TextureLock(order=0.8, phase_deg=float(d), offset_deg=0, p_value=1, locked=False)
                 for d in rng.uniform(-30, 30, 30)]
    assert inference.pooled_lock_test(scattered, n_shuffle=2000)["p"] > 0.01
    assert inference.pooled_lock_test(aligned[:1]) is None
