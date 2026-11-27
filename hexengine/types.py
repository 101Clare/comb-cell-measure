"""Shared data types passed between hexengine modules.

Kept deliberately plain (dataclasses of numpy arrays and floats) so results are easy to inspect,
serialise to JSON and print from a notebook.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class CubeState(Enum):
    """The two dual (Necker) readings of a hexagon as an isometric cube."""

    CONVEX = 1    # vertex-out: centre is the near corner, primary sign +1
    CONCAVE = -1  # vertex-in: centre is the far / interior corner, primary sign -1

    @property
    def sign(self) -> int:
        return self.value


class TransformKind(str, Enum):
    AFFINE = "affine"
    HOMOGRAPHY = "homography"
    SIMILARITY = "similarity"


@dataclass
class HexFit:
    """Result of fitting one observed hexagon to the canonical hexagon.

    image_points[i] corresponds to canonical vertex V_i (after ordering + rotation offset).
    All matrices map image pixels -> canonical units (3x3, homogeneous).
    """

    image_points: np.ndarray          # (6, 2) float64, reordered
    order: np.ndarray                 # (6,) int, indices into the user's original click list
    rotation_offset: int              # 0..5 as applied
    affine: np.ndarray                # (3, 3)
    homography: np.ndarray            # (3, 3)
    similarity: np.ndarray            # (3, 3)

    def matrix(self, kind: TransformKind | str) -> np.ndarray:
        return getattr(self, TransformKind(kind).value)


@dataclass
class Scores:
    """Residual / symmetry-deviation scores, all in canonical units (circumradius = 1)."""

    affine_mean: float
    affine_max: float
    homography_mean: float
    homography_max: float
    similarity_mean: float
    similarity_max: float
    central_symmetry_error: float     # 0 = perfectly centrally symmetric (opposite sides parallel & equal)
    per_vertex: dict[str, list[float]] = field(default_factory=dict)  # kind -> 6 residuals
    null_percentile: dict[str, float] = field(default_factory=dict)   # metric -> % of random hexagons scoring LOWER (better)
    verdict: str = ""                 # "regular" | "irregular" | "random-like" (noise-floor calibrated, see metrics)
    verdict_basis: str = "affine"     # which transform's mean residual the verdict was judged on
    noise_floor: float = float("nan")  # residual a perfect hexagon of this size would show from marking error alone
    excess: float = float("nan")      # verdict-basis residual minus noise_floor (> 0 = real deviation)
    radius_px: float = float("nan")   # hexagon size in image px
    corner_sigma_px: float = float("nan")  # corner uncertainty used for the floor
    sigma_source: str = ""            # "refined" (measured by refine) or "assumed" (hand clicks)
    threshold_verdict: str = ""       # the original fixed-threshold verdict ("strong" | "weak" | "noise"), for reference


@dataclass
class Polyline:
    """A drawable path in canonical units."""

    name: str
    points: np.ndarray                # (N, 2) canonical units
    hidden: bool = False              # True -> draw dashed (behind the cube for this Necker state)
    closed: bool = False
    arrows: bool = False              # draw direction arrows along the path


@dataclass
class Face:
    """One visible cube face (rhombus) in canonical units for a given CubeState."""

    name: str
    polygon: np.ndarray               # (4, 2): centre, then three outer vertices in order
    outer_vertex_ids: tuple[int, int, int]
    radial_theta: float               # orientation [0, pi) of long diagonal (centre -> middle outer vertex)
    tangential_theta: float           # orientation [0, pi) of short diagonal
    axis_ids: tuple[int, int]         # the two spoke vertex ids bounding this face


@dataclass
class OrientationField:
    """Dense structure-tensor field over the canonical frame (pixel grid, y down)."""

    theta: np.ndarray                 # (H, W) dominant LINE orientation in [0, pi) (perpendicular to gradient)
    coherence: np.ndarray             # (H, W) in [0, 1]
    energy: np.ndarray                # (H, W) gradient energy (trace of tensor)


@dataclass
class FaceResult:
    face: Face
    dominant_theta: float             # [0, pi)
    coherence: float                  # [0, 1]
    ternary: int                      # +1 / 0 / -1
    radial_alignment: float           # |cos| of angle to radial line, [0, 1]
    tangential_alignment: float       # |cos| of angle to tangential line, [0, 1]


@dataclass
class PrimaryLine:
    start: np.ndarray                 # (2,) canonical units (centre)
    end: np.ndarray                   # (2,) canonical units (outer vertex)
    theta: float                      # orientation [0, pi)
    ternary: int
    angle_vs_reference_deg: float     # signed angle from the neutral vertical reference V3->V0
    face_name: str


@dataclass
class TextureLock:
    """Does the texture inside one cell line up with that cell's own hexagon directions?"""

    order: float                      # |Z| in [0, 1]: how strongly the texture is six-fold at all
    phase_deg: float                  # where the six-fold pattern sits: 0 = along the cube axes, +-30 = across them
    offset_deg: float                 # distance to the nearest of the 12 hexagon directions, [0, 15]
    p_value: float                    # rotation-shuffle p: chance a randomly rotated cell lines up this closely
    locked: bool                      # p_value < LOCK_P and order >= LOCK_MIN_ORDER


@dataclass
class SharedEdge:
    cell_a: int
    cell_b: int
    edge_a: tuple[int, int]           # canonical vertex ids in cell_a
    edge_b: tuple[int, int]           # canonical vertex ids in cell_b
    image_segment: np.ndarray         # (2, 2) image px


@dataclass
class EdgeContinuity:
    edge: SharedEdge
    theta_a_image: float              # dominant orientation of the adjacent face in cell a, mapped to image space
    theta_b_image: float
    delta_deg: float                  # undirected angle difference [0, 90]
    score: float                      # |cos(delta)| in [0, 1]; 1 = perfectly continuous
    ternary_a: int
    ternary_b: int
    relation: str                     # "same" | "flipped" | "unrelated"
