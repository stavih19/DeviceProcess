from __future__ import annotations

from AlgoSteps.debug_utils import debug_print_context

from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MatplotlibPath
from matplotlib.patches import Polygon, Rectangle
from numpy.typing import NDArray
from scipy import ndimage as ndi
from scipy.signal import find_peaks

try:
    from AlgoSteps.step1_pivot_candidates import (
        BoolArray,
        BoundingBox,
        FloatArray,
    )
    from AlgoSteps.step3_pivot_segmentation import PivotSegmentationResult
except ImportError:
    from step1_pivot_candidates import (
        BoolArray,
        BoundingBox,
        FloatArray,
    )
    from step3_pivot_segmentation import PivotSegmentationResult


SideName = Literal["left", "right"]
OuterRegionType = Literal["neighbor", "background", "unknown"]


@dataclass(frozen=True)
class XpanderSegmentationConfig:
    """
    Stage 6 configuration.

    Geometry followed by the detector:

    1. Leave the Pivot to the left and right.
    2. Find a side-wall edge cluster on each side.
    3. Continue through the wall's shadow/halo and move upward.
    4. Find the first horizontal corner-edge cluster above the Pivot and cross
       the whole cluster, so the selected reference corner is beyond the halo.
    5. From the left reference corner continue LEFT.
       From the right reference corner continue RIGHT.
    6. Ignore the initial halo around the reference corner, find a stable
       Xpander plateau, and then detect:

           Xpander -> raised threshold -> neighbor/background

       The Xpander-to-threshold edge is the bottom Xpander corner.
    7. From both bottom corners move upward (decreasing Y), slightly inside the
       Xpander. After establishing a stable Xpander plateau, detect the ordered
       top-edge signature:

           Xpander -> small rise -> stronger fall [-> optional recovery rise]

       The FIRST, small rise is the top Xpander bounding-box boundary. The
       stronger fall is the required validation that the path entered the sill.
       A later recovery rise is optional: when present it significantly
       strengthens the candidate, but its absence does not reject it.
    """

    # Mild 2-D and 1-D smoothing. Strong smoothing can enlarge the halo.
    gaussian_sigma: float = 0.75
    profile_gaussian_sigma: float = 0.35
    profile_band_half_width: int = 3

    # Bottom horizontal profile aggregation.
    bottom_profile_band_half_width: int = 2
    bottom_profile_band_percentile: float = 75.0

    # Side-wall search outside the Pivot.
    side_wall_min_gap_fraction: float = 0.03
    side_wall_max_distance_factor: float = 3.0
    side_wall_gradient_mad_multiplier: float = 2.5
    side_wall_min_prominence: float = 0.01

    # Legacy threshold retained for the older single-reference helper.
    # The active multi-candidate detector below no longer hard-filters wall
    # clusters only by their strength relative to the strongest wall.
    side_wall_cluster_min_relative_strength: float = 0.25

    # Active side-wall candidate ranking. Strength and geometric proximity are
    # both soft evidence. A farther edge can still win when it is strong enough,
    # and a nearby edge does not win merely because it is close.
    reference_wall_strength_weight: float = 0.45
    reference_wall_distance_weight: float = 0.55
    reference_wall_distance_scale_fraction: float = 0.75
    reference_wall_min_strength_score: float = 0.05
    reference_wall_strength_reference_percentile: float = 90.0

    # Reference-corner search above the Pivot.
    reference_corner_gap_fraction: float = 0.04
    reference_corner_max_distance_factor: float = 3.0
    reference_corner_gradient_mad_multiplier: float = 2.5
    reference_corner_min_prominence: float = 0.01
    reference_corner_cluster_min_relative_strength: float = 0.25

    reference_wall_max_candidates: int = 3
    reference_corner_max_candidates: int = 6
    reference_total_max_candidates: int = 12

    # Nearby parallel edges are treated as one shadow/halo cluster.
    # The algorithm deliberately selects the far edge of the first cluster in
    # the direction of travel.
    halo_cluster_max_gap_pixels: int = 16
    reference_departure_pixels: int = 2

    # Stable Xpander plateau used to get beyond the reference-corner halo.
    stable_window_pixels: int = 8
    stable_search_max_fraction: float = 0.35
    stable_max_std_fraction: float = 0.004
    stable_max_derivative_fraction: float = 0.0025
    stable_minimum_skip_pixels: int = 2

    # Raised-threshold detection.
    transition_gradient_mad_multiplier: float = 2.0
    transition_min_gradient: float = 0.005
    transition_min_prominence: float = 0.005
    transition_probe_pixels: int = 5
    threshold_min_width_pixels: int = 1
    threshold_max_width_pixels: int = 60
    threshold_max_width_fraction: float = 0.25
    minimum_threshold_rise_fraction: float = 0.0015
    minimum_threshold_rise_absolute: float = 0.02

    # Search several horizontal profiles around the approximate
    # reference-corner Y coordinate.
    bottom_profile_y_search_radius: int = 20
    bottom_profile_y_search_step: int = 2
    bottom_profile_y_distance_weight: float = 0.10

    # The region before the threshold must still resemble the stable Xpander
    # plateau. This prevents an outer shadow edge from being selected.
    xpander_plateau_tolerance_fraction: float = 0.025
    side_region_max_std_fraction: float = 0.02

    # Top search starts slightly inside the detected side boundaries.
    top_search_horizontal_inset_fraction: float = 0.03
    top_search_minimum_inset_pixels: int = 3
    top_search_start_gap_pixels: int = 4

    # Top-edge signature while travelling upward:
    # stable Xpander -> small rise -> stronger fall [-> optional recovery rise].
    # The first small rise is the actual top BB boundary. The fall is required;
    # recovery is only a confidence boost.
    top_initial_rise_min_fraction: float = 0.00035
    top_initial_rise_min_absolute: float = 0.006
    top_main_fall_min_fraction: float = 0.0015
    top_main_fall_min_absolute: float = 0.02
    top_recovery_rise_min_fraction: float = 0.0010
    top_recovery_rise_min_absolute: float = 0.015
    top_recovery_score_bonus: float = 0.20
    top_main_fall_to_initial_rise_ratio: float = 1.10
    top_pattern_min_gap_pixels: int = 1
    top_pattern_max_gap_pixels: int = 55
    top_pattern_probe_pixels: int = 5
    top_pattern_max_after_std_fraction: float = 0.025

    # The recovered region is expected to resemble background, but this is used
    # primarily as a score rather than as a hard rejection rule.
    background_tolerance_fraction: float = 0.08
    background_probe_pixels: int = 8
    background_max_std_fraction: float = 0.025

    # Geometry validation.
    maximum_bottom_corner_y_difference_fraction: float = 0.12
    maximum_top_corner_y_difference_fraction: float = 0.12
    minimum_xpander_width_fraction_of_pivot: float = 0.50
    minimum_xpander_height_fraction_of_pivot: float = 0.50

    # Final confidence.
    transition_score_weight: float = 0.60
    reference_score_weight: float = 0.20
    geometry_score_weight: float = 0.20
    minimum_confidence: float = 0.45

    bottom_profile_y_penalty_per_pixel: float = 0.025

    # Cross-side geometry used when selecting the LEFT/RIGHT pair jointly.
    # These are soft scores, not hard rejection limits.
    reference_y_alignment_tolerance_pixels: float = 8.0
    bottom_y_alignment_tolerance_pixels: float = 8.0
    outward_distance_symmetry_tolerance_pixels: float = 20.0

    # Keep several bottom hypotheses for every reference corner.  This is
    # important because a locally strong transition can be the wrong inner
    # edge; the correct outer Xpander corner may have a slightly lower local
    # transition score but form a much more consistent LEFT/RIGHT pair.
    bottom_candidates_per_reference: int = 5
    reference_bottom_total_max_candidates: int = 40
    pair_debug_max_candidates: int = 24

    # Final LEFT/RIGHT pair score. No single geometric cue is a hard rule.
    # In particular, outward-distance symmetry is useful but deliberately kept
    # weak: neighboring components can also form a very symmetric pair.
    # The additional surface-height term asks whether the plateau before the
    # detected threshold actually resembles the Xpander surface surrounding
    # the Pivot.
    pair_transition_weight: float = 0.35
    pair_reference_weight: float = 0.30
    pair_bottom_y_alignment_weight: float = 0.20
    pair_reference_y_alignment_weight: float = 0.03
    pair_outward_distance_symmetry_weight: float = 0.02
    pair_surface_height_weight: float = 0.10

    # Expected Xpander-surface consistency. The expected surface height is
    # derived from the OUTSIDE heights of the Pivot's left/right/top walls.
    # The bottom Pivot wall is intentionally excluded because below it is the
    # lower structure/background rather than the surrounding Xpander plateau.
    pair_surface_height_tolerance_fraction: float = 0.04
    pair_surface_height_tolerance_absolute: float = 0.50
    pair_surface_height_free_band_multiplier: float = 1.50

    # After the top raised strip, the profile must fall clearly below
    # the Xpander plateau. This rejects small bumps that return to the
    # same Xpander surface.
    top_post_fall_drop_min_fraction: float = 0.01
    top_post_fall_drop_min_absolute: float = 0.10

    top_min_progress_toward_background_fraction: float = 0.10


@dataclass(frozen=True)
class PixelPoint:
    x: int
    y: int


@dataclass(frozen=True)
class ReferenceCorner:
    """Structural corner selected after crossing the side shadow/halo."""

    side: SideName
    point: PixelPoint
    wall_x: int
    wall_score: float
    corner_score: float
    wall_cluster_start: int
    wall_cluster_end: int
    corner_cluster_start: int
    corner_cluster_end: int
    wall_was_inferred: bool
    corner_was_inferred: bool

    @property
    def score(self) -> float:
        return float(0.5 * self.wall_score + 0.5 * self.corner_score)


@dataclass(frozen=True)
class PathDiagnostics:
    """A profile stored in travel order for plotting and debugging."""

    positions: NDArray[np.int64]
    profile: NDArray[np.float64]
    stable_start_index: int
    stable_stop_index: int
    stable_height: float


@dataclass(frozen=True)
class ThresholdTransition:
    """Raised threshold found while moving in the requested direction."""

    rise_position: int
    fall_position: int
    rise_path_index: int
    fall_path_index: int

    before_height: float
    threshold_height: float
    after_height: float
    before_std: float
    after_std: float

    rise_strength: float
    fall_strength: float
    threshold_width_pixels: int
    threshold_rise: float
    score: float

    # Used by the top-edge detector for the required stronger fall.
    validation_fall_position: int | None = None
    validation_fall_path_index: int | None = None

    # Optional recovery rise after the sill. It strengthens confidence but is
    # not required for accepting the top boundary.
    validation_recovery_position: int | None = None
    validation_recovery_path_index: int | None = None


@dataclass(frozen=True)
class SideCornerDetection:
    """Bottom Xpander corner found by continuing outward from a reference."""

    side: SideName
    reference_corner: ReferenceCorner
    bottom_corner: PixelPoint
    transition: ThresholdTransition
    outer_region_type: OuterRegionType
    outer_height: float
    xpander_height: float
    diagnostics: PathDiagnostics


@dataclass(frozen=True)
class BottomProfileCandidate:
    """One valid bottom-corner hypothesis from one horizontal scan row."""

    detection: SideCornerDetection
    selection_score: float
    y_distance: int
    y_offset: int
    phase_name: str


@dataclass(frozen=True)
class ReferenceBottomCandidate:
    """A reference corner together with one valid bottom-corner hypothesis."""

    side: SideName
    reference: ReferenceCorner
    bottom: BottomProfileCandidate
    local_score: float

    @property
    def detection(self) -> SideCornerDetection:
        return self.bottom.detection

    @property
    def outward_distance_pixels(self) -> int:
        if self.side == "left":
            return int(
                self.reference.point.x
                - self.detection.bottom_corner.x
            )
        return int(
            self.detection.bottom_corner.x
            - self.reference.point.x
        )


@dataclass(frozen=True)
class ReferenceBottomPairSelection:
    """Jointly selected LEFT/RIGHT reference+bottom pair and diagnostics."""

    left: ReferenceBottomCandidate
    right: ReferenceBottomCandidate
    score: float
    mean_transition_score: float
    mean_reference_score: float
    reference_y_difference: int
    reference_y_alignment_score: float
    bottom_y_difference: int
    bottom_y_alignment_score: float
    outward_distance_difference: int
    outward_distance_symmetry_score: float

    expected_xpander_surface_height: float
    surface_height_tolerance: float
    left_surface_height_difference: float
    right_surface_height_difference: float
    left_surface_height_score: float
    right_surface_height_score: float
    surface_height_score: float
    left_right_surface_height_difference: float
    left_right_surface_consistency_score: float


@dataclass(frozen=True)
class TopCornerDetection:
    """Top Xpander corner found by moving upward from a bottom corner."""

    side: SideName
    bottom_corner: PixelPoint
    top_corner: PixelPoint
    search_x: int
    transition: ThresholdTransition
    background_height_after_threshold: float
    background_match_score: float
    diagnostics: PathDiagnostics

    @property
    def score(self) -> float:
        return float(
            0.75 * self.transition.score
            + 0.25 * self.background_match_score
        )


@dataclass
class XpanderSegmentationResult:
    """Complete Stage-6 result."""

    bounding_box: BoundingBox
    xpander_mask: BoolArray

    bottom_left: PixelPoint
    bottom_right: PixelPoint
    top_left: PixelPoint
    top_right: PixelPoint

    left_reference_corner: ReferenceCorner
    right_reference_corner: ReferenceCorner
    left_side_detection: SideCornerDetection
    right_side_detection: SideCornerDetection
    left_top_detection: TopCornerDetection
    right_top_detection: TopCornerDetection

    background_height: float
    robust_height_range: float
    confidence: float
    is_confident: bool

    smoothed_height_map: NDArray[np.float32]
    gradient_x: NDArray[np.float32]
    gradient_y: NDArray[np.float32]

    @property
    def corners(self) -> tuple[PixelPoint, PixelPoint, PixelPoint, PixelPoint]:
        return (
            self.top_left,
            self.top_right,
            self.bottom_right,
            self.bottom_left,
        )


def segment_xpander(
    height_map: FloatArray,
    pivot_segmentation: PivotSegmentationResult,
    config: XpanderSegmentationConfig | None = None,
    show_corner_debug: bool = False,
) -> XpanderSegmentationResult:
    """Detect and label the Xpander using the corrected outward geometry."""
    if config is None:
        config = XpanderSegmentationConfig()

    _validate_inputs(height_map, pivot_segmentation, config)

    height_map_float = height_map.astype(np.float32, copy=False)
    pivot_box = pivot_segmentation.bounding_box

    smoothed = ndi.gaussian_filter(
        height_map_float,
        sigma=config.gaussian_sigma,
    ).astype(np.float32)

    gradient_x = (ndi.sobel(smoothed, axis=1) / 8.0).astype(np.float32)
    gradient_y = (ndi.sobel(smoothed, axis=0) / 8.0).astype(np.float32)

    robust_height_range = float(
        np.percentile(smoothed, 99) - np.percentile(smoothed, 1)
    )
    robust_height_range = max(robust_height_range, 1e-8)

    background_height = _estimate_background_height(
        smoothed=smoothed,
        gradient_x=gradient_x,
        gradient_y=gradient_y,
    )

    detected_corners: list[tuple[str, PixelPoint]] = []

    # Detect reference candidates on both sides first.  Do not commit to one
    # side independently: the final reference+bottom decision is made jointly
    # so a locally strong inner transition cannot lock the other side into a
    # geometrically inconsistent solution.
    try:
        left_reference_candidates = _find_reference_corner_candidates(
            side="left",
            gradient_x=gradient_x,
            gradient_y=gradient_y,
            pivot_box=pivot_box,
            config=config,
        )
        right_reference_candidates = _find_reference_corner_candidates(
            side="right",
            gradient_x=gradient_x,
            gradient_y=gradient_y,
            pivot_box=pivot_box,
            config=config,
        )
    except ValueError as error:
        raise ValueError(
            f"Xpander reference-corner search failed: {error}"
        ) from error

    try:
        left_reference_bottom_candidates = (
            _collect_reference_bottom_candidates(
                side="left",
                smoothed=smoothed,
                reference_candidates=left_reference_candidates,
                background_height=background_height,
                robust_height_range=robust_height_range,
                config=config,
            )
        )
    except ValueError as error:
        raise ValueError(
            "Left Xpander reference/bottom candidate search failed: "
            f"{error}"
        ) from error

    try:
        right_reference_bottom_candidates = (
            _collect_reference_bottom_candidates(
                side="right",
                smoothed=smoothed,
                reference_candidates=right_reference_candidates,
                background_height=background_height,
                robust_height_range=robust_height_range,
                config=config,
            )
        )
    except ValueError as error:
        raise ValueError(
            "Right Xpander reference/bottom candidate search failed: "
            f"{error}"
        ) from error

    # Stage 3 already measured the height immediately OUTSIDE the Pivot walls.
    # Left, right and top all border the Xpander surface, so their robust median
    # is a useful physical reference for rejecting false bottom transitions
    # that begin on a lower shadow/neighbor plateau. The bottom wall is excluded
    # because its outside region belongs to a different vertical structure.
    expected_xpander_surface_height = float(
        np.median(
            np.asarray(
                [
                    pivot_segmentation.left_boundary.outside_height,
                    pivot_segmentation.right_boundary.outside_height,
                    pivot_segmentation.top_boundary.outside_height,
                ],
                dtype=np.float64,
            )
        )
    )

    print(
        "Expected Xpander surface height from Pivot outside walls: "
        f"left={pivot_segmentation.left_boundary.outside_height:.6f}, "
        f"right={pivot_segmentation.right_boundary.outside_height:.6f}, "
        f"top={pivot_segmentation.top_boundary.outside_height:.6f}, "
        f"median={expected_xpander_surface_height:.6f}"
    )

    pair_selection = _select_reference_bottom_pair(
        left_candidates=left_reference_bottom_candidates,
        right_candidates=right_reference_bottom_candidates,
        pivot_box=pivot_box,
        expected_xpander_surface_height=expected_xpander_surface_height,
        robust_height_range=robust_height_range,
        config=config,
    )

    left_reference = pair_selection.left.reference
    left_side = pair_selection.left.detection
    right_reference = pair_selection.right.reference
    right_side = pair_selection.right.detection

    # Show only the finally selected geometry.  Candidate details remain in the
    # textual debug log so plots stay readable.
    detected_corners.append(("Left reference", left_reference.point))
    if show_corner_debug:
        _show_detected_xpander_corners(
            height_map=height_map,
            pivot_box=pivot_box,
            detected_corners=detected_corners,
            title="Selected left reference corner",
        )

    detected_corners.append(("Left bottom", left_side.bottom_corner))
    if show_corner_debug:
        _show_detected_xpander_corners(
            height_map=height_map,
            pivot_box=pivot_box,
            detected_corners=detected_corners,
            title="Left bottom Xpander corner found",
        )

    detected_corners.append(("Right reference", right_reference.point))
    if show_corner_debug:
        _show_detected_xpander_corners(
            height_map=height_map,
            pivot_box=pivot_box,
            detected_corners=detected_corners,
            title="Selected right reference corner",
        )

    detected_corners.append(("Right bottom", right_side.bottom_corner))
    if show_corner_debug:
        _show_detected_xpander_corners(
            height_map=height_map,
            pivot_box=pivot_box,
            detected_corners=detected_corners,
            title="Right bottom Xpander corner found",
        )

    bottom_left = left_side.bottom_corner
    bottom_right = right_side.bottom_corner

    if bottom_left.x >= bottom_right.x:
        raise ValueError(
            "Detected bottom corners are invalid: "
            f"left={bottom_left}, right={bottom_right}."
        )

    xpander_width = bottom_right.x - bottom_left.x
    inset = max(
        config.top_search_minimum_inset_pixels,
        int(round(xpander_width * config.top_search_horizontal_inset_fraction)),
    )

    try:
        left_top = _find_top_xpander_corner(
            side="left",
            smoothed=smoothed,
            bottom_corner=bottom_left,
            search_x=min(bottom_right.x - 1, bottom_left.x + inset),
            background_height=background_height,
            robust_height_range=robust_height_range,
            config=config,
        )
    except ValueError as error:
        raise ValueError(f"Left top Xpander corner failed: {error}") from error
    detected_corners.append(("Left top", left_top.top_corner))
    if show_corner_debug:
        _show_detected_xpander_corners(
            height_map=height_map,
            pivot_box=pivot_box,
            detected_corners=detected_corners,
            title="Left top Xpander corner found",
        )

    try:
        right_top = _find_top_xpander_corner(
            side="right",
            smoothed=smoothed,
            bottom_corner=bottom_right,
            search_x=max(bottom_left.x + 1, bottom_right.x - inset),
            background_height=background_height,
            robust_height_range=robust_height_range,
            config=config,
        )
    except ValueError as error:
        raise ValueError(f"Right top Xpander corner failed: {error}") from error
    detected_corners.append(("Right top", right_top.top_corner))
    if show_corner_debug:
        _show_detected_xpander_corners(
            height_map=height_map,
            pivot_box=pivot_box,
            detected_corners=detected_corners,
            title="Right top Xpander corner found",
        )

    top_left = PixelPoint(x=bottom_left.x, y=left_top.top_corner.y)
    top_right = PixelPoint(x=bottom_right.x, y=right_top.top_corner.y)

    _validate_detected_geometry(
        pivot_box=pivot_box,
        top_left=top_left,
        top_right=top_right,
        bottom_left=bottom_left,
        bottom_right=bottom_right,
        config=config,
    )

    xpander_mask = _polygon_to_mask(
        image_shape=height_map.shape,
        polygon_points=(top_left, top_right, bottom_right, bottom_left),
    )
    xpander_mask &= ~pivot_segmentation.pivot_mask

    mask_y, mask_x = np.nonzero(xpander_mask)
    if mask_x.size == 0:
        raise ValueError("The detected Xpander polygon produced an empty mask.")

    bounding_box = BoundingBox(
        x_min=int(mask_x.min()),
        y_min=int(mask_y.min()),
        x_max=int(mask_x.max()) + 1,
        y_max=int(mask_y.max()) + 1,
    )

    transition_score = float(
        np.mean(
            [
                left_side.transition.score,
                right_side.transition.score,
                left_top.score,
                right_top.score,
            ]
        )
    )
    reference_score = float(
        np.mean([left_reference.score, right_reference.score])
    )
    base_geometry_score = _calculate_geometry_score(
        pivot_box=pivot_box,
        top_left=top_left,
        top_right=top_right,
        bottom_left=bottom_left,
        bottom_right=bottom_right,
        config=config,
    )

    # Include the cross-side consistency used to choose the bottom corners in
    # the final geometry confidence.  This prevents a high-confidence result
    # when both local transitions are strong but the pair geometry is poor.
    geometry_score = float(
        0.75 * base_geometry_score
        + 0.25
        * np.mean(
            [
                pair_selection.reference_y_alignment_score,
                pair_selection.bottom_y_alignment_score,
                pair_selection.outward_distance_symmetry_score,
                pair_selection.surface_height_score,
            ]
        )
    )

    confidence = float(
        config.transition_score_weight * transition_score
        + config.reference_score_weight * reference_score
        + config.geometry_score_weight * geometry_score
    )

    return XpanderSegmentationResult(
        bounding_box=bounding_box,
        xpander_mask=xpander_mask,
        bottom_left=bottom_left,
        bottom_right=bottom_right,
        top_left=top_left,
        top_right=top_right,
        left_reference_corner=left_reference,
        right_reference_corner=right_reference,
        left_side_detection=left_side,
        right_side_detection=right_side,
        left_top_detection=left_top,
        right_top_detection=right_top,
        background_height=background_height,
        robust_height_range=robust_height_range,
        confidence=confidence,
        is_confident=confidence >= config.minimum_confidence,
        smoothed_height_map=smoothed,
        gradient_x=gradient_x,
        gradient_y=gradient_y,
    )

def _get_strong_clusters(
    clusters: list[list[tuple[int, float]]],
    minimum_relative_strength: float,
) -> list[list[tuple[int, float]]]:
    """
    Keep clusters whose strongest peak is sufficiently strong relative
    to the strongest peak among all clusters.
    """
    if not clusters:
        return []

    maximum_strength = max(
        strength
        for cluster in clusters
        for _, strength in cluster
    )

    minimum_strength = (
        minimum_relative_strength
        * maximum_strength
    )

    strong_clusters = [
        cluster
        for cluster in clusters
        if max(
            strength
            for _, strength in cluster
        ) >= minimum_strength
    ]

    return strong_clusters or clusters


def _find_reference_corner(
    side: SideName,
    gradient_x: NDArray[np.float32],
    gradient_y: NDArray[np.float32],
    pivot_box: BoundingBox,
    config: XpanderSegmentationConfig,
) -> ReferenceCorner:
    """
    Leave the Pivot in the requested direction, find the nearest side-wall
    edge cluster, cross its halo, and then move upward to the nearest corner
    cluster above the Pivot. The far edge of each cluster is selected.
    """
    image_height, image_width = gradient_x.shape

    side_gap = max(1, int(round(pivot_box.width * config.side_wall_min_gap_fraction)))
    max_side_distance = max(
        pivot_box.width,
        int(round(pivot_box.width * config.side_wall_max_distance_factor)),
    )

    y_start = max(0, pivot_box.y_min)
    y_stop = min(image_height, pivot_box.y_max)
    if y_stop <= y_start:
        raise ValueError("Pivot vertical range is invalid for side-wall search.")

    vertical_strength = np.median(
        np.abs(gradient_x[y_start:y_stop, :]),
        axis=0,
    )

    if side == "left":
        search_start = max(0, pivot_box.x_min - max_side_distance)
        search_stop = max(search_start + 1, pivot_box.x_min - side_gap)
    else:
        search_start = min(image_width - 1, pivot_box.x_max + side_gap)
        search_stop = min(image_width, pivot_box.x_max + max_side_distance)

    wall_positions, wall_strengths = _find_profile_peaks(
        profile=vertical_strength,
        start=search_start,
        stop=search_stop,
        mad_multiplier=config.side_wall_gradient_mad_multiplier,
        minimum_prominence=config.side_wall_min_prominence,
    )

    wall_was_inferred = False
    if wall_positions:
        wall_clusters = _cluster_peaks(
            wall_positions,
            wall_strengths,
            max_gap=config.halo_cluster_max_gap_pixels,
        )
        wall_cluster = _nearest_strong_cluster_to_pivot(
            clusters=wall_clusters,
            side=side,
            pivot_box=pivot_box,
            minimum_relative_strength=(
                config.side_wall_cluster_min_relative_strength
            ),
        )
        wall_cluster_positions = [item[0] for item in wall_cluster]
        # Continue through the halo in the outward direction.
        wall_x = (
            min(wall_cluster_positions)
            if side == "left"
            else max(wall_cluster_positions)
        )
        wall_score = _cluster_score(wall_cluster, wall_strengths)
        wall_cluster_start = min(wall_cluster_positions)
        wall_cluster_end = max(wall_cluster_positions)
    else:
        wall_was_inferred = True
        segment = vertical_strength[search_start:search_stop]
        if segment.size == 0:
            raise ValueError("No valid side-wall search interval remains.")
        wall_x = search_start + int(np.argmax(segment))
        wall_score = 0.15
        wall_cluster_start = wall_x
        wall_cluster_end = wall_x

    half_band = config.profile_band_half_width
    wall_band_start = max(0, wall_x - half_band)
    wall_band_stop = min(image_width, wall_x + half_band + 1)
    horizontal_strength = np.median(
        np.abs(gradient_y[:, wall_band_start:wall_band_stop]),
        axis=1,
    )

    corner_gap = max(
        1,
        int(round(pivot_box.height * config.reference_corner_gap_fraction)),
    )
    max_corner_distance = max(
        pivot_box.height,
        int(round(pivot_box.height * config.reference_corner_max_distance_factor)),
    )
    corner_search_start = max(0, pivot_box.y_min - max_corner_distance)
    corner_search_stop = max(corner_search_start + 1, pivot_box.y_min - corner_gap)

    corner_positions, corner_strengths = _find_profile_peaks(
        profile=horizontal_strength,
        start=corner_search_start,
        stop=corner_search_stop,
        mad_multiplier=config.reference_corner_gradient_mad_multiplier,
        minimum_prominence=config.reference_corner_min_prominence,
    )

    corner_was_inferred = False
    if corner_positions:
        corner_clusters = _cluster_peaks(
            corner_positions,
            corner_strengths,
            max_gap=config.halo_cluster_max_gap_pixels,
        )
        # Moving upward from the Pivot means the first cluster is the one with
        # the largest Y coordinate. Cross the complete halo by taking the
        # smallest Y coordinate inside that first cluster.
        corner_cluster = _nearest_strong_cluster_above_pivot(
            clusters=corner_clusters,
            pivot_y=pivot_box.y_min,
            minimum_relative_strength=(
                config.reference_corner_cluster_min_relative_strength
            ),
        )
        corner_cluster_positions = [item[0] for item in corner_cluster]
        corner_y = min(corner_cluster_positions)
        corner_score = _cluster_score(corner_cluster, corner_strengths)
        corner_cluster_start = min(corner_cluster_positions)
        corner_cluster_end = max(corner_cluster_positions)
    else:
        corner_was_inferred = True
        segment = horizontal_strength[corner_search_start:corner_search_stop]
        if segment.size == 0:
            raise ValueError("No valid reference-corner search interval remains.")
        corner_y = corner_search_start + int(np.argmax(segment))
        corner_score = 0.15
        corner_cluster_start = corner_y
        corner_cluster_end = corner_y

    return ReferenceCorner(
        side=side,
        point=PixelPoint(x=int(wall_x), y=int(corner_y)),
        wall_x=int(wall_x),
        wall_score=float(wall_score),
        corner_score=float(corner_score),
        wall_cluster_start=int(wall_cluster_start),
        wall_cluster_end=int(wall_cluster_end),
        corner_cluster_start=int(corner_cluster_start),
        corner_cluster_end=int(corner_cluster_end),
        wall_was_inferred=wall_was_inferred,
        corner_was_inferred=corner_was_inferred,
    )

def _find_reference_corner_candidates(
    side: SideName,
    gradient_x: NDArray[np.float32],
    gradient_y: NDArray[np.float32],
    pivot_box: BoundingBox,
    config: XpanderSegmentationConfig,
) -> list[ReferenceCorner]:
    """
    Return several possible reference corners.

    Side-wall clusters are ranked by a soft combination of:

        * edge strength;
        * distance from the Pivot.

    This replaces the previous hard rule that discarded a wall only because
    it was weaker than a fixed fraction of the globally strongest wall.
    Distance is a preference, not a hard assumption: a farther wall can still
    rank highly if its edge evidence is sufficiently strong.

    The best wall candidates are then expanded into horizontal corner
    candidates. The caller still performs the important physical validation:
    every reference candidate must lead to a valid bottom Xpander transition.
    """
    image_height, image_width = gradient_x.shape

    side_gap = max(
        1,
        int(
            round(
                pivot_box.width
                * config.side_wall_min_gap_fraction
            )
        ),
    )

    max_side_distance = max(
        pivot_box.width,
        int(
            round(
                pivot_box.width
                * config.side_wall_max_distance_factor
            )
        ),
    )

    y_start = max(0, pivot_box.y_min)
    y_stop = min(image_height, pivot_box.y_max)

    if y_stop <= y_start:
        raise ValueError(
            "Pivot vertical range is invalid "
            "for side-wall search."
        )

    vertical_strength = np.median(
        np.abs(
            gradient_x[
                y_start:y_stop,
                :,
            ]
        ),
        axis=0,
    )

    if side == "left":
        search_start = max(
            0,
            pivot_box.x_min - max_side_distance,
        )
        search_stop = max(
            search_start + 1,
            pivot_box.x_min - side_gap,
        )
    else:
        search_start = min(
            image_width - 1,
            pivot_box.x_max + side_gap,
        )
        search_stop = min(
            image_width,
            pivot_box.x_max + max_side_distance,
        )

    wall_positions, wall_strengths = _find_profile_peaks(
        profile=vertical_strength,
        start=search_start,
        stop=search_stop,
        mad_multiplier=(
            config.side_wall_gradient_mad_multiplier
        ),
        minimum_prominence=(
            config.side_wall_min_prominence
        ),
    )

    # cluster, inferred, wall_score, strength_score, distance_score, distance_px
    wall_cluster_candidates: list[
        tuple[
            list[tuple[int, float]],
            bool,
            float,
            float,
            float,
            float,
        ]
    ] = []

    if wall_positions:
        wall_clusters = _cluster_peaks(
            wall_positions,
            wall_strengths,
            max_gap=config.halo_cluster_max_gap_pixels,
        )

        cluster_strength_values = np.asarray(
            [
                max(
                    strength
                    for _, strength in cluster
                )
                for cluster in wall_clusters
            ],
            dtype=np.float64,
        )

        # A percentile reference is less sensitive than the absolute maximum to
        # one exceptional edge elsewhere in the search interval.
        strength_reference = max(
            float(
                np.percentile(
                    cluster_strength_values,
                    config.reference_wall_strength_reference_percentile,
                )
            ),
            1e-8,
        )

        distance_scale = max(
            1.0,
            config.reference_wall_distance_scale_fraction
            * pivot_box.width,
        )

        scored_wall_clusters: list[
            tuple[
                float,
                list[tuple[int, float]],
                float,
                float,
                float,
                float,
            ]
        ] = []

        for cluster in wall_clusters:
            positions = [
                position
                for position, _ in cluster
            ]

            cluster_strength = float(
                max(
                    strength
                    for _, strength in cluster
                )
            )

            strength_score = float(
                np.clip(
                    cluster_strength / strength_reference,
                    0.0,
                    1.0,
                )
            )

            if side == "left":
                distance_pixels = float(
                    pivot_box.x_min - max(positions)
                )
            else:
                distance_pixels = float(
                    min(positions) - pivot_box.x_max
                )

            distance_pixels = max(distance_pixels, 0.0)

            distance_score = float(
                np.exp(
                    -distance_pixels / distance_scale
                )
            )

            wall_score = float(
                config.reference_wall_strength_weight
                * strength_score
                + config.reference_wall_distance_weight
                * distance_score
            )

            # Very weak safety floor only. The important change is that a wall
            # is no longer rejected because another remote wall is much stronger.
            if (
                strength_score
                < config.reference_wall_min_strength_score
            ):
                print(
                    f"{side.capitalize()} wall cluster rejected by "
                    f"minimum strength floor: "
                    f"cluster=({min(positions)}, {max(positions)}), "
                    f"raw_strength={cluster_strength:.6f}, "
                    f"strength_score={strength_score:.4f}"
                )
                continue

            scored_wall_clusters.append(
                (
                    wall_score,
                    cluster,
                    cluster_strength,
                    strength_score,
                    distance_score,
                    distance_pixels,
                )
            )

        scored_wall_clusters.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        print(
            f"\n{side.upper()} WALL CANDIDATES "
            "(strength + distance soft ranking):"
        )
        print("-" * 135)
        print(
            f"strength_reference(p"
            f"{config.reference_wall_strength_reference_percentile:.0f})="
            f"{strength_reference:.6f}, "
            f"distance_scale={distance_scale:.2f}px, "
            f"weights=(strength="
            f"{config.reference_wall_strength_weight:.2f}, "
            f"distance={config.reference_wall_distance_weight:.2f})"
        )

        for rank, (
            wall_score,
            cluster,
            cluster_strength,
            strength_score,
            distance_score,
            distance_pixels,
        ) in enumerate(scored_wall_clusters, start=1):
            positions = [
                position
                for position, _ in cluster
            ]
            kept_marker = (
                " <-- KEPT"
                if rank <= config.reference_wall_max_candidates
                else ""
            )
            print(
                f"Rank {rank}: "
                f"cluster=({min(positions)}, {max(positions)}), "
                f"raw_strength={cluster_strength:.6f}, "
                f"strength_score={strength_score:.4f}, "
                f"distance={distance_pixels:.1f}px, "
                f"distance_score={distance_score:.4f}, "
                f"wall_score={wall_score:.4f}"
                f"{kept_marker}"
            )
        print("-" * 135)

        for (
            wall_score,
            cluster,
            _cluster_strength,
            strength_score,
            distance_score,
            distance_pixels,
        ) in scored_wall_clusters[
            :config.reference_wall_max_candidates
        ]:
            wall_cluster_candidates.append(
                (
                    cluster,
                    False,
                    wall_score,
                    strength_score,
                    distance_score,
                    distance_pixels,
                )
            )

    else:
        segment = vertical_strength[
            search_start:search_stop
        ]

        if segment.size == 0:
            raise ValueError(
                "No valid side-wall search interval remains."
            )

        inferred_wall_x = (
            search_start
            + int(np.argmax(segment))
        )

        if side == "left":
            inferred_distance = float(
                max(
                    pivot_box.x_min - inferred_wall_x,
                    0,
                )
            )
        else:
            inferred_distance = float(
                max(
                    inferred_wall_x - pivot_box.x_max,
                    0,
                )
            )

        distance_scale = max(
            1.0,
            config.reference_wall_distance_scale_fraction
            * pivot_box.width,
        )
        inferred_distance_score = float(
            np.exp(-inferred_distance / distance_scale)
        )

        wall_cluster_candidates.append(
            (
                [(inferred_wall_x, 0.15)],
                True,
                0.15,
                0.15,
                inferred_distance_score,
                inferred_distance,
            )
        )

    candidates: list[ReferenceCorner] = []

    for (
        wall_cluster,
        wall_was_inferred,
        ranked_wall_score,
        wall_strength_score,
        wall_distance_score,
        wall_distance_pixels,
    ) in wall_cluster_candidates:
        wall_cluster_positions = [
            position
            for position, _ in wall_cluster
        ]

        # Move through the wall halo in the outward direction.
        wall_x = (
            min(wall_cluster_positions)
            if side == "left"
            else max(wall_cluster_positions)
        )

        wall_cluster_start = min(
            wall_cluster_positions
        )
        wall_cluster_end = max(
            wall_cluster_positions
        )

        # The ReferenceCorner wall score now already contains both strength and
        # proximity evidence.
        wall_score = float(ranked_wall_score)

        half_band = config.profile_band_half_width

        wall_band_start = max(
            0,
            wall_x - half_band,
        )
        wall_band_stop = min(
            image_width,
            wall_x + half_band + 1,
        )

        horizontal_strength = np.median(
            np.abs(
                gradient_y[
                    :,
                    wall_band_start:wall_band_stop,
                ]
            ),
            axis=1,
        )

        corner_gap = max(
            1,
            int(
                round(
                    pivot_box.height
                    * config.reference_corner_gap_fraction
                )
            ),
        )

        max_corner_distance = max(
            pivot_box.height,
            int(
                round(
                    pivot_box.height
                    * config.reference_corner_max_distance_factor
                )
            ),
        )

        corner_search_start = max(
            0,
            pivot_box.y_min - max_corner_distance,
        )
        corner_search_stop = max(
            corner_search_start + 1,
            pivot_box.y_min - corner_gap,
        )

        (
            corner_positions,
            corner_strengths,
        ) = _find_profile_peaks(
            profile=horizontal_strength,
            start=corner_search_start,
            stop=corner_search_stop,
            mad_multiplier=(
                config.reference_corner_gradient_mad_multiplier
            ),
            minimum_prominence=(
                config.reference_corner_min_prominence
            ),
        )

        corner_cluster_candidates: list[
            tuple[
                list[tuple[int, float]],
                bool,
            ]
        ] = []

        if corner_positions:
            corner_clusters = _cluster_peaks(
                corner_positions,
                corner_strengths,
                max_gap=(
                    config.halo_cluster_max_gap_pixels
                ),
            )

            # Keep the existing corner logic unchanged in this revision.
            corner_clusters = _get_strong_clusters(
                clusters=corner_clusters,
                minimum_relative_strength=(
                    config.reference_corner_cluster_min_relative_strength
                ),
            )

            # While moving upward, the nearest cluster has the greatest Y.
            corner_clusters = sorted(
                corner_clusters,
                key=lambda cluster: (
                    pivot_box.y_min
                    - max(
                        position
                        for position, _ in cluster
                    )
                ),
            )

            for cluster in corner_clusters[
                :config.reference_corner_max_candidates
            ]:
                corner_cluster_candidates.append(
                    (
                        cluster,
                        False,
                    )
                )

        else:
            segment = horizontal_strength[
                corner_search_start:
                corner_search_stop
            ]

            if segment.size == 0:
                continue

            inferred_corner_y = (
                corner_search_start
                + int(np.argmax(segment))
            )

            corner_cluster_candidates.append(
                (
                    [(inferred_corner_y, 0.15)],
                    True,
                )
            )

        for (
            corner_cluster,
            corner_was_inferred,
        ) in corner_cluster_candidates:
            corner_cluster_positions = [
                position
                for position, _ in corner_cluster
            ]

            # Cross the entire local halo cluster while moving upward.
            corner_y = min(
                corner_cluster_positions
            )

            corner_score = float(
                max(
                    strength
                    for _, strength in corner_cluster
                )
                / max(
                    max(corner_strengths)
                    if corner_strengths
                    else 1.0,
                    1e-8,
                )
            )

            candidate = ReferenceCorner(
                side=side,
                point=PixelPoint(
                    x=int(wall_x),
                    y=int(corner_y),
                ),
                wall_x=int(wall_x),
                wall_score=wall_score,
                corner_score=corner_score,
                wall_cluster_start=int(
                    wall_cluster_start
                ),
                wall_cluster_end=int(
                    wall_cluster_end
                ),
                corner_cluster_start=int(
                    min(corner_cluster_positions)
                ),
                corner_cluster_end=int(
                    max(corner_cluster_positions)
                ),
                wall_was_inferred=(
                    wall_was_inferred
                ),
                corner_was_inferred=(
                    corner_was_inferred
                ),
            )

            candidates.append(candidate)

            print(
                f"{side.capitalize()} reference candidate generated: "
                f"point={candidate.point}, "
                f"wall_cluster=({wall_cluster_start}, {wall_cluster_end}), "
                f"wall_strength_score={wall_strength_score:.4f}, "
                f"wall_distance={wall_distance_pixels:.1f}px, "
                f"wall_distance_score={wall_distance_score:.4f}, "
                f"wall_score={wall_score:.4f}, "
                f"corner_score={corner_score:.4f}, "
                f"reference_score={candidate.score:.4f}"
            )

    if not candidates:
        raise ValueError(
            f"No {side} reference-corner candidates were found."
        )

    # Preserve the existing candidate ordering/truncation behaviour so this
    # change affects only wall ranking rather than introducing a second,
    # unrelated selection change.
    def candidate_distance(
        candidate: ReferenceCorner,
    ) -> float:
        if side == "left":
            horizontal_distance = (
                pivot_box.x_min
                - candidate.point.x
            )
        else:
            horizontal_distance = (
                candidate.point.x
                - pivot_box.x_max
            )

        vertical_distance = (
            pivot_box.y_min
            - candidate.point.y
        )

        return float(
            max(horizontal_distance, 0)
            + max(vertical_distance, 0)
        )

    candidates.sort(
        key=candidate_distance
    )

    print(
        f"\n{side.upper()} REFERENCE-CORNER CANDIDATES "
        "(before bottom-transition validation):"
    )
    print("-" * 135)
    for rank, candidate in enumerate(candidates, start=1):
        kept_marker = (
            " <-- KEPT"
            if rank <= config.reference_total_max_candidates
            else ""
        )
        print(
            f"Rank {rank}: point={candidate.point}, "
            f"wall_score={candidate.wall_score:.4f}, "
            f"corner_score={candidate.corner_score:.4f}, "
            f"reference_score={candidate.score:.4f}, "
            f"distance_metric={candidate_distance(candidate):.1f}"
            f"{kept_marker}"
        )
    print("-" * 135)

    return candidates[
        :config.reference_total_max_candidates
    ]

def _collect_reference_bottom_candidates(
    side: SideName,
    smoothed: NDArray[np.float32],
    reference_candidates: list[ReferenceCorner],
    background_height: float,
    robust_height_range: float,
    config: XpanderSegmentationConfig,
) -> list[ReferenceBottomCandidate]:
    """
    Keep several valid reference->bottom hypotheses for one side.

    The previous implementation selected one best bottom row immediately for
    each reference corner.  That loses useful alternatives.  In files such as
    7.npy, a very strong local transition can occur only ~20 px from the
    reference, while the physically correct Xpander boundary is farther out
    with a slightly lower local score.  We keep both and let the LEFT/RIGHT
    pair geometry decide later.
    """
    successful: list[ReferenceBottomCandidate] = []
    failure_messages: list[str] = []

    for reference_index, reference in enumerate(
        reference_candidates,
        start=1,
    ):
        try:
            bottom_candidates = _find_bottom_xpander_corner_candidates(
                side=side,
                smoothed=smoothed,
                reference_corner=reference,
                background_height=background_height,
                robust_height_range=robust_height_range,
                config=config,
            )
        except ValueError as error:
            failure_messages.append(
                f"reference {reference_index} at "
                f"({reference.point.x}, {reference.point.y}): {error}"
            )
            print(
                f"{side.capitalize()} reference candidate "
                f"#{reference_index} FAILED: "
                f"reference={reference.point}, reason={error}"
            )
            continue

        for bottom_index, bottom_candidate in enumerate(
            bottom_candidates,
            start=1,
        ):
            detection = bottom_candidate.detection

            # Local quality is used only for ordering/pruning.  The final
            # decision is the cross-side pair score below.
            local_score = float(
                0.30 * reference.score
                + 0.70
                * np.clip(
                    bottom_candidate.selection_score,
                    0.0,
                    1.0,
                )
            )

            candidate = ReferenceBottomCandidate(
                side=side,
                reference=reference,
                bottom=bottom_candidate,
                local_score=local_score,
            )
            successful.append(candidate)

            print(
                f"{side.capitalize()} reference/bottom candidate "
                f"R{reference_index}-B{bottom_index} PASSED: "
                f"reference={reference.point}, "
                f"bottom={detection.bottom_corner}, "
                f"phase={bottom_candidate.phase_name}, "
                f"offset_y={bottom_candidate.y_offset}, "
                f"outward_distance={candidate.outward_distance_pixels}px, "
                f"reference_score={reference.score:.4f}, "
                f"transition_score={detection.transition.score:.4f}, "
                f"profile_selection_score="
                f"{bottom_candidate.selection_score:.4f}, "
                f"local_score={local_score:.4f}"
            )

    if not successful:
        recent_failures = "; ".join(failure_messages[-3:])
        raise ValueError(
            f"Tried {len(reference_candidates)} {side} reference candidates, "
            "but none produced a valid bottom Xpander boundary. "
            f"Recent failures: {recent_failures}"
        )

    successful.sort(
        key=lambda candidate: candidate.local_score,
        reverse=True,
    )

    print(
        f"\n{side.upper()} REFERENCE/BOTTOM HYPOTHESES "
        "(local ranking before LEFT/RIGHT pairing):"
    )
    print("-" * 170)
    for rank, candidate in enumerate(successful, start=1):
        kept = (
            " <-- KEPT FOR PAIRING"
            if rank <= config.reference_bottom_total_max_candidates
            else ""
        )
        print(
            f"Rank {rank}: reference={candidate.reference.point}, "
            f"bottom={candidate.detection.bottom_corner}, "
            f"outward={candidate.outward_distance_pixels}px, "
            f"transition={candidate.detection.transition.score:.4f}, "
            f"profile_score={candidate.bottom.selection_score:.4f}, "
            f"ref_score={candidate.reference.score:.4f}, "
            f"local={candidate.local_score:.4f}{kept}"
        )
    print("-" * 170)

    return successful[:config.reference_bottom_total_max_candidates]


def _soft_alignment_score(
    difference: float,
    tolerance: float,
) -> float:
    return float(
        np.exp(
            -max(float(difference), 0.0)
            / max(float(tolerance), 1e-8)
        )
    )


def _select_reference_bottom_pair(
    left_candidates: list[ReferenceBottomCandidate],
    right_candidates: list[ReferenceBottomCandidate],
    pivot_box: BoundingBox,
    expected_xpander_surface_height: float,
    robust_height_range: float,
    config: XpanderSegmentationConfig,
) -> ReferenceBottomPairSelection:
    """
    Select LEFT and RIGHT reference+bottom hypotheses jointly.

    The score uses soft evidence only:
      * mean transition quality;
      * mean reference quality;
      * reference-Y agreement;
      * bottom-Y agreement;
      * weak symmetry of outward reference->bottom travel;
      * consistency of the detected Xpander plateau height with the physical
        Xpander surface measured immediately outside the Pivot in Stage 3.

    The outward-distance term is intentionally weak. A neighboring component
    can be just as symmetric as the real Xpander, as seen in 6.npy. Surface
    height therefore provides an independent physical cue: the plateau before
    the bottom threshold must look like the surrounding Xpander surface rather
    than a lower shadow/sill plateau.
    """
    del pivot_box  # Retained in the signature for pair-geometry extensibility.

    if not left_candidates:
        raise ValueError("No valid left reference/bottom hypotheses remain.")
    if not right_candidates:
        raise ValueError("No valid right reference/bottom hypotheses remain.")

    if not np.isfinite(expected_xpander_surface_height):
        raise ValueError(
            "Expected Xpander surface height is not finite: "
            f"{expected_xpander_surface_height}."
        )

    surface_height_tolerance = max(
        float(config.pair_surface_height_tolerance_absolute),
        float(config.pair_surface_height_tolerance_fraction)
        * float(robust_height_range),
        1e-8,
    )

    weight_sum = float(
        config.pair_transition_weight
        + config.pair_reference_weight
        + config.pair_bottom_y_alignment_weight
        + config.pair_reference_y_alignment_weight
        + config.pair_outward_distance_symmetry_weight
        + config.pair_surface_height_weight
    )
    if not np.isclose(weight_sum, 1.0, atol=1e-6):
        raise ValueError(
            "Pair-selection weights must sum to 1.0; "
            f"received {weight_sum:.6f}."
        )

    pair_candidates: list[ReferenceBottomPairSelection] = []

    for left in left_candidates:
        left_bottom = left.detection.bottom_corner
        left_outward = left.outward_distance_pixels

        if left_outward <= 0:
            continue

        for right in right_candidates:
            right_bottom = right.detection.bottom_corner
            right_outward = right.outward_distance_pixels

            if right_outward <= 0:
                continue
            if left_bottom.x >= right_bottom.x:
                continue

            reference_y_difference = abs(
                left.reference.point.y
                - right.reference.point.y
            )
            bottom_y_difference = abs(
                left_bottom.y
                - right_bottom.y
            )
            outward_distance_difference = abs(
                left_outward
                - right_outward
            )

            reference_y_alignment_score = _soft_alignment_score(
                difference=reference_y_difference,
                tolerance=config.reference_y_alignment_tolerance_pixels,
            )
            bottom_y_alignment_score = _soft_alignment_score(
                difference=bottom_y_difference,
                tolerance=config.bottom_y_alignment_tolerance_pixels,
            )
            outward_distance_symmetry_score = _soft_alignment_score(
                difference=outward_distance_difference,
                tolerance=(
                    config.outward_distance_symmetry_tolerance_pixels
                ),
            )

            mean_transition_score = float(
                0.5
                * (
                    left.detection.transition.score
                    + right.detection.transition.score
                )
            )
            mean_reference_score = float(
                0.5
                * (
                    left.reference.score
                    + right.reference.score
                )
            )

            # The height before the detected raised threshold is what the
            # detector currently calls the Xpander plateau. Compare it to the
            # Stage-3 outside-wall estimate of the actual surrounding surface.
            left_surface_height_difference = abs(
                float(left.detection.xpander_height)
                - expected_xpander_surface_height
            )
            right_surface_height_difference = abs(
                float(right.detection.xpander_height)
                - expected_xpander_surface_height
            )

            # Treat Stage-3 outside height as a plausibility band rather than a
            # target to optimize exactly. Local halo/smoothing can shift the
            # bottom-profile plateau by about one tolerance unit. Candidates
            # inside the free band therefore receive the same full score; only
            # clearly inconsistent lower/upper plateaus are penalized. This
            # prevents a remote neighbor surface from winning merely because
            # its absolute height happens to match Stage 3 more closely.
            surface_free_band = (
                config.pair_surface_height_free_band_multiplier
                * surface_height_tolerance
            )
            left_surface_excess = max(
                left_surface_height_difference - surface_free_band,
                0.0,
            )
            right_surface_excess = max(
                right_surface_height_difference - surface_free_band,
                0.0,
            )
            left_surface_height_score = _soft_alignment_score(
                difference=left_surface_excess,
                tolerance=surface_height_tolerance,
            )
            right_surface_height_score = _soft_alignment_score(
                difference=right_surface_excess,
                tolerance=surface_height_tolerance,
            )
            surface_height_score = float(
                0.5
                * (
                    left_surface_height_score
                    + right_surface_height_score
                )
            )

            # Diagnostic only for now. This is useful for spotting a pair where
            # one side is on the real Xpander plateau and the other side is on a
            # lower halo/neighbor plateau. It is deliberately not a separate
            # weighted term so we do not double-count the same physical cue.
            left_right_surface_height_difference = abs(
                float(left.detection.xpander_height)
                - float(right.detection.xpander_height)
            )
            left_right_surface_consistency_score = _soft_alignment_score(
                difference=left_right_surface_height_difference,
                tolerance=surface_height_tolerance,
            )

            pair_score = float(
                config.pair_transition_weight
                * mean_transition_score
                + config.pair_reference_weight
                * mean_reference_score
                + config.pair_bottom_y_alignment_weight
                * bottom_y_alignment_score
                + config.pair_reference_y_alignment_weight
                * reference_y_alignment_score
                + config.pair_outward_distance_symmetry_weight
                * outward_distance_symmetry_score
                + config.pair_surface_height_weight
                * surface_height_score
            )

            pair_candidates.append(
                ReferenceBottomPairSelection(
                    left=left,
                    right=right,
                    score=pair_score,
                    mean_transition_score=mean_transition_score,
                    mean_reference_score=mean_reference_score,
                    reference_y_difference=reference_y_difference,
                    reference_y_alignment_score=(
                        reference_y_alignment_score
                    ),
                    bottom_y_difference=bottom_y_difference,
                    bottom_y_alignment_score=bottom_y_alignment_score,
                    outward_distance_difference=(
                        outward_distance_difference
                    ),
                    outward_distance_symmetry_score=(
                        outward_distance_symmetry_score
                    ),
                    expected_xpander_surface_height=(
                        expected_xpander_surface_height
                    ),
                    surface_height_tolerance=surface_height_tolerance,
                    left_surface_height_difference=(
                        left_surface_height_difference
                    ),
                    right_surface_height_difference=(
                        right_surface_height_difference
                    ),
                    left_surface_height_score=left_surface_height_score,
                    right_surface_height_score=right_surface_height_score,
                    surface_height_score=surface_height_score,
                    left_right_surface_height_difference=(
                        left_right_surface_height_difference
                    ),
                    left_right_surface_consistency_score=(
                        left_right_surface_consistency_score
                    ),
                )
            )

    if not pair_candidates:
        raise ValueError(
            "No geometrically valid LEFT/RIGHT reference-bottom pair was found."
        )

    pair_candidates.sort(
        key=lambda candidate: candidate.score,
        reverse=True,
    )

    print("\nLEFT/RIGHT REFERENCE-BOTTOM PAIR CANDIDATES:")
    print("-" * 245)
    print(
        "weights: "
        f"transition={config.pair_transition_weight:.2f}, "
        f"reference={config.pair_reference_weight:.2f}, "
        f"bottom_y={config.pair_bottom_y_alignment_weight:.2f}, "
        f"reference_y={config.pair_reference_y_alignment_weight:.2f}, "
        f"outward_symmetry="
        f"{config.pair_outward_distance_symmetry_weight:.2f}, "
        f"surface_height={config.pair_surface_height_weight:.2f}; "
        f"outward_tolerance="
        f"{config.outward_distance_symmetry_tolerance_pixels:.1f}px, "
        f"expected_surface={expected_xpander_surface_height:.6f}, "
        f"surface_tolerance={surface_height_tolerance:.6f}, "
        f"surface_free_band="
        f"{config.pair_surface_height_free_band_multiplier * surface_height_tolerance:.6f}"
    )

    debug_limit = max(1, config.pair_debug_max_candidates)
    for rank, pair in enumerate(
        pair_candidates[:debug_limit],
        start=1,
    ):
        selected_marker = " <-- SELECTED" if rank == 1 else ""
        print(
            f"Rank {rank}: "
            f"Lref={pair.left.reference.point}, "
            f"Lbottom={pair.left.detection.bottom_corner}, "
            f"Lout={pair.left.outward_distance_pixels}px, "
            f"Lheight={pair.left.detection.xpander_height:.4f}, "
            f"Lheight_diff={pair.left_surface_height_difference:.4f}, "
            f"Lheight_score={pair.left_surface_height_score:.4f} | "
            f"Rref={pair.right.reference.point}, "
            f"Rbottom={pair.right.detection.bottom_corner}, "
            f"Rout={pair.right.outward_distance_pixels}px, "
            f"Rheight={pair.right.detection.xpander_height:.4f}, "
            f"Rheight_diff={pair.right_surface_height_difference:.4f}, "
            f"Rheight_score={pair.right_surface_height_score:.4f} | "
            f"transition={pair.mean_transition_score:.4f}, "
            f"reference={pair.mean_reference_score:.4f}, "
            f"ref_y_diff={pair.reference_y_difference}px, "
            f"ref_y_score={pair.reference_y_alignment_score:.4f}, "
            f"bottom_y_diff={pair.bottom_y_difference}px, "
            f"bottom_y_score={pair.bottom_y_alignment_score:.4f}, "
            f"outward_diff={pair.outward_distance_difference}px, "
            f"outward_score={pair.outward_distance_symmetry_score:.4f}, "
            f"surface_score={pair.surface_height_score:.4f}, "
            f"LR_height_diff={pair.left_right_surface_height_difference:.4f}, "
            f"LR_height_score="
            f"{pair.left_right_surface_consistency_score:.4f}, "
            f"FINAL={pair.score:.4f}{selected_marker}"
        )

    if len(pair_candidates) > debug_limit:
        print(
            f"... {len(pair_candidates) - debug_limit} additional "
            "pair candidates omitted from debug output."
        )
    print("-" * 245)

    best = pair_candidates[0]
    print(
        "SELECTED LEFT/RIGHT PAIR: "
        f"left_reference={best.left.reference.point}, "
        f"left_bottom={best.left.detection.bottom_corner}, "
        f"right_reference={best.right.reference.point}, "
        f"right_bottom={best.right.detection.bottom_corner}, "
        f"left_outward={best.left.outward_distance_pixels}px, "
        f"right_outward={best.right.outward_distance_pixels}px, "
        f"outward_difference={best.outward_distance_difference}px, "
        f"left_height={best.left.detection.xpander_height:.6f}, "
        f"right_height={best.right.detection.xpander_height:.6f}, "
        f"expected_surface={best.expected_xpander_surface_height:.6f}, "
        f"surface_score={best.surface_height_score:.4f}, "
        f"left_right_height_difference="
        f"{best.left_right_surface_height_difference:.6f}, "
        f"pair_score={best.score:.4f}"
    )

    return best


# def _find_bottom_xpander_corner(
#     side: SideName,
#     smoothed: NDArray[np.float32],
#     reference_corner: ReferenceCorner,
#     background_height: float,
#     robust_height_range: float,
#     config: XpanderSegmentationConfig,
# ) -> SideCornerDetection:
#     """
#     Search several horizontal rows around the approximate reference-corner Y.

#     The reference detector provides an approximate structural location. It does
#     not guarantee that its exact Y coordinate crosses a clean Xpander plateau.
#     """
#     image_height = smoothed.shape[0]
#     reference_y = reference_corner.point.y

#     radius = max(
#         0,
#         config.bottom_profile_y_search_radius,
#     )
#     step = max(
#         1,
#         config.bottom_profile_y_search_step,
#     )

#     candidate_y_values: list[int] = [
#         reference_y,
#     ]

#     for offset in range(
#         step,
#         radius + 1,
#         step,
#     ):
#         # Try the nearest rows first.
#         candidate_y_values.append(
#             reference_y - offset
#         )
#         candidate_y_values.append(
#             reference_y + offset
#         )

#     # Remove duplicates and invalid coordinates while preserving order.
#     valid_y_values: list[int] = []
#     seen: set[int] = set()

#     for candidate_y in candidate_y_values:
#         if candidate_y < 0 or candidate_y >= image_height:
#             continue
#         if candidate_y in seen:
#             continue

#         seen.add(candidate_y)
#         valid_y_values.append(candidate_y)

#     successful_detections: list[
#         tuple[
#             float,
#             SideCornerDetection,
#         ]
#     ] = []

#     failure_messages: list[str] = []

#     for scan_y in valid_y_values:
#         try:
#             detection = _find_bottom_xpander_corner_at_y(
#                 side=side,
#                 smoothed=smoothed,
#                 reference_corner=reference_corner,
#                 scan_y=scan_y,
#                 background_height=background_height,
#                 robust_height_range=robust_height_range,
#                 config=config,
#             )
#         except ValueError as error:
#             failure_messages.append(
#                 f"y={scan_y}: {error}"
#             )
#             continue

#         distance = abs(
#             scan_y - reference_y
#         )

#         distance_score = float(
#             np.exp(
#                 -distance
#                 / max(
#                     radius,
#                     1,
#                 )
#             )
#         )

#         combined_score = float(
#             (
#                 1.0
#                 - config.bottom_profile_y_distance_weight
#             )
#             * detection.transition.score
#             + config.bottom_profile_y_distance_weight
#             * distance_score
#         )

#         successful_detections.append(
#             (
#                 combined_score,
#                 detection,
#             )
#         )

#     if not successful_detections:
#         recent_failures = "; ".join(
#             failure_messages[-4:]
#         )

#         raise ValueError(
#             f"No valid {side} bottom transition was found "
#             f"within Y={reference_y - radius}:"
#             f"{reference_y + radius} around reference "
#             f"{reference_corner.point}. "
#             f"Recent failures: {recent_failures}"
#         )

#     (
#         selected_score,
#         selected_detection,
#     ) = max(
#         successful_detections,
#         key=lambda item: item[0],
#     )

#     print(
#         f"Selected {side} bottom profile: "
#         f"reference={reference_corner.point}, "
#         f"scan_y={selected_detection.bottom_corner.y}, "
#         f"bottom_x={selected_detection.bottom_corner.x}, "
#         f"transition_score="
#         f"{selected_detection.transition.score:.4f}, "
#         f"combined_score={selected_score:.4f}"
#     )

#     return selected_detection


def _collect_bottom_profile_offsets(
    side: SideName,
    smoothed: NDArray[np.float32],
    reference_corner: ReferenceCorner,
    offsets: tuple[int, ...],
    background_height: float,
    robust_height_range: float,
    config: XpanderSegmentationConfig,
    phase_name: str,
) -> list[BottomProfileCandidate]:
    """Return every valid bottom hypothesis from the supplied scan rows."""
    image_height = smoothed.shape[0]
    reference_y = reference_corner.point.y
    successful: list[BottomProfileCandidate] = []

    for offset in offsets:
        scan_y = reference_y + offset

        if scan_y < 0 or scan_y >= image_height:
            continue

        try:
            detection = _find_bottom_xpander_corner_at_y(
                side=side,
                smoothed=smoothed,
                reference_corner=reference_corner,
                scan_y=scan_y,
                background_height=background_height,
                robust_height_range=robust_height_range,
                config=config,
            )
        except ValueError as error:
            print(
                f"{side.capitalize()} bottom {phase_name} candidate "
                f"failed: y={scan_y}, offset={offset}: {error}"
            )
            continue

        distance = abs(offset)
        selection_score = float(
            detection.transition.score
            - config.bottom_profile_y_penalty_per_pixel
            * distance
        )

        candidate = BottomProfileCandidate(
            detection=detection,
            selection_score=selection_score,
            y_distance=distance,
            y_offset=offset,
            phase_name=phase_name,
        )
        successful.append(candidate)

        print(
            f"{side.capitalize()} bottom {phase_name} candidate "
            f"passed: y={scan_y}, offset={offset}, "
            f"x={detection.bottom_corner.x}, "
            f"transition_score={detection.transition.score:.4f}, "
            f"selection_score={selection_score:.4f}"
        )

    successful.sort(
        key=lambda candidate: candidate.selection_score,
        reverse=True,
    )
    return successful


def _find_bottom_xpander_corner_candidates(
    side: SideName,
    smoothed: NDArray[np.float32],
    reference_corner: ReferenceCorner,
    background_height: float,
    robust_height_range: float,
    config: XpanderSegmentationConfig,
) -> list[BottomProfileCandidate]:
    """
    Return several bottom-corner hypotheses for one reference corner.

    Primary rows are preferred.  The wider fallback is used only when no
    primary row produces any valid transition, preserving the existing
    exceptional-input behaviour.
    """
    primary_offsets = (
        0,
        -2,
        2,
        -4,
        4,
    )

    primary_candidates = _collect_bottom_profile_offsets(
        side=side,
        smoothed=smoothed,
        reference_corner=reference_corner,
        offsets=primary_offsets,
        background_height=background_height,
        robust_height_range=robust_height_range,
        config=config,
        phase_name="primary",
    )

    if primary_candidates:
        kept = primary_candidates[
            :max(1, config.bottom_candidates_per_reference)
        ]
        print(
            f"Keeping {len(kept)} {side} primary bottom hypotheses "
            f"for reference={reference_corner.point}."
        )
        return kept

    fallback_offsets: list[int] = []
    for distance in range(
        6,
        config.bottom_profile_y_search_radius + 1,
        config.bottom_profile_y_search_step,
    ):
        fallback_offsets.extend([-distance, distance])

    fallback_candidates = _collect_bottom_profile_offsets(
        side=side,
        smoothed=smoothed,
        reference_corner=reference_corner,
        offsets=tuple(fallback_offsets),
        background_height=background_height,
        robust_height_range=robust_height_range,
        config=config,
        phase_name="fallback",
    )

    if not fallback_candidates:
        raise ValueError(
            f"No valid {side} bottom transition was found "
            f"around reference {reference_corner.point}, "
            "including the exceptional-input fallback search."
        )

    kept = fallback_candidates[
        :max(1, config.bottom_candidates_per_reference)
    ]
    print(
        f"Keeping {len(kept)} {side} FALLBACK bottom hypotheses "
        f"for reference={reference_corner.point}."
    )
    return kept


def _try_bottom_profile_offsets(
    side: SideName,
    smoothed: NDArray[np.float32],
    reference_corner: ReferenceCorner,
    offsets: tuple[int, ...],
    background_height: float,
    robust_height_range: float,
    config: XpanderSegmentationConfig,
    phase_name: str,
) -> SideCornerDetection | None:
    """Legacy single-best wrapper around _collect_bottom_profile_offsets()."""
    candidates = _collect_bottom_profile_offsets(
        side=side,
        smoothed=smoothed,
        reference_corner=reference_corner,
        offsets=offsets,
        background_height=background_height,
        robust_height_range=robust_height_range,
        config=config,
        phase_name=phase_name,
    )

    if not candidates:
        return None

    best = candidates[0]
    print(
        f"Selected {side} {phase_name} row: "
        f"x={best.detection.bottom_corner.x}, "
        f"y={best.detection.bottom_corner.y}, "
        f"distance={best.y_distance}, "
        f"transition_score="
        f"{best.detection.transition.score:.4f}, "
        f"selection_score={best.selection_score:.4f}"
    )
    return best.detection


def _find_bottom_xpander_corner(
    side: SideName,
    smoothed: NDArray[np.float32],
    reference_corner: ReferenceCorner,
    background_height: float,
    robust_height_range: float,
    config: XpanderSegmentationConfig,
) -> SideCornerDetection:
    """
    First try rows very close to the detected reference corner.

    Only if the normal search fails, expand the Y search for exceptional
    inputs with damaged or missing corner pixels, such as 6.npy.
    """
    image_height = smoothed.shape[0]
    reference_y = reference_corner.point.y

    # Normal inputs should remain close to the detected reference Y.
    primary_offsets = (
        0,
        -2,
        2,
        -4,
        4,
    )

    # Wider fallback used only when the normal search completely fails.
    fallback_offsets: list[int] = []
    for distance in range(
        6,
        config.bottom_profile_y_search_radius + 1,
        config.bottom_profile_y_search_step,
    ):
        fallback_offsets.extend(
            [
                -distance,
                distance,
            ]
        )

    primary_result = _try_bottom_profile_offsets(
        side=side,
        smoothed=smoothed,
        reference_corner=reference_corner,
        offsets=primary_offsets,
        background_height=background_height,
        robust_height_range=robust_height_range,
        config=config,
        phase_name="primary",
    )

    if primary_result is not None:
        print(
            f"Selected {side} bottom profile using primary search: "
            f"reference={reference_corner.point}, "
            f"corner={primary_result.bottom_corner}, "
            f"offset_y="
            f"{primary_result.bottom_corner.y - reference_y}, "
            f"score={primary_result.transition.score:.4f}"
        )
        return primary_result

    fallback_result = _try_bottom_profile_offsets(
        side=side,
        smoothed=smoothed,
        reference_corner=reference_corner,
        offsets=tuple(fallback_offsets),
        background_height=background_height,
        robust_height_range=robust_height_range,
        config=config,
        phase_name="fallback",
    )

    if fallback_result is None:
        raise ValueError(
            f"No valid {side} bottom transition was found "
            f"around reference {reference_corner.point}, "
            "including the exceptional-input fallback search."
        )

    print(
        f"Selected {side} bottom profile using FALLBACK search: "
        f"reference={reference_corner.point}, "
        f"corner={fallback_result.bottom_corner}, "
        f"offset_y="
        f"{fallback_result.bottom_corner.y - reference_y}, "
        f"score={fallback_result.transition.score:.4f}"
    )

    return fallback_result


def _find_bottom_xpander_corner_at_y(
    side: SideName,
    smoothed: NDArray[np.float32],
    reference_corner: ReferenceCorner,
    scan_y: int,
    background_height: float,
    robust_height_range: float,
    config: XpanderSegmentationConfig,
) -> SideCornerDetection:
    """
    Continue in the SAME direction from the reference corner:

        left reference  -> move left
        right reference -> move right

    After skipping the reference halo and finding stable Xpander pixels, detect
    the first valid Xpander -> raised threshold -> outside transition.
    """
    image_height, image_width = smoothed.shape
    reference_x = reference_corner.point.x
    scan_y = int(np.clip(scan_y, 0, image_height - 1))

    bottom_half_band = (config.bottom_profile_band_half_width)

    y_min = max(0, scan_y - bottom_half_band)
    y_max = min(image_height, scan_y + bottom_half_band + 1)

    horizontal_band = smoothed[y_min:y_max,:,]

    # The separating marker is raised and may appear in only part of the
    # vertical band. Median aggregation can suppress it, so use an upper
    # percentile rather than the median.
    horizontal_profile = np.percentile(
        horizontal_band,
        config.bottom_profile_band_percentile,
        axis=0,
    )

    departure = max(1, config.reference_departure_pixels)
    if side == "left":
        start_x = reference_x - departure
        path_positions = np.arange(start_x, -1, -1, dtype=np.int64)
    else:
        start_x = reference_x + departure
        path_positions = np.arange(start_x, image_width, 1, dtype=np.int64)

    path_positions = path_positions[
        (path_positions >= 0) & (path_positions < image_width)
    ]
    if path_positions.size < max(10, 2 * config.stable_window_pixels):
        raise ValueError(
            f"The {side} outward path is too short after leaving the reference corner."
        )

    raw_path_profile = horizontal_profile[path_positions]
    profile = _smooth_profile(raw_path_profile, config.profile_gaussian_sigma)

    stable_start, stable_stop, stable_height = _find_stable_plateau(
        profile=profile,
        robust_height_range=robust_height_range,
        config=config,
    )

    transition = _find_raised_threshold_transition(
        profile=profile,
        path_positions=path_positions,
        search_start_index=stable_stop,
        stable_height=stable_height,
        robust_height_range=robust_height_range,
        config=config,
        require_after_background=False,
        background_height=background_height,
    )

    # Along the corrected outward path, the RISING edge is
    # Xpander -> threshold, so it is the Xpander bottom corner.
    bottom_x = transition.rise_position
    xpander_height = transition.before_height
    outer_height = transition.after_height

    return SideCornerDetection(
        side=side,
        reference_corner=reference_corner,
        bottom_corner=PixelPoint(x=int(bottom_x), y=int(scan_y)),
        transition=transition,
        outer_region_type=_classify_outer_region(
            outer_height=outer_height,
            background_height=background_height,
            robust_height_range=robust_height_range,
            config=config,
        ),
        outer_height=float(outer_height),
        xpander_height=float(xpander_height),
        diagnostics=PathDiagnostics(
            positions=path_positions,
            profile=profile,
            stable_start_index=stable_start,
            stable_stop_index=stable_stop,
            stable_height=float(stable_height),
        ),
    )


def _find_top_xpander_corner(
    side: SideName,
    smoothed: NDArray[np.float32],
    bottom_corner: PixelPoint,
    search_x: int,
    background_height: float,
    robust_height_range: float,
    config: XpanderSegmentationConfig,
) -> TopCornerDetection:
    """Move upward and detect Xpander -> raised threshold -> background."""
    image_height, image_width = smoothed.shape
    search_x = int(np.clip(search_x, 0, image_width - 1))

    x_min = max(0, search_x - config.profile_band_half_width)
    x_max = min(image_width, search_x + config.profile_band_half_width + 1)
    vertical_profile = np.median(smoothed[:, x_min:x_max], axis=1)

    start_y = int(
        np.clip(
            bottom_corner.y - config.top_search_start_gap_pixels,
            0,
            image_height - 1,
        )
    )
    path_positions = np.arange(start_y, -1, -1, dtype=np.int64)
    if path_positions.size < max(10, 2 * config.stable_window_pixels):
        raise ValueError(f"The {side} upward path is too short.")

    raw_path_profile = vertical_profile[path_positions]
    profile = _smooth_profile(raw_path_profile, config.profile_gaussian_sigma)

    stable_start, stable_stop, stable_height = _find_stable_plateau(
        profile=profile,
        robust_height_range=robust_height_range,
        config=config,
    )

    transition = _find_top_boundary_transition(
        profile=profile,
        path_positions=path_positions,
        search_start_index=stable_stop,
        stable_height=stable_height,
        background_height=background_height,
        robust_height_range=robust_height_range,
        config=config,
    )

    tolerance = max(
        config.background_tolerance_fraction * robust_height_range,
        1e-8,
    )
    background_match_score = float(
        np.exp(-abs(transition.after_height - background_height) / tolerance)
    )

    return TopCornerDetection(
        side=side,
        bottom_corner=bottom_corner,
        top_corner=PixelPoint(x=int(bottom_corner.x), y=int(transition.rise_position)),
        search_x=search_x,
        transition=transition,
        background_height_after_threshold=float(transition.after_height),
        background_match_score=background_match_score,
        diagnostics=PathDiagnostics(
            positions=path_positions,
            profile=profile,
            stable_start_index=stable_start,
            stable_stop_index=stable_stop,
            stable_height=float(stable_height),
        ),
    )



def _find_top_boundary_transition(
    profile: NDArray[np.float64],
    path_positions: NDArray[np.int64],
    search_start_index: int,
    stable_height: float,
    background_height: float,
    robust_height_range: float,
    config: XpanderSegmentationConfig,
) -> ThresholdTransition:
    """
    Detect the real top Xpander boundary while travelling upward.

    Required order after the stable Xpander interior:

        stable Xpander
            -> small positive edge       [the top BB boundary]
            -> stronger negative edge    [drop into the upper sill]

    A later positive recovery edge is optional. When it exists, it significantly
    strengthens the candidate, but a valid small-rise -> stronger-fall pair is
    sufficient by itself.

    The first positive edge is returned as ``rise_position``.
    """
    if profile.size != path_positions.size:
        raise ValueError("Top profile and path positions have different sizes.")

    derivative = np.gradient(profile)
    derivative_center = float(np.median(derivative))
    derivative_mad = float(
        np.median(np.abs(derivative - derivative_center))
    )
    derivative_sigma = max(1.4826 * derivative_mad, 1e-8)

    initial_rise_threshold = max(
        config.top_initial_rise_min_absolute,
        config.top_initial_rise_min_fraction * robust_height_range,
        1.20 * derivative_sigma,
    )
    main_fall_threshold = max(
        config.top_main_fall_min_absolute,
        config.top_main_fall_min_fraction * robust_height_range,
        1.75 * derivative_sigma,
    )
    recovery_rise_threshold = max(
        config.top_recovery_rise_min_absolute,
        config.top_recovery_rise_min_fraction * robust_height_range,
        1.50 * derivative_sigma,
    )

    positive_peaks, _ = find_peaks(
        derivative,
        height=initial_rise_threshold,
        prominence=max(
            config.transition_min_prominence * 0.5,
            0.75 * derivative_sigma,
        ),
    )
    negative_peaks, _ = find_peaks(
        -derivative,
        height=main_fall_threshold,
        prominence=max(
            config.transition_min_prominence,
            derivative_sigma,
        ),
    )

    positive_peaks = positive_peaks[
        positive_peaks >= search_start_index
    ]
    negative_peaks = negative_peaks[
        negative_peaks >= search_start_index
    ]

    if positive_peaks.size == 0:
        raise ValueError(
            "No initial small rise was found after the stable Xpander plateau."
        )
    if negative_peaks.size == 0:
        raise ValueError(
            "Initial rises were found, but no stronger validating fall followed."
        )

    minimum_gap = max(1, config.top_pattern_min_gap_pixels)
    maximum_gap = max(minimum_gap, config.top_pattern_max_gap_pixels)
    probe = max(2, config.top_pattern_probe_pixels)

    plateau_tolerance = max(
        config.xpander_plateau_tolerance_fraction * robust_height_range,
        1e-8,
    )
    low_std_limit = max(
        config.top_pattern_max_after_std_fraction * robust_height_range,
        1e-8,
    )
    background_tolerance = max(
        config.background_tolerance_fraction * robust_height_range,
        1e-8,
    )

    candidates: list[ThresholdTransition] = []

    for initial_rise_index in positive_peaks:
        validating_falls = negative_peaks[
            (negative_peaks > initial_rise_index)
            & (negative_peaks - initial_rise_index >= minimum_gap)
            & (negative_peaks - initial_rise_index <= maximum_gap)
        ]

        for validating_fall_index in validating_falls:
            before_start = max(
                search_start_index,
                initial_rise_index - probe,
            )
            before_values = profile[
                before_start:initial_rise_index
            ]

            raised_stop = max(
                initial_rise_index + 1,
                validating_fall_index,
            )
            raised_values = profile[
                initial_rise_index:raised_stop
            ]

            # Search for an optional recovery after the required fall.
            recovery_candidates = positive_peaks[
                (positive_peaks > validating_fall_index)
                & (positive_peaks - validating_fall_index >= minimum_gap)
                & (positive_peaks - validating_fall_index <= maximum_gap)
                & (derivative[positive_peaks] >= recovery_rise_threshold)
            ]
            recovery_rise_index: int | None = (
                int(recovery_candidates[0])
                if recovery_candidates.size > 0
                else None
            )

            low_start = min(
                profile.size,
                validating_fall_index + 1,
            )
            if recovery_rise_index is not None:
                low_stop = max(
                    low_start + 1,
                    recovery_rise_index,
                )
            else:
                low_stop = min(
                    profile.size,
                    low_start + probe,
                )
            low_values = profile[low_start:low_stop]

            if (
                before_values.size < 2
                or raised_values.size < 1
                or low_values.size < 2
            ):
                continue

            before_height = float(np.median(before_values))
            raised_height = float(np.max(raised_values))
            low_height = float(np.median(low_values))

            plateau_background_contrast = abs(
                stable_height - background_height
            )

            if plateau_background_contrast <= 1e-8:
                continue

            # Direction from the Xpander plateau toward the background:
            # negative when the background is lower, positive when it is higher.
            direction_toward_background = np.sign(
                background_height - stable_height
            )

            progress_toward_background = float(
                (
                    low_height - stable_height
                )
                * direction_toward_background
            )

            progress_fraction = float(
                progress_toward_background
                / plateau_background_contrast
            )

            minimum_progress = float(
                config.top_min_progress_toward_background_fraction
                * plateau_background_contrast
            )

            # The post-fall region must actually move from the Xpander plateau
            # toward the background. Local bumps that return to the same surface
            # or remain on the opposite side of the plateau are rejected.
            if progress_toward_background < minimum_progress:
                continue

            before_std = float(np.std(before_values))
            low_std = float(np.std(low_values))

            initial_rise_amount = raised_height - before_height
            main_fall_amount = raised_height - low_height

            # A real top boundary must leave the Xpander plateau after the fall.
            # A local noise bump usually rises and then returns to approximately
            # the same surface height, so it must not be accepted.
            minimum_post_fall_drop = max(
                config.top_post_fall_drop_min_absolute,
                (
                    config.top_post_fall_drop_min_fraction
                    * robust_height_range
                ),
            )

            post_fall_drop_from_xpander = (
                before_height - low_height
            )

            if (
                post_fall_drop_from_xpander
                < minimum_post_fall_drop
            ):
                continue

            if abs(before_height - stable_height) > plateau_tolerance:
                continue
            if initial_rise_amount < initial_rise_threshold:
                continue
            if main_fall_amount < main_fall_threshold:
                continue
            if (
                main_fall_amount
                < config.top_main_fall_to_initial_rise_ratio
                * initial_rise_amount
            ):
                continue
            # The post-fall region should behave like a real sill rather than a
            # one-pixel derivative spike. Keep this tolerance deliberately
            # relaxed because the sill may not be perfectly flat.
            if low_std > 2.0 * low_std_limit:
                continue

            initial_strength = float(
                derivative[initial_rise_index]
            )
            fall_strength = float(
                -derivative[validating_fall_index]
            )

            plateau_score = float(
                np.exp(
                    -abs(before_height - stable_height)
                    / plateau_tolerance
                )
            )
            initial_score = float(
                1.0
                - np.exp(
                    -initial_strength
                    / (initial_rise_threshold + 1e-8)
                )
            )
            fall_score = float(
                1.0
                - np.exp(
                    -fall_strength
                    / (main_fall_threshold + 1e-8)
                )
            )
            sill_flatness_score = float(
                np.exp(
                    -low_std
                    / (low_std_limit + 1e-8)
                )
            )

            required_pattern_score = float(
                np.clip(
                    0.30 * initial_score
                    + 0.35 * fall_score
                    + 0.20 * plateau_score
                    + 0.15 * sill_flatness_score,
                    0.0,
                    1.0,
                )
            )

            recovery_position: int | None = None
            recovery_path_index: int | None = None
            recovery_bonus = 0.0
            after_height = low_height
            after_std = low_std
            final_path_index = int(validating_fall_index)
            final_position = int(
                path_positions[validating_fall_index]
            )

            if recovery_rise_index is not None:
                after_start = min(
                    profile.size,
                    recovery_rise_index + 1,
                )
                after_stop = min(
                    profile.size,
                    after_start + max(
                        probe,
                        config.background_probe_pixels,
                    ),
                )
                after_values = profile[after_start:after_stop]

                if after_values.size >= 2:
                    candidate_after_height = float(
                        np.median(after_values)
                    )
                    candidate_after_std = float(
                        np.std(after_values)
                    )
                    recovery_rise_amount = (
                        candidate_after_height - low_height
                    )

                    if recovery_rise_amount >= recovery_rise_threshold:
                        recovery_strength = float(
                            derivative[recovery_rise_index]
                        )
                        recovery_edge_score = float(
                            1.0
                            - np.exp(
                                -recovery_strength
                                / (recovery_rise_threshold + 1e-8)
                            )
                        )
                        background_score = float(
                            np.exp(
                                -abs(
                                    candidate_after_height
                                    - background_height
                                )
                                / background_tolerance
                            )
                        )
                        recovery_bonus = float(
                            config.top_recovery_score_bonus
                            * (
                                0.65 * recovery_edge_score
                                + 0.35 * background_score
                            )
                        )
                        recovery_position = int(
                            path_positions[recovery_rise_index]
                        )
                        recovery_path_index = int(
                            recovery_rise_index
                        )
                        after_height = candidate_after_height
                        after_std = candidate_after_std
                        final_path_index = recovery_path_index
                        final_position = recovery_position

            score = float(
                np.clip(
                    required_pattern_score + recovery_bonus,
                    0.0,
                    1.0,
                )
            )

            candidates.append(
                ThresholdTransition(
                    # This is the actual top BB boundary.
                    rise_position=int(
                        path_positions[initial_rise_index]
                    ),
                    # The required fall ends the minimum accepted pattern.
                    fall_position=final_position,
                    rise_path_index=int(initial_rise_index),
                    fall_path_index=final_path_index,
                    before_height=before_height,
                    threshold_height=low_height,
                    after_height=after_height,
                    before_std=before_std,
                    after_std=after_std,
                    rise_strength=initial_strength,
                    fall_strength=fall_strength,
                    threshold_width_pixels=int(
                        validating_fall_index - initial_rise_index
                    ),
                    threshold_rise=main_fall_amount,
                    score=score,
                    validation_fall_position=int(
                        path_positions[validating_fall_index]
                    ),
                    validation_fall_path_index=int(
                        validating_fall_index
                    ),
                    validation_recovery_position=recovery_position,
                    validation_recovery_path_index=recovery_path_index,
                )
            )

    if not candidates:
        raise ValueError(
            "Top-edge peaks were found, but no valid "
            "small-rise -> stronger-fall pattern started from "
            "the stable Xpander plateau."
        )

    # Sort only for readable debugging output.
    candidates_sorted = sorted(
        candidates,
        key=lambda candidate: candidate.rise_path_index,
    )

    for candidate_index, candidate in enumerate(
        candidates_sorted,
        start=1,
    ):
        print(
            f"[{candidate_index}] "
            f"rise_x={candidate.rise_position}, "
            f"fall_x={candidate.fall_position}, "
            f"rise_path_index={candidate.rise_path_index}, "
            f"fall_path_index={candidate.fall_path_index}, "
            f"distance_from_search_start="
            f"{candidate.rise_path_index - search_start_index}, "
            f"width={candidate.threshold_width_pixels}, "
            f"before={candidate.before_height:.6f}, "
            f"threshold={candidate.threshold_height:.6f}, "
            f"after={candidate.after_height:.6f}, "
            f"threshold_rise={candidate.threshold_rise:.6f}, "
            f"rise_strength={candidate.rise_strength:.6f}, "
            f"fall_strength={candidate.fall_strength:.6f}, "
            f"before_std={candidate.before_std:.6f}, "
            f"after_std={candidate.after_std:.6f}, "
            f"score={candidate.score:.4f}"
        )

    # The algorithm first identifies the earliest valid boundary in travel order.
    first_boundary_index = min(
        candidate.rise_path_index
        for candidate in candidates
    )

    # Candidates inside the same local halo/edge cluster are considered equivalent
    # spatially. Score is used only inside this near-first group.
    near_first = [
        candidate
        for candidate in candidates
        if candidate.rise_path_index
        <= first_boundary_index + config.halo_cluster_max_gap_pixels
    ]

    print(
        "\nFirst valid boundary:"
        f" path_index={first_boundary_index}, "
        f"position={int(path_positions[first_boundary_index])}"
    )

    print(
        "Near-first acceptance range:"
        f" path_index={first_boundary_index}:"
        f"{first_boundary_index + config.halo_cluster_max_gap_pixels}"
    )

    print("Candidates eligible for final selection:")

    for candidate in sorted(
        near_first,
        key=lambda item: item.rise_path_index,
    ):
        print(
            f"  rise_x={candidate.rise_position}, "
            f"path_index={candidate.rise_path_index}, "
            f"score={candidate.score:.4f}"
        )

    selected_candidate = max(
        near_first,
        key=lambda candidate: candidate.score,
    )

    print(
        "\nSELECTED TRANSITION:"
        f" rise_x={selected_candidate.rise_position}, "
        f"fall_x={selected_candidate.fall_position}, "
        f"rise_path_index={selected_candidate.rise_path_index}, "
        f"score={selected_candidate.score:.4f}"
    )

    later_candidates = [
        candidate
        for candidate in candidates
        if candidate not in near_first
    ]

    if later_candidates:
        print("Later candidates excluded because they are not near the first edge:")

        for candidate in sorted(
            later_candidates,
            key=lambda item: item.rise_path_index,
        ):
            print(
                f"  rise_x={candidate.rise_position}, "
                f"path_index={candidate.rise_path_index}, "
                f"score={candidate.score:.4f}, "
                f"distance_from_first="
                f"{candidate.rise_path_index - first_boundary_index}"
            )

    print("=" * 90 + "\n")

    print("\nALL TRANSITIONS:")

    for c in candidates:
        print(
            f"rise={c.rise_position}, "
            f"fall={c.fall_position}, "
            f"score={c.score:.4f}"
        )

    print(
        f"\nSELECTED: rise={selected_candidate.rise_position}, "
        f"score={selected_candidate.score:.4f}"
    )

    print("\n" + "=" * 100)
    print("TOP BOUNDARY CANDIDATES")
    print(
        f"stable_height={stable_height:.6f}, "
        f"background_height={background_height:.6f}, "
        f"initial_rise_threshold={initial_rise_threshold:.6f}, "
        f"main_fall_threshold={main_fall_threshold:.6f}"
    )

    ordered_candidates = sorted(
        candidates,
        key=lambda candidate: candidate.rise_path_index,
    )

    for index, candidate in enumerate(
        ordered_candidates,
        start=1,
    ):
        post_fall_drop = (
            candidate.before_height
            - candidate.threshold_height
        )

        print(
            f"[{index}] "
            f"rise_y={candidate.rise_position}, "
            f"validation_fall_y="
            f"{candidate.validation_fall_position}, "
            f"recovery_y="
            f"{candidate.validation_recovery_position}, "
            f"path_index={candidate.rise_path_index}, "
            f"gap={candidate.threshold_width_pixels}, "
            f"before={candidate.before_height:.6f}, "
            f"low={candidate.threshold_height:.6f}, "
            f"after={candidate.after_height:.6f}, "
            f"post_fall_drop={post_fall_drop:.6f}, "
            f"rise_strength={candidate.rise_strength:.6f}, "
            f"fall_strength={candidate.fall_strength:.6f}, "
            f"low_std={candidate.after_std:.6f}, "
            f"score={candidate.score:.4f}"
        )

    first_boundary_index = min(
        candidate.rise_path_index
        for candidate in candidates
    )

    near_first = [
        candidate
        for candidate in candidates
        if candidate.rise_path_index
        <= (
            first_boundary_index
            + config.halo_cluster_max_gap_pixels
        )
    ]

    selected = max(
        near_first,
        key=lambda candidate: candidate.score,
    )

    print(
        "\nSELECTED TOP BOUNDARY: "
        f"rise_y={selected.rise_position}, "
        f"fall_y={selected.validation_fall_position}, "
        f"score={selected.score:.4f}, "
        f"first_path_index={first_boundary_index}, "
        f"near_first_limit="
        f"{first_boundary_index + config.halo_cluster_max_gap_pixels}"
    )
    print("=" * 100 + "\n")

    return selected_candidate

def _find_stable_plateau(
    profile: NDArray[np.float64],
    robust_height_range: float,
    config: XpanderSegmentationConfig,
) -> tuple[int, int, float]:
    """
    Find the first stable window after leaving a reference corner.

    This is the explicit halo-overcoming step: transitions before this stable
    window are ignored and cannot be selected as Xpander thresholds.
    """
    window = max(3, config.stable_window_pixels)
    if profile.size < window:
        raise ValueError("Profile is too short to establish a stable Xpander plateau.")

    max_start = min(
        profile.size - window,
        max(
            config.stable_minimum_skip_pixels,
            int(round(profile.size * config.stable_search_max_fraction)),
        ),
    )

    std_limit = max(
        config.stable_max_std_fraction * robust_height_range,
        1e-8,
    )
    derivative_limit = max(
        config.stable_max_derivative_fraction * robust_height_range,
        1e-8,
    )

    derivative = np.gradient(profile)
    start_index = max(0, config.stable_minimum_skip_pixels)

    best: tuple[float, int, int, float] | None = None
    for start in range(start_index, max_start + 1):
        stop = start + window
        values = profile[start:stop]
        local_derivative = derivative[start:stop]
        local_std = float(np.std(values))
        local_slope = float(np.median(np.abs(local_derivative)))
        quality = local_std / std_limit + local_slope / derivative_limit

        if local_std <= std_limit and local_slope <= derivative_limit:
            return start, stop, float(np.median(values))

        if best is None or quality < best[0]:
            best = (quality, start, stop, float(np.median(values)))

    if best is None:
        raise ValueError("Could not establish a stable Xpander plateau after the halo.")

    # Controlled fallback: use the flattest early window, but keep the result
    # visible in diagnostics instead of selecting a shadow transition blindly.
    _, start, stop, height = best
    return start, stop, height


def _find_raised_threshold_transition(
    profile: NDArray[np.float64],
    path_positions: NDArray[np.int64],
    search_start_index: int,
    stable_height: float,
    robust_height_range: float,
    config: XpanderSegmentationConfig,
    require_after_background: bool,
    background_height: float,
) -> ThresholdTransition:
    """Find the first valid rise-then-fall after the stable Xpander plateau."""
    derivative = np.gradient(profile)
    derivative_median = float(np.median(derivative))
    derivative_mad = float(
        np.median(np.abs(derivative - derivative_median))
    )
    derivative_sigma = max(1.4826 * derivative_mad, 1e-8)
    edge_threshold = max(
        config.transition_min_gradient,
        config.transition_gradient_mad_multiplier * derivative_sigma,
    )

    positive_peaks, _ = find_peaks(
        derivative,
        height=edge_threshold,
        prominence=config.transition_min_prominence,
    )
    negative_peaks, _ = find_peaks(
        -derivative,
        height=edge_threshold,
        prominence=config.transition_min_prominence,
    )

    positive_peaks = positive_peaks[positive_peaks >= search_start_index]
    if positive_peaks.size == 0 or negative_peaks.size == 0:
        raise ValueError(
            "No raised threshold was found after the stable Xpander plateau."
        )

    maximum_width = min(
        config.threshold_max_width_pixels,
        max(
            config.threshold_min_width_pixels,
            int(round(path_positions.size * config.threshold_max_width_fraction)),
        ),
    )
    minimum_rise = max(
        config.minimum_threshold_rise_absolute,
        config.minimum_threshold_rise_fraction * robust_height_range,
    )
    plateau_tolerance = max(
        config.xpander_plateau_tolerance_fraction * robust_height_range,
        1e-8,
    )
    side_std_limit = max(
        config.side_region_max_std_fraction * robust_height_range,
        1e-8,
    )
    probe = max(2, config.transition_probe_pixels)

    candidates: list[ThresholdTransition] = []

    for rise_index in positive_peaks:
        valid_falls = negative_peaks[
            (negative_peaks > rise_index)
            & (negative_peaks - rise_index >= config.threshold_min_width_pixels)
            & (negative_peaks - rise_index <= maximum_width)
        ]

        for fall_index in valid_falls:
            before_start = max(search_start_index, rise_index - probe)
            before_values = profile[before_start:rise_index]
            threshold_values = profile[rise_index:fall_index + 1]
            after_start = min(profile.size, fall_index + 1)
            after_stop = min(profile.size, after_start + probe)
            after_values = profile[after_start:after_stop]

            if (
                before_values.size < 2
                or threshold_values.size < 1
                or after_values.size < 2
            ):
                continue

            before_height = float(np.median(before_values))
            threshold_height = float(np.median(threshold_values))
            after_height = float(np.median(after_values))
            before_std = float(np.std(before_values))
            after_std = float(np.std(after_values))
            threshold_rise = float(
                threshold_height - max(before_height, after_height)
            )

            # The inside region must still be the stable Xpander plateau.
            if abs(before_height - stable_height) > plateau_tolerance:
                continue
            if before_std > side_std_limit or after_std > side_std_limit:
                continue
            if threshold_rise < minimum_rise:
                continue

            if require_after_background:
                background_tolerance = max(
                    config.background_tolerance_fraction * robust_height_range,
                    1e-8,
                )
                background_std_limit = max(
                    config.background_max_std_fraction * robust_height_range,
                    1e-8,
                )
                if abs(after_height - background_height) > 2.5 * background_tolerance:
                    continue
                if after_std > background_std_limit:
                    continue

            rise_strength = float(derivative[rise_index])
            fall_strength = float(-derivative[fall_index])
            edge_score = float(
                1.0
                - np.exp(
                    -(rise_strength + fall_strength)
                    / (0.05 * robust_height_range + 1e-8)
                )
            )
            height_score = float(
                1.0
                - np.exp(
                    -threshold_rise
                    / (0.03 * robust_height_range + 1e-8)
                )
            )
            plateau_score = float(
                np.exp(-abs(before_height - stable_height) / plateau_tolerance)
            )
            flatness_score = float(
                np.exp(-(before_std + after_std) / (side_std_limit + 1e-8))
            )
            distance_score = float(
                np.exp(
                    -(rise_index - search_start_index)
                    / max(0.30 * profile.size, 1.0)
                )
            )

            if require_after_background:
                background_tolerance = max(
                    config.background_tolerance_fraction * robust_height_range,
                    1e-8,
                )
                background_score = float(
                    np.exp(-abs(after_height - background_height) / background_tolerance)
                )
            else:
                background_score = 1.0

            score = float(
                0.25 * edge_score
                + 0.25 * height_score
                + 0.20 * plateau_score
                + 0.10 * flatness_score
                + 0.10 * distance_score
                + 0.10 * background_score
            )

            candidates.append(
                ThresholdTransition(
                    rise_position=int(path_positions[rise_index]),
                    fall_position=int(path_positions[fall_index]),
                    rise_path_index=int(rise_index),
                    fall_path_index=int(fall_index),
                    before_height=before_height,
                    threshold_height=threshold_height,
                    after_height=after_height,
                    before_std=before_std,
                    after_std=after_std,
                    rise_strength=rise_strength,
                    fall_strength=fall_strength,
                    threshold_width_pixels=int(fall_index - rise_index),
                    threshold_rise=threshold_rise,
                    score=score,
                )
            )

            candidate = candidates[-1]
            print(
                f"ADD TRANSITION: "
                f"rise={candidate.rise_position}, "
                f"fall={candidate.fall_position}, "
                f"score={candidate.score:.4f}"
            )

    if not candidates:
        raise ValueError(
            "Raised edges existed, but none represented a stable "
            "Xpander -> threshold -> outside transition after the halo."
        )

    # Stop at the first valid threshold in the actual direction of travel.
    first_index = min(candidate.rise_path_index for candidate in candidates)
    nearby_limit = first_index + max(4, 2 * maximum_width)
    nearby = [
        candidate
        for candidate in candidates
        if candidate.rise_path_index <= nearby_limit
    ]
    return max(nearby, key=lambda candidate: candidate.score)


def _find_profile_peaks(
    profile: NDArray[np.float32] | NDArray[np.float64],
    start: int,
    stop: int,
    mad_multiplier: float,
    minimum_prominence: float,
) -> tuple[list[int], list[float]]:
    if stop <= start:
        return [], []

    segment = np.asarray(profile[start:stop], dtype=np.float64)
    median = float(np.median(segment))
    mad = float(np.median(np.abs(segment - median)))
    robust_sigma = max(1.4826 * mad, 1e-8)
    threshold = median + mad_multiplier * robust_sigma

    peaks, properties = find_peaks(
        segment,
        height=threshold,
        prominence=minimum_prominence,
    )
    positions = [start + int(index) for index in peaks]
    strengths = [
        float(value)
        for value in properties.get("peak_heights", segment[peaks])
    ]
    return positions, strengths


def _cluster_peaks(
    positions: list[int],
    strengths: list[float],
    max_gap: int,
) -> list[list[tuple[int, float]]]:
    if not positions:
        return []

    ordered = sorted(zip(positions, strengths), key=lambda item: item[0])
    clusters: list[list[tuple[int, float]]] = [[ordered[0]]]

    for position, strength in ordered[1:]:
        if position - clusters[-1][-1][0] <= max_gap:
            clusters[-1].append((position, strength))
        else:
            clusters.append([(position, strength)])
    return clusters


def _nearest_strong_cluster_to_pivot(
    clusters: list[list[tuple[int, float]]],
    side: SideName,
    pivot_box: BoundingBox,
    minimum_relative_strength: float,
) -> list[tuple[int, float]]:
    maximum_strength = max(
        item[1]
        for cluster in clusters
        for item in cluster
    )
    minimum_strength = minimum_relative_strength * maximum_strength
    strong_clusters = [
        cluster
        for cluster in clusters
        if max(item[1] for item in cluster) >= minimum_strength
    ]
    if not strong_clusters:
        strong_clusters = clusters

    if side == "left":
        return min(
            strong_clusters,
            key=lambda cluster: pivot_box.x_min - max(item[0] for item in cluster),
        )
    return min(
        strong_clusters,
        key=lambda cluster: min(item[0] for item in cluster) - pivot_box.x_max,
    )


def _nearest_strong_cluster_above_pivot(
    clusters: list[list[tuple[int, float]]],
    pivot_y: int,
    minimum_relative_strength: float,
) -> list[tuple[int, float]]:
    maximum_strength = max(
        item[1]
        for cluster in clusters
        for item in cluster
    )
    minimum_strength = minimum_relative_strength * maximum_strength
    strong_clusters = [
        cluster
        for cluster in clusters
        if max(item[1] for item in cluster) >= minimum_strength
    ]
    if not strong_clusters:
        strong_clusters = clusters

    # The nearest cluster reached while moving upward has the largest Y.
    return min(
        strong_clusters,
        key=lambda cluster: pivot_y - max(item[0] for item in cluster),
    )


def _cluster_score(
    cluster: list[tuple[int, float]],
    all_strengths: list[float],
) -> float:
    maximum = max(max(all_strengths), 1e-8)
    return float(max(item[1] for item in cluster) / maximum)


def _smooth_profile(
    profile: NDArray[np.float32] | NDArray[np.float64],
    sigma: float,
) -> NDArray[np.float64]:
    values = np.asarray(profile, dtype=np.float64)
    if sigma > 0:
        values = ndi.gaussian_filter1d(values, sigma=sigma)
    return values


def _estimate_background_height(
    smoothed: NDArray[np.float32],
    gradient_x: NDArray[np.float32],
    gradient_y: NDArray[np.float32],
) -> float:
    gradient_magnitude = np.hypot(gradient_x, gradient_y)
    flat_threshold = float(np.percentile(gradient_magnitude, 35))
    flat_values = smoothed[gradient_magnitude <= flat_threshold]
    if flat_values.size < 100:
        flat_values = smoothed.ravel()

    lower = float(np.percentile(flat_values, 1))
    upper = float(np.percentile(flat_values, 99))
    if upper <= lower:
        return float(np.median(flat_values))

    counts, edges = np.histogram(flat_values, bins=256, range=(lower, upper))
    best_bin = int(np.argmax(counts))
    return float(0.5 * (edges[best_bin] + edges[best_bin + 1]))


def _classify_outer_region(
    outer_height: float,
    background_height: float,
    robust_height_range: float,
    config: XpanderSegmentationConfig,
) -> OuterRegionType:
    tolerance = max(
        config.background_tolerance_fraction * robust_height_range,
        1e-8,
    )
    difference = abs(outer_height - background_height)
    if difference <= tolerance:
        return "background"
    if difference >= 2.0 * tolerance:
        return "neighbor"
    return "unknown"


def _polygon_to_mask(
    image_shape: tuple[int, int],
    polygon_points: tuple[PixelPoint, PixelPoint, PixelPoint, PixelPoint],
) -> BoolArray:
    image_height, image_width = image_shape
    x_values = np.array([point.x for point in polygon_points], dtype=np.float64)
    y_values = np.array([point.y for point in polygon_points], dtype=np.float64)

    x_min = max(0, int(np.floor(x_values.min())))
    x_max = min(image_width, int(np.ceil(x_values.max())) + 1)
    y_min = max(0, int(np.floor(y_values.min())))
    y_max = min(image_height, int(np.ceil(y_values.max())) + 1)

    polygon = MatplotlibPath(np.column_stack([x_values, y_values]))
    grid_y, grid_x = np.mgrid[y_min:y_max, x_min:x_max]
    points = np.column_stack([grid_x.ravel() + 0.5, grid_y.ravel() + 0.5])
    local_mask = polygon.contains_points(points, radius=1e-9).reshape(grid_y.shape)

    mask = np.zeros(image_shape, dtype=bool)
    mask[y_min:y_max, x_min:x_max] = local_mask
    return mask


def _validate_detected_geometry(
    pivot_box: BoundingBox,
    top_left: PixelPoint,
    top_right: PixelPoint,
    bottom_left: PixelPoint,
    bottom_right: PixelPoint,
    config: XpanderSegmentationConfig,
) -> None:
    if bottom_left.x >= bottom_right.x:
        raise ValueError("Xpander left boundary is not left of its right boundary.")
    if top_left.y >= bottom_left.y or top_right.y >= bottom_right.y:
        raise ValueError("Xpander top corners must be above the bottom corners.")

    width = bottom_right.x - bottom_left.x
    mean_height = 0.5 * (
        (bottom_left.y - top_left.y) + (bottom_right.y - top_right.y)
    )

    if width < config.minimum_xpander_width_fraction_of_pivot * pivot_box.width:
        raise ValueError("Detected Xpander width is too small relative to the Pivot.")
    if mean_height < config.minimum_xpander_height_fraction_of_pivot * pivot_box.height:
        raise ValueError("Detected Xpander height is too small relative to the Pivot.")

    max_bottom_difference = max(
        2.0,
        config.maximum_bottom_corner_y_difference_fraction * mean_height,
    )
    max_top_difference = max(
        2.0,
        config.maximum_top_corner_y_difference_fraction * mean_height,
    )
    if abs(bottom_left.y - bottom_right.y) > max_bottom_difference:
        raise ValueError("The two Xpander bottom corners are not horizontally consistent.")
    if abs(top_left.y - top_right.y) > max_top_difference:
        raise ValueError("The two Xpander top corners are not horizontally consistent.")


def _calculate_geometry_score(
    pivot_box: BoundingBox,
    top_left: PixelPoint,
    top_right: PixelPoint,
    bottom_left: PixelPoint,
    bottom_right: PixelPoint,
    config: XpanderSegmentationConfig,
) -> float:
    width = max(1.0, float(bottom_right.x - bottom_left.x))
    mean_height = max(
        1.0,
        0.5
        * (
            (bottom_left.y - top_left.y)
            + (bottom_right.y - top_right.y)
        ),
    )

    bottom_tolerance = max(
        1.0,
        config.maximum_bottom_corner_y_difference_fraction * mean_height,
    )
    top_tolerance = max(
        1.0,
        config.maximum_top_corner_y_difference_fraction * mean_height,
    )

    alignment_score = float(
        0.5 * np.exp(-abs(bottom_left.y - bottom_right.y) / bottom_tolerance)
        + 0.5 * np.exp(-abs(top_left.y - top_right.y) / top_tolerance)
    )
    width_score = float(
        1.0
        - np.exp(
            -width
            / max(
                config.minimum_xpander_width_fraction_of_pivot * pivot_box.width,
                1.0,
            )
        )
    )
    height_score = float(
        1.0
        - np.exp(
            -mean_height
            / max(
                config.minimum_xpander_height_fraction_of_pivot * pivot_box.height,
                1.0,
            )
        )
    )
    return float((alignment_score + width_score + height_score) / 3.0)


def _show_detected_xpander_corners(
    height_map: FloatArray,
    pivot_box: BoundingBox,
    detected_corners: list[tuple[str, PixelPoint]],
    title: str,
) -> None:
    """Unconditionally show every Xpander point detected so far."""
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(height_map, aspect="auto", interpolation="nearest")
    figure.colorbar(image, ax=axis, label="Height")

    axis.add_patch(
        Rectangle(
            (pivot_box.x_min, pivot_box.y_min),
            pivot_box.width,
            pivot_box.height,
            fill=False,
            edgecolor="white",
            linestyle="--",
            linewidth=1.2,
            label="Pivot",
        )
    )

    for index, (label, point) in enumerate(detected_corners):
        newest = index == len(detected_corners) - 1
        color = "red" if newest else "cyan"
        axis.scatter(
            point.x,
            point.y,
            color=color,
            marker="x",
            s=90 if newest else 55,
            linewidths=2.0 if newest else 1.4,
            label=f"{label}: ({point.x}, {point.y})",
        )
        axis.text(
            point.x + 5,
            point.y - 5,
            f"{label}\n({point.x}, {point.y})",
            color=color,
            fontsize=9,
        )

    axis.set_title(title)
    axis.set_xlabel("X [pixels]")
    axis.set_ylabel("Y [pixels]")
    axis.legend(loc="best")
    figure.tight_layout()
    plt.show()


def print_xpander_segmentation(result: XpanderSegmentationResult) -> None:
    """Print Stage-6 geometry, directions, halo handling and scores."""
    print("\nXpander segmentation:")
    print("-" * 100)
    print(f"Bounding box: {result.bounding_box}")
    print(
        "Corners: "
        f"TL={result.top_left}, TR={result.top_right}, "
        f"BL={result.bottom_left}, BR={result.bottom_right}"
    )
    print(
        "Reference corners after halo: "
        f"left={result.left_reference_corner.point}, "
        f"right={result.right_reference_corner.point}"
    )
    print(
        "Horizontal directions: left reference -> LEFT, "
        "right reference -> RIGHT"
    )

    for detection in (result.left_side_detection, result.right_side_detection):
        transition = detection.transition
        print(
            f"{detection.side.capitalize()} bottom: "
            f"corner={detection.bottom_corner}, "
            f"Xpander={detection.xpander_height:.6f}, "
            f"threshold={transition.threshold_height:.6f}, "
            f"outside={detection.outer_height:.6f} "
            f"({detection.outer_region_type}), "
            f"width={transition.threshold_width_pixels}px, "
            f"rise={transition.threshold_rise:.6f}, "
            f"score={transition.score:.4f}"
        )

    for detection in (result.left_top_detection, result.right_top_detection):
        print(
            f"{detection.side.capitalize()} top: "
            f"corner={detection.top_corner}, search_x={detection.search_x}, "
            f"background={detection.background_height_after_threshold:.6f}, "
            f"score={detection.score:.4f}"
        )

    print(f"Estimated background: {result.background_height:.6f}")
    print(f"Confidence: {result.confidence:.4f}")
    print(f"Confident: {result.is_confident}")
    print("-" * 100)


def plot_xpander_segmentation(
    height_map: FloatArray,
    pivot_segmentation: PivotSegmentationResult,
    result: XpanderSegmentationResult,
) -> None:
    """
    Display geometry, final label, and all four travel-order profiles.

    The profile plots explicitly show the stable plateau that is used to skip
    the shadow/halo before threshold detection.
    """
    pivot_box = pivot_segmentation.bounding_box
    xpander_box = result.bounding_box

    horizontal_padding = max(pivot_box.width, xpander_box.width) // 5
    vertical_padding = max(pivot_box.height, xpander_box.height) // 8
    crop_box = BoundingBox(
        x_min=max(
            0,
            min(
                result.left_reference_corner.point.x,
                xpander_box.x_min,
                pivot_box.x_min,
            )
            - horizontal_padding,
        ),
        y_min=max(0, xpander_box.y_min - vertical_padding),
        x_max=min(
            height_map.shape[1],
            max(
                result.right_reference_corner.point.x,
                xpander_box.x_max,
                pivot_box.x_max,
            )
            + horizontal_padding,
        ),
        y_max=min(height_map.shape[0], pivot_box.y_max + vertical_padding),
    )

    crop = height_map[crop_box.y_min:crop_box.y_max, crop_box.x_min:crop_box.x_max]
    mask_crop = result.xpander_mask[
        crop_box.y_min:crop_box.y_max,
        crop_box.x_min:crop_box.x_max,
    ]

    figure = plt.figure(figsize=(17, 12))
    grid = figure.add_gridspec(2, 2)
    geometry_axis = figure.add_subplot(grid[0, 0])
    mask_axis = figure.add_subplot(grid[0, 1])
    side_profile_axis = figure.add_subplot(grid[1, 0])
    top_profile_axis = figure.add_subplot(grid[1, 1])

    image = geometry_axis.imshow(crop, aspect="auto", interpolation="nearest")
    figure.colorbar(image, ax=geometry_axis, label="Height")

    polygon = np.array(
        [
            [point.x - crop_box.x_min, point.y - crop_box.y_min]
            for point in result.corners
        ],
        dtype=np.float64,
    )
    geometry_axis.add_patch(Polygon(polygon, fill=False, linewidth=3.0, label="Xpander"))

    pivot_local = _translate_box(pivot_box, -crop_box.x_min, -crop_box.y_min)
    geometry_axis.add_patch(
        Rectangle(
            (pivot_local.x_min, pivot_local.y_min),
            pivot_local.width,
            pivot_local.height,
            fill=False,
            linestyle="--",
            linewidth=2.0,
            label="Pivot",
        )
    )

    for point, label in (
        (result.left_reference_corner.point, "Left reference"),
        (result.right_reference_corner.point, "Right reference"),
        (result.bottom_left, "Bottom-left"),
        (result.bottom_right, "Bottom-right"),
        (result.top_left, "Top-left"),
        (result.top_right, "Top-right"),
    ):
        local_x = point.x - crop_box.x_min
        local_y = point.y - crop_box.y_min
        geometry_axis.scatter(local_x, local_y, marker="x", s=80, linewidths=2.5)
        geometry_axis.text(local_x + 3, local_y - 3, label)

    # Corrected horizontal movement: same outward direction after reference.
    geometry_axis.plot(
        [
            result.left_reference_corner.point.x - crop_box.x_min,
            result.bottom_left.x - crop_box.x_min,
        ],
        [
            result.left_reference_corner.point.y - crop_box.y_min,
            result.bottom_left.y - crop_box.y_min,
        ],
        linestyle=":",
        linewidth=2.0,
        label="Continue outward",
    )
    geometry_axis.plot(
        [
            result.right_reference_corner.point.x - crop_box.x_min,
            result.bottom_right.x - crop_box.x_min,
        ],
        [
            result.right_reference_corner.point.y - crop_box.y_min,
            result.bottom_right.y - crop_box.y_min,
        ],
        linestyle=":",
        linewidth=2.0,
    )
    geometry_axis.plot(
        [result.bottom_left.x - crop_box.x_min, result.top_left.x - crop_box.x_min],
        [result.bottom_left.y - crop_box.y_min, result.top_left.y - crop_box.y_min],
        linestyle=":",
        linewidth=2.0,
    )
    geometry_axis.plot(
        [result.bottom_right.x - crop_box.x_min, result.top_right.x - crop_box.x_min],
        [result.bottom_right.y - crop_box.y_min, result.top_right.y - crop_box.y_min],
        linestyle=":",
        linewidth=2.0,
    )
    geometry_axis.set_title("Stage 6 — corrected outward Xpander detection")
    geometry_axis.set_xlabel("Local X [pixels]")
    geometry_axis.set_ylabel("Local Y [pixels]")
    geometry_axis.legend()

    mask_axis.imshow(crop, aspect="auto", interpolation="nearest")
    mask_axis.imshow(
        np.ma.masked_where(~mask_crop, mask_crop),
        alpha=0.45,
        aspect="auto",
        interpolation="nearest",
    )
    mask_axis.add_patch(Polygon(polygon, fill=False, linewidth=3.0))
    mask_axis.set_title(
        "Final Xpander label\n"
        f"confidence={result.confidence:.3f}, confident={result.is_confident}"
    )
    mask_axis.set_xlabel("Local X [pixels]")
    mask_axis.set_ylabel("Local Y [pixels]")

    _plot_profile(
        side_profile_axis,
        result.left_side_detection.diagnostics,
        result.left_side_detection.transition,
        label="Left: travel toward decreasing X",
    )
    _plot_profile(
        side_profile_axis,
        result.right_side_detection.diagnostics,
        result.right_side_detection.transition,
        label="Right: travel toward increasing X",
    )
    side_profile_axis.set_title("Bottom-corner profiles — halo skipped first")
    side_profile_axis.set_xlabel("Travel index")
    side_profile_axis.set_ylabel("Height")
    side_profile_axis.legend()

    _plot_profile(
        top_profile_axis,
        result.left_top_detection.diagnostics,
        result.left_top_detection.transition,
        label="Left upward path",
    )
    _plot_profile(
        top_profile_axis,
        result.right_top_detection.diagnostics,
        result.right_top_detection.transition,
        label="Right upward path",
    )
    top_profile_axis.set_title("Top-corner profiles — decreasing Y")
    top_profile_axis.set_xlabel("Travel index")
    top_profile_axis.set_ylabel("Height")
    top_profile_axis.legend()

    figure.tight_layout()
    plt.show()


def get_xpander_segmentation(
    height_map: FloatArray,
    pivot_segmentation: PivotSegmentationResult,
    print_debug: bool = False,
    show_debug: bool = False,
    show_corner_debug: bool = False,
) -> XpanderSegmentationResult:
    """Run Step 6 through the same wrapper interface as the other stages."""
    with debug_print_context(print_debug):
        segmentation_result = segment_xpander(
            height_map=height_map,
            pivot_segmentation=pivot_segmentation,
            show_corner_debug=show_corner_debug,
        )

    if print_debug:
        print_xpander_segmentation(segmentation_result)

    if show_debug:
        plot_xpander_segmentation(
            height_map=height_map,
            pivot_segmentation=pivot_segmentation,
            result=segmentation_result,
        )

    return segmentation_result


def _plot_profile(
    axis: plt.Axes,
    diagnostics: PathDiagnostics,
    transition: ThresholdTransition,
    label: str,
) -> None:
    travel_index = np.arange(diagnostics.profile.size)
    axis.plot(travel_index, diagnostics.profile, label=label)
    axis.axvspan(
        diagnostics.stable_start_index,
        diagnostics.stable_stop_index,
        alpha=0.15,
    )
    axis.axvline(transition.rise_path_index, linestyle="--")
    if transition.validation_fall_path_index is not None:
        axis.axvline(
            transition.validation_fall_path_index,
            linestyle="-.",
        )
        if transition.validation_recovery_path_index is not None:
            axis.axvline(
                transition.validation_recovery_path_index,
                linestyle=":",
            )
    else:
        # Bottom-threshold transitions use the ordinary fall edge.
        axis.axvline(transition.fall_path_index, linestyle=":")


def _translate_box(box: BoundingBox, dx: int, dy: int) -> BoundingBox:
    return BoundingBox(
        x_min=box.x_min + dx,
        y_min=box.y_min + dy,
        x_max=box.x_max + dx,
        y_max=box.y_max + dy,
    )


def _validate_inputs(
    height_map: np.ndarray,
    pivot_segmentation: PivotSegmentationResult,
    config: XpanderSegmentationConfig,
) -> None:
    if height_map.ndim != 2:
        raise ValueError(
            f"Expected a 2-D height map, received shape {height_map.shape}."
        )
    if not np.issubdtype(height_map.dtype, np.number):
        raise TypeError(f"Expected numeric height data, received {height_map.dtype}.")
    if not np.isfinite(height_map).all():
        raise ValueError("Height map contains NaN or infinite values.")
    if pivot_segmentation.pivot_mask.shape != height_map.shape:
        raise ValueError("Pivot mask and height map must have the same shape.")
    if config.profile_band_half_width < 0:
        raise ValueError("profile_band_half_width cannot be negative.")
    if config.halo_cluster_max_gap_pixels < 1:
        raise ValueError("halo_cluster_max_gap_pixels must be at least 1.")
    if config.stable_window_pixels < 3:
        raise ValueError("stable_window_pixels must be at least 3.")
    if config.threshold_min_width_pixels < 1:
        raise ValueError("threshold_min_width_pixels must be at least 1.")
    if config.threshold_max_width_pixels < config.threshold_min_width_pixels:
        raise ValueError("threshold_max_width_pixels is smaller than the minimum.")

    wall_ranking_weights = (
        config.reference_wall_strength_weight
        + config.reference_wall_distance_weight
    )
    if not np.isclose(wall_ranking_weights, 1.0):
        raise ValueError(
            "Reference wall strength/distance weights must sum to 1.0."
        )
    if config.reference_wall_distance_scale_fraction <= 0:
        raise ValueError(
            "reference_wall_distance_scale_fraction must be positive."
        )
    if not 0.0 <= config.reference_wall_min_strength_score <= 1.0:
        raise ValueError(
            "reference_wall_min_strength_score must be in [0, 1]."
        )
    if not (
        0.0
        < config.reference_wall_strength_reference_percentile
        <= 100.0
    ):
        raise ValueError(
            "reference_wall_strength_reference_percentile must be in (0, 100]."
        )

    weights = (
        config.transition_score_weight
        + config.reference_score_weight
        + config.geometry_score_weight
    )
    if not np.isclose(weights, 1.0):
        raise ValueError("Xpander confidence weights must sum to 1.0.")