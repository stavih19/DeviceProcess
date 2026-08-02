from __future__ import annotations

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
    from AlgoSteps.pivot_candidates import (
        BoolArray,
        BoundingBox,
        FloatArray,
    )
    from AlgoSteps.pivot_segmentation import PivotSegmentationResult
except ImportError:
    from pivot_candidates import (
        BoolArray,
        BoundingBox,
        FloatArray,
    )
    from pivot_segmentation import PivotSegmentationResult


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

    # Side-wall search outside the Pivot.
    side_wall_min_gap_fraction: float = 0.03
    side_wall_max_distance_factor: float = 3.0
    side_wall_gradient_mad_multiplier: float = 2.5
    side_wall_min_prominence: float = 0.01
    side_wall_cluster_min_relative_strength: float = 0.25

    # Reference-corner search above the Pivot.
    reference_corner_gap_fraction: float = 0.04
    reference_corner_max_distance_factor: float = 3.0
    reference_corner_gradient_mad_multiplier: float = 2.5
    reference_corner_min_prominence: float = 0.01
    reference_corner_cluster_min_relative_strength: float = 0.25

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

    try:
        left_reference = _find_reference_corner(
            side="left",
            gradient_x=gradient_x,
            gradient_y=gradient_y,
            pivot_box=pivot_box,
            config=config,
        )
    except ValueError as error:
        raise ValueError(f"Left Xpander reference corner failed: {error}") from error

    try:
        right_reference = _find_reference_corner(
            side="right",
            gradient_x=gradient_x,
            gradient_y=gradient_y,
            pivot_box=pivot_box,
            config=config,
        )
    except ValueError as error:
        raise ValueError(f"Right Xpander reference corner failed: {error}") from error

    try:
        left_side = _find_bottom_xpander_corner(
            side="left",
            smoothed=smoothed,
            reference_corner=left_reference,
            background_height=background_height,
            robust_height_range=robust_height_range,
            config=config,
        )
    except ValueError as error:
        raise ValueError(f"Left bottom Xpander corner failed: {error}") from error

    try:
        right_side = _find_bottom_xpander_corner(
            side="right",
            smoothed=smoothed,
            reference_corner=right_reference,
            background_height=background_height,
            robust_height_range=robust_height_range,
            config=config,
        )
    except ValueError as error:
        raise ValueError(f"Right bottom Xpander corner failed: {error}") from error

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
    geometry_score = _calculate_geometry_score(
        pivot_box=pivot_box,
        top_left=top_left,
        top_right=top_right,
        bottom_left=bottom_left,
        bottom_right=bottom_right,
        config=config,
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


def _find_bottom_xpander_corner(
    side: SideName,
    smoothed: NDArray[np.float32],
    reference_corner: ReferenceCorner,
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
    reference_y = reference_corner.point.y

    y_min = max(0, reference_y - config.profile_band_half_width)
    y_max = min(image_height, reference_y + config.profile_band_half_width + 1)
    horizontal_profile = np.median(smoothed[y_min:y_max, :], axis=0)

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
        bottom_corner=PixelPoint(x=int(bottom_x), y=int(reference_y)),
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

            before_std = float(np.std(before_values))
            low_std = float(np.std(low_values))

            initial_rise_amount = raised_height - before_height
            main_fall_amount = raised_height - low_height

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

    # Prefer the first valid boundary encountered after the stable Xpander
    # plateau. Score resolves candidates beginning at nearly the same edge and
    # gives a meaningful boost when optional recovery is present.
    first_boundary_index = min(
        candidate.rise_path_index
        for candidate in candidates
    )
    near_first = [
        candidate
        for candidate in candidates
        if candidate.rise_path_index
        <= first_boundary_index + config.halo_cluster_max_gap_pixels
    ]

    return max(
        near_first,
        key=lambda candidate: candidate.score,
    )

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

    weights = (
        config.transition_score_weight
        + config.reference_score_weight
        + config.geometry_score_weight
    )
    if not np.isclose(weights, 1.0):
        raise ValueError("Xpander confidence weights must sum to 1.0.")
