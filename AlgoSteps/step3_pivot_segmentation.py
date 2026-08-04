from __future__ import annotations

from AlgoSteps.debug_utils import debug_print_context

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from numpy.typing import NDArray
from scipy import ndimage as ndi
from scipy.signal import find_peaks

from AlgoSteps.step2_lower_cross_detection import LowerCrossDetectionResult
from AlgoSteps.step1_pivot_candidates import (
    BoolArray,
    BoundingBox,
    FloatArray,
    LowerPlaneDetectionResult,
)


@dataclass(frozen=True)
class PivotSegmentationConfig:
    """
    Configuration for Stage 3: complete Pivot labeling.

    This implementation is intended for Task 1, where the Pivot is axis-aligned.
    The lower plane and lower cross found in Stages 1 and 2 are used only as
    anchors. The final Pivot boundaries are detected from height transitions.
    """

    gaussian_sigma: float = 1.2

    # Patch used to estimate the height of the upper Pivot body.
    reference_y_start_fraction: float = 0.80
    reference_y_end_fraction: float = 0.35
    reference_half_width_fraction: float = 0.18

    # Search ranges relative to the lower-plane dimensions.
    horizontal_search_factor: float = 1.50
    vertical_search_above_factor: float = 1.80
    vertical_search_below_factor: float = 0.80

    # Prevent lower-plane edges from being selected as outer Pivot walls.
    minimum_side_gap_fraction: float = 0.15
    top_exclusion_fraction: float = 0.25
    bottom_gap_fraction: float = 0.05

    # Width of strips sampled on both sides of a possible edge.
    edge_sample_offset_pixels: int = 6

    # Peak detection.
    peak_min_prominence: float = 0.03
    peak_robust_prominence_multiplier: float = 2.0
    peak_min_distance_pixels: int = 4

    # Side bands used to detect the upper and lower horizontal boundaries.
    side_band_fraction: float = 0.15
    side_band_margin_pixels: int = 8

    # Optional expansion or contraction of the resulting rectangle.
    # Positive values expand; negative values contract.
    boundary_padding_pixels: int = 0

    # Candidate-score weights.
    inside_height_weight: float = 0.45
    height_step_weight: float = 0.25
    gradient_strength_weight: float = 0.20
    distance_weight: float = 0.10

    minimum_confidence: float = 0.45

    # Joint selection of all four boundaries. The selected left, right, top
    # and bottom walls should describe the same transition: the same Pivot
    # body height on the inside and the same background height on the outside.
    max_candidates_per_boundary: int = 8

    local_boundary_group_weight: float = 0.70
    boundary_consistency_weight: float = 0.30

    inside_consistency_weight: float = 0.40
    outside_consistency_weight: float = 0.30
    step_consistency_weight: float = 0.30

    inside_height_tolerance_fraction: float = 0.05
    outside_height_tolerance_fraction: float = 0.05
    height_step_tolerance_fraction: float = 0.08

    require_consistent_boundaries: bool = True


@dataclass(frozen=True)
class BoundaryCandidate:
    """Evidence for one detected Pivot boundary."""

    position: int
    score: float
    gradient_strength: float
    inside_height: float
    outside_height: float
    height_step: float
    distance_from_lower_plane: float


@dataclass(frozen=True)
class BoundaryConsistency:
    """Consistency of the height transition across all four Pivot walls."""

    inside_reference_height: float
    outside_reference_height: float
    reference_height_step: float

    inside_mean_deviation: float
    outside_mean_deviation: float
    step_mean_deviation: float

    inside_max_deviation: float
    outside_max_deviation: float
    step_max_deviation: float

    inside_score: float
    outside_score: float
    step_score: float
    score: float
    is_consistent: bool


@dataclass(frozen=True)
class _BoundaryGroupSelection:
    """Internal result of jointly selecting all four Pivot boundaries."""

    left: BoundaryCandidate
    right: BoundaryCandidate
    top: BoundaryCandidate
    bottom: BoundaryCandidate

    horizontal_x_indices: NDArray[np.int_]
    y_gradient_profile: NDArray[np.float32]

    local_boundary_score: float
    consistency: BoundaryConsistency
    score: float


@dataclass
class PivotSegmentationResult:
    """
    Stage-3 output.

    bounding_box:
        Axis-aligned box of the complete Pivot.

    pivot_mask:
        Boolean label mask with the same shape as the source height map.

    confidence:
        Mean score of the four selected boundaries.
    """

    bounding_box: BoundingBox
    pivot_mask: BoolArray
    search_roi: BoundingBox

    body_reference_height: float

    left_boundary: BoundaryCandidate
    right_boundary: BoundaryCandidate
    top_boundary: BoundaryCandidate
    bottom_boundary: BoundaryCandidate

    local_boundary_score: float
    boundary_consistency: BoundaryConsistency
    consistency_score: float

    confidence: float
    is_confident: bool

    x_gradient_profile: NDArray[np.float32]
    y_gradient_profile: NDArray[np.float32]


def segment_pivot(
    height_map: FloatArray,
    lower_plane_detection: LowerPlaneDetectionResult,
    lower_cross_detection: LowerCrossDetectionResult | None = None,
    config: PivotSegmentationConfig | None = None,
) -> PivotSegmentationResult:
    """
    Label the complete Pivot using the lower plane and lower cross as anchors.

    Processing:
    1. Estimate the height of the upper Pivot body.
    2. Detect the nearest valid left and right Pivot walls.
    3. Use side regions of the Pivot to detect the top and bottom boundaries.
    4. Create a filled boolean Pivot mask from the detected rectangle.

    The function does not use a fixed Pivot size. The scale of the search area
    is derived from the detected lower-plane bounding box.
    """
    if config is None:
        config = PivotSegmentationConfig()

    _validate_inputs(
        height_map=height_map,
        lower_plane_detection=lower_plane_detection,
        config=config,
    )

    lower_plane = lower_plane_detection.best_candidate
    if lower_plane is None:
        raise ValueError(
            "Pivot segmentation requires a valid lower-plane candidate."
        )

    plane_box = lower_plane.bounding_box
    image_height, image_width = height_map.shape

    # Prefer the lower-cross centre as the anchor. Fall back to the lower-plane
    # centroid if the cross was not detected.
    lower_cross = (
        lower_cross_detection.best_candidate
        if lower_cross_detection is not None
        else None
    )

    if lower_cross is not None:
        anchor_x = float(lower_cross.center_x)
        anchor_y = float(lower_cross.center_y)
    else:
        anchor_x = float(lower_plane.centroid_x)
        anchor_y = float(lower_plane.centroid_y)

    smoothed = ndi.gaussian_filter(
        height_map.astype(np.float32, copy=False),
        sigma=config.gaussian_sigma,
    )

    # Sobel is divided by 8 to keep the scale close to a central derivative.
    gradient_x = ndi.sobel(smoothed, axis=1) / 8.0
    gradient_y = ndi.sobel(smoothed, axis=0) / 8.0

    plane_width = plane_box.width
    plane_height = plane_box.height

    robust_height_range = float(
        np.percentile(smoothed, 99)
        - np.percentile(smoothed, 1)
    )
    robust_height_range = max(robust_height_range, 1e-8)

    # Estimate the normal height of the upper Pivot body from a centred patch
    # above the detected lower plane.
    reference_box = _create_reference_box(
        image_shape=height_map.shape,
        plane_box=plane_box,
        anchor_x=anchor_x,
        config=config,
    )

    reference_values = smoothed[
        reference_box.y_min:reference_box.y_max,
        reference_box.x_min:reference_box.x_max,
    ]

    if reference_values.size == 0:
        raise ValueError(
            "Could not create a valid Pivot-body reference region."
        )

    body_reference_height = float(
        np.median(reference_values)
    )

    search_roi = BoundingBox(
        x_min=max(
            0,
            int(
                np.floor(
                    anchor_x
                    - config.horizontal_search_factor * plane_width
                )
            ),
        ),
        y_min=max(
            0,
            int(
                np.floor(
                    plane_box.y_min
                    - config.vertical_search_above_factor * plane_height
                )
            ),
        ),
        x_max=min(
            image_width,
            int(
                np.ceil(
                    anchor_x
                    + config.horizontal_search_factor * plane_width
                )
            ),
        ),
        y_max=min(
            image_height,
            int(
                np.ceil(
                    plane_box.y_max
                    + config.vertical_search_below_factor * plane_height
                )
            ),
        ),
    )

    # Use the upper-body reference rows for side-wall detection. This avoids
    # the curved/high lower-plane region.
    reference_y_slice = slice(
        reference_box.y_min,
        reference_box.y_max,
    )

    x_gradient_profile = np.median(
        gradient_x[reference_y_slice, :],
        axis=0,
    ).astype(np.float32)

    minimum_side_gap = max(
        1,
        int(
            round(
                plane_width
                * config.minimum_side_gap_fraction
            )
        ),
    )

    left_search_start = search_roi.x_min
    left_search_stop = max(
        left_search_start + 1,
        plane_box.x_min - minimum_side_gap,
    )

    right_search_start = min(
        image_width - 1,
        plane_box.x_max + minimum_side_gap,
    )
    right_search_stop = search_roi.x_max

    left_positions = _find_edge_peaks(
        profile=x_gradient_profile,
        start=left_search_start,
        stop=left_search_stop,
        config=config,
    )

    right_positions = _find_edge_peaks(
        profile=-x_gradient_profile,
        start=right_search_start,
        stop=right_search_stop,
        config=config,
    )

    left_candidates = [
        _score_vertical_boundary(
            position=position,
            side="left",
            profile=x_gradient_profile,
            smoothed=smoothed,
            reference_y_slice=reference_y_slice,
            body_reference_height=body_reference_height,
            robust_height_range=robust_height_range,
            plane_box=plane_box,
            config=config,
        )
        for position in left_positions
    ]

    right_candidates = [
        _score_vertical_boundary(
            position=position,
            side="right",
            profile=x_gradient_profile,
            smoothed=smoothed,
            reference_y_slice=reference_y_slice,
            body_reference_height=body_reference_height,
            robust_height_range=robust_height_range,
            plane_box=plane_box,
            config=config,
        )
        for position in right_positions
    ]

    boundary_group = _select_best_boundary_group(
        left_candidates=left_candidates,
        right_candidates=right_candidates,
        gradient_y=gradient_y,
        smoothed=smoothed,
        search_roi=search_roi,
        body_reference_height=body_reference_height,
        robust_height_range=robust_height_range,
        plane_box=plane_box,
        config=config,
    )

    left_boundary = boundary_group.left
    right_boundary = boundary_group.right
    top_boundary = boundary_group.top
    bottom_boundary = boundary_group.bottom
    y_gradient_profile = boundary_group.y_gradient_profile

    padding = config.boundary_padding_pixels

    pivot_box = BoundingBox(
        x_min=max(
            0,
            left_boundary.position - padding,
        ),
        y_min=max(
            0,
            top_boundary.position - padding,
        ),
        x_max=min(
            image_width,
            right_boundary.position + 1 + padding,
        ),
        y_max=min(
            image_height,
            bottom_boundary.position + 1 + padding,
        ),
    )

    _validate_final_box(
        pivot_box=pivot_box,
        plane_box=plane_box,
        anchor_x=anchor_x,
        anchor_y=anchor_y,
    )

    pivot_mask = np.zeros(
        height_map.shape,
        dtype=bool,
    )
    pivot_mask[
        pivot_box.y_min:pivot_box.y_max,
        pivot_box.x_min:pivot_box.x_max,
    ] = True

    confidence = boundary_group.score
    consistency = boundary_group.consistency

    is_confident = confidence >= config.minimum_confidence
    if config.require_consistent_boundaries:
        is_confident = is_confident and consistency.is_consistent

    return PivotSegmentationResult(
        bounding_box=pivot_box,
        pivot_mask=pivot_mask,
        search_roi=search_roi,
        body_reference_height=body_reference_height,
        left_boundary=left_boundary,
        right_boundary=right_boundary,
        top_boundary=top_boundary,
        bottom_boundary=bottom_boundary,
        local_boundary_score=boundary_group.local_boundary_score,
        boundary_consistency=consistency,
        consistency_score=consistency.score,
        confidence=confidence,
        is_confident=is_confident,
        x_gradient_profile=x_gradient_profile,
        y_gradient_profile=y_gradient_profile,
    )


def print_pivot_segmentation(
    result: PivotSegmentationResult,
) -> None:
    """Print the complete Stage-3 result."""
    print("\nPivot segmentation:")
    print(f"Bounding box: {result.bounding_box}")
    print(f"Corners: {result.bounding_box.corners}")
    print(
        f"Size: "
        f"{result.bounding_box.width} x "
        f"{result.bounding_box.height} pixels"
    )
    print(
        f"Body reference height: "
        f"{result.body_reference_height:.4f}"
    )
    print(
        f"Local four-boundary score: "
        f"{result.local_boundary_score:.4f}"
    )
    print(
        f"Boundary consistency score: "
        f"{result.consistency_score:.4f}"
    )
    print(f"Confidence: {result.confidence:.4f}")
    print(f"Confident: {result.is_confident}")

    consistency = result.boundary_consistency
    print("\nFour-wall height consistency:")
    print(
        f"Inside reference height: "
        f"{consistency.inside_reference_height:.4f}, "
        f"mean deviation={consistency.inside_mean_deviation:.4f}, "
        f"max deviation={consistency.inside_max_deviation:.4f}"
    )
    print(
        f"Outside reference height: "
        f"{consistency.outside_reference_height:.4f}, "
        f"mean deviation={consistency.outside_mean_deviation:.4f}, "
        f"max deviation={consistency.outside_max_deviation:.4f}"
    )
    print(
        f"Reference height step: "
        f"{consistency.reference_height_step:.4f}, "
        f"mean deviation={consistency.step_mean_deviation:.4f}, "
        f"max deviation={consistency.step_max_deviation:.4f}"
    )
    print(
        f"Consistent walls: "
        f"{consistency.is_consistent}"
    )

    for name, boundary in (
        ("Left", result.left_boundary),
        ("Right", result.right_boundary),
        ("Top", result.top_boundary),
        ("Bottom", result.bottom_boundary),
    ):
        print(
            f"{name} boundary: "
            f"position={boundary.position}, "
            f"score={boundary.score:.4f}, "
            f"gradient={boundary.gradient_strength:.4f}, "
            f"inside_height={boundary.inside_height:.4f}, "
            f"outside_height={boundary.outside_height:.4f}, "
            f"step={boundary.height_step:.4f}"
        )


def plot_pivot_segmentation(
    height_map: FloatArray,
    result: PivotSegmentationResult,
    lower_plane_detection: LowerPlaneDetectionResult,
    lower_cross_detection: LowerCrossDetectionResult | None = None,
) -> None:
    """
    Display the detected Pivot bounding box and boolean label mask.
    """
    plane_candidate = lower_plane_detection.best_candidate
    if plane_candidate is None:
        raise ValueError(
            "Cannot plot Pivot segmentation without a lower-plane candidate."
        )

    crop_box = result.search_roi
    crop = height_map[
        crop_box.y_min:crop_box.y_max,
        crop_box.x_min:crop_box.x_max,
    ]
    mask_crop = result.pivot_mask[
        crop_box.y_min:crop_box.y_max,
        crop_box.x_min:crop_box.x_max,
    ]

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(16, 7),
    )

    height_image = axes[0].imshow(
        crop,
        cmap="viridis",
        aspect="auto",
    )
    figure.colorbar(
        height_image,
        ax=axes[0],
        label="Height",
    )

    pivot_box_local = _translate_box(
        result.bounding_box,
        dx=-crop_box.x_min,
        dy=-crop_box.y_min,
    )
    plane_box_local = _translate_box(
        plane_candidate.bounding_box,
        dx=-crop_box.x_min,
        dy=-crop_box.y_min,
    )

    axes[0].add_patch(
        Rectangle(
            (
                pivot_box_local.x_min,
                pivot_box_local.y_min,
            ),
            pivot_box_local.width,
            pivot_box_local.height,
            fill=False,
            edgecolor="lime",
            linewidth=3,
            label="Pivot",
        )
    )

    axes[0].add_patch(
        Rectangle(
            (
                plane_box_local.x_min,
                plane_box_local.y_min,
            ),
            plane_box_local.width,
            plane_box_local.height,
            fill=False,
            edgecolor="red",
            linewidth=2,
            linestyle="--",
            label="Lower plane",
        )
    )

    best_cross = (
        lower_cross_detection.best_candidate
        if lower_cross_detection is not None
        else None
    )

    if best_cross is not None:
        axes[0].scatter(
            best_cross.center_x - crop_box.x_min,
            best_cross.center_y - crop_box.y_min,
            marker="x",
            s=110,
            linewidths=3,
            color="red",
            label="Lower cross",
        )

    axes[0].set_title(
        "Detected complete Pivot"
    )
    axes[0].set_xlabel("Local X [pixels]")
    axes[0].set_ylabel("Local Y [pixels]")
    axes[0].legend()

    axes[1].imshow(
        crop,
        cmap="gray",
        aspect="auto",
    )
    axes[1].imshow(
        np.ma.masked_where(
            ~mask_crop,
            mask_crop,
        ),
        cmap="spring",
        alpha=0.45,
        aspect="auto",
    )
    axes[1].set_title(
        "Pivot label mask"
    )
    axes[1].set_xlabel("Local X [pixels]")
    axes[1].set_ylabel("Local Y [pixels]")

    figure.suptitle(
        f"Stage 3 — Pivot segmentation "
        f"(confidence={result.confidence:.3f}, "
        f"consistency={result.consistency_score:.3f})"
    )
    figure.tight_layout()
    plt.show()


def get_pivot_segmentation(
    height_map: FloatArray,
    lower_plane_detection: LowerPlaneDetectionResult,
    lower_cross_detection: LowerCrossDetectionResult,
    print_debug: bool = False,
    show_debug: bool = False,
) -> PivotSegmentationResult:
    """Segment the Pivot and optionally display its debug plot."""
    with debug_print_context(print_debug):
        segmentation_result = segment_pivot(
            height_map=height_map,
            lower_plane_detection=lower_plane_detection,
            lower_cross_detection=lower_cross_detection,
        )

    if print_debug:
        print_pivot_segmentation(segmentation_result)

    if show_debug:
        plot_pivot_segmentation(
            height_map=height_map,
            result=segmentation_result,
            lower_plane_detection=lower_plane_detection,
            lower_cross_detection=lower_cross_detection,
        )

    return segmentation_result


def _create_reference_box(
    image_shape: tuple[int, int],
    plane_box: BoundingBox,
    anchor_x: float,
    config: PivotSegmentationConfig,
) -> BoundingBox:
    image_height, image_width = image_shape

    y_min = int(
        round(
            plane_box.y_min
            - config.reference_y_start_fraction
            * plane_box.height
        )
    )
    y_max = int(
        round(
            plane_box.y_min
            - config.reference_y_end_fraction
            * plane_box.height
        )
    )

    half_width = max(
        2,
        int(
            round(
                config.reference_half_width_fraction
                * plane_box.width
            )
        ),
    )

    return BoundingBox(
        x_min=max(
            0,
            int(round(anchor_x)) - half_width,
        ),
        y_min=max(0, y_min),
        x_max=min(
            image_width,
            int(round(anchor_x)) + half_width + 1,
        ),
        y_max=min(
            image_height,
            max(y_min + 1, y_max),
        ),
    )


def _find_edge_peaks(
    profile: NDArray[np.float32],
    start: int,
    stop: int,
    config: PivotSegmentationConfig,
) -> list[int]:
    if stop <= start:
        raise ValueError(
            f"Invalid boundary search range: {start}:{stop}."
        )

    segment = np.asarray(
        profile[start:stop],
        dtype=np.float64,
    )

    median = float(np.median(segment))
    mad = float(
        np.median(
            np.abs(segment - median)
        )
    )
    robust_sigma = max(
        1.4826 * mad,
        1e-8,
    )

    prominence = max(
        config.peak_min_prominence,
        config.peak_robust_prominence_multiplier
        * robust_sigma,
    )

    peaks, _ = find_peaks(
        segment,
        prominence=prominence,
        distance=config.peak_min_distance_pixels,
    )

    if peaks.size == 0:
        peaks = np.array(
            [int(np.argmax(segment))],
            dtype=np.int64,
        )

    return [
        start + int(position)
        for position in peaks
    ]


def _score_vertical_boundary(
    position: int,
    side: str,
    profile: NDArray[np.float32],
    smoothed: NDArray[np.float32],
    reference_y_slice: slice,
    body_reference_height: float,
    robust_height_range: float,
    plane_box: BoundingBox,
    config: PivotSegmentationConfig,
) -> BoundaryCandidate:
    offset = config.edge_sample_offset_pixels
    image_width = smoothed.shape[1]

    if side == "left":
        inside_slice = slice(
            min(image_width, position + offset),
            min(image_width, position + 2 * offset),
        )
        outside_slice = slice(
            max(0, position - 2 * offset),
            max(0, position - offset),
        )
        gradient_strength = max(
            float(profile[position]),
            0.0,
        )
        distance = float(
            plane_box.x_min - position
        )
    elif side == "right":
        inside_slice = slice(
            max(0, position - 2 * offset),
            max(0, position - offset),
        )
        outside_slice = slice(
            min(image_width, position + offset),
            min(image_width, position + 2 * offset),
        )
        gradient_strength = max(
            float(-profile[position]),
            0.0,
        )
        distance = float(
            position - plane_box.x_max
        )
    else:
        raise ValueError(
            f"Unsupported vertical side: {side}."
        )

    inside_values = smoothed[
        reference_y_slice,
        inside_slice,
    ]
    outside_values = smoothed[
        reference_y_slice,
        outside_slice,
    ]

    if inside_values.size == 0 or outside_values.size == 0:
        return BoundaryCandidate(
            position=position,
            score=0.0,
            gradient_strength=gradient_strength,
            inside_height=float("nan"),
            outside_height=float("nan"),
            height_step=0.0,
            distance_from_lower_plane=distance,
        )

    inside_height = float(
        np.median(inside_values)
    )
    outside_height = float(
        np.median(outside_values)
    )

    return _create_boundary_candidate(
        position=position,
        gradient_strength=gradient_strength,
        inside_height=inside_height,
        outside_height=outside_height,
        body_reference_height=body_reference_height,
        robust_height_range=robust_height_range,
        distance=distance,
        distance_scale=max(
            plane_box.width * 0.60,
            1.0,
        ),
        config=config,
    )


def _score_horizontal_boundary(
    position: int,
    side: str,
    profile: NDArray[np.float32],
    smoothed: NDArray[np.float32],
    x_indices: NDArray[np.int_],
    body_reference_height: float,
    robust_height_range: float,
    plane_box: BoundingBox,
    config: PivotSegmentationConfig,
) -> BoundaryCandidate:
    offset = config.edge_sample_offset_pixels
    image_height = smoothed.shape[0]

    if side == "top":
        inside_rows = slice(
            min(image_height, position + offset),
            min(image_height, position + 2 * offset),
        )
        outside_rows = slice(
            max(0, position - 2 * offset),
            max(0, position - offset),
        )
        gradient_strength = max(
            float(profile[position]),
            0.0,
        )
        distance = float(
            plane_box.y_min - position
        )
    elif side == "bottom":
        inside_rows = slice(
            max(0, position - 2 * offset),
            max(0, position - offset),
        )
        outside_rows = slice(
            min(image_height, position + offset),
            min(image_height, position + 2 * offset),
        )
        gradient_strength = max(
            float(-profile[position]),
            0.0,
        )
        distance = float(
            position - plane_box.y_max
        )
    else:
        raise ValueError(
            f"Unsupported horizontal side: {side}."
        )

    inside_values = smoothed[
        inside_rows,
        :,
    ][:, x_indices]
    outside_values = smoothed[
        outside_rows,
        :,
    ][:, x_indices]

    if inside_values.size == 0 or outside_values.size == 0:
        return BoundaryCandidate(
            position=position,
            score=0.0,
            gradient_strength=gradient_strength,
            inside_height=float("nan"),
            outside_height=float("nan"),
            height_step=0.0,
            distance_from_lower_plane=distance,
        )

    inside_height = float(
        np.median(inside_values)
    )
    outside_height = float(
        np.median(outside_values)
    )

    return _create_boundary_candidate(
        position=position,
        gradient_strength=gradient_strength,
        inside_height=inside_height,
        outside_height=outside_height,
        body_reference_height=body_reference_height,
        robust_height_range=robust_height_range,
        distance=distance,
        distance_scale=max(
            plane_box.height * 0.80,
            1.0,
        ),
        config=config,
    )


def _create_boundary_candidate(
    position: int,
    gradient_strength: float,
    inside_height: float,
    outside_height: float,
    body_reference_height: float,
    robust_height_range: float,
    distance: float,
    distance_scale: float,
    config: PivotSegmentationConfig,
) -> BoundaryCandidate:
    height_step = max(
        inside_height - outside_height,
        0.0,
    )

    inside_height_score = float(
        np.exp(
            -abs(
                inside_height
                - body_reference_height
            )
            / (
                0.15
                * robust_height_range
            )
        )
    )

    height_step_score = float(
        1.0
        - np.exp(
            -height_step
            / (
                0.08
                * robust_height_range
            )
        )
    )

    gradient_strength_score = float(
        1.0
        - np.exp(
            -gradient_strength
            / (
                0.05
                * robust_height_range
            )
        )
    )

    distance_score = float(
        np.exp(
            -max(distance, 0.0)
            / distance_scale
        )
    )

    score = (
        config.inside_height_weight
        * inside_height_score
        + config.height_step_weight
        * height_step_score
        + config.gradient_strength_weight
        * gradient_strength_score
        + config.distance_weight
        * distance_score
    )

    return BoundaryCandidate(
        position=position,
        score=float(score),
        gradient_strength=gradient_strength,
        inside_height=inside_height,
        outside_height=outside_height,
        height_step=height_step,
        distance_from_lower_plane=distance,
    )


def _create_horizontal_boundary_x_indices(
    left_x: int,
    right_x: int,
    plane_box: BoundingBox,
    config: PivotSegmentationConfig,
) -> NDArray[np.int_]:
    pivot_width = right_x - left_x

    if pivot_width <= 0:
        raise ValueError(
            "Cannot create horizontal-boundary bands from invalid side walls."
        )

    margin = config.side_band_margin_pixels
    band_width = max(
        5,
        int(
            round(
                pivot_width
                * config.side_band_fraction
            )
        ),
    )

    left_start = left_x + margin
    left_stop = min(
        plane_box.x_min - margin,
        left_start + band_width,
    )

    right_stop = right_x - margin
    right_start = max(
        plane_box.x_max + margin,
        right_stop - band_width,
    )

    index_groups: list[NDArray[np.int_]] = []

    if left_stop > left_start:
        index_groups.append(
            np.arange(
                left_start,
                left_stop,
                dtype=np.int_,
            )
        )

    if right_stop > right_start:
        index_groups.append(
            np.arange(
                right_start,
                right_stop,
                dtype=np.int_,
            )
        )

    if not index_groups:
        # Fallback: use the complete interior, excluding only a small margin.
        fallback_start = left_x + margin
        fallback_stop = right_x - margin

        if fallback_stop <= fallback_start:
            raise ValueError(
                "The detected Pivot is too narrow for horizontal-boundary "
                "analysis."
            )

        return np.arange(
            fallback_start,
            fallback_stop,
            dtype=np.int_,
        )

    return np.concatenate(index_groups)


def _select_best_boundary_group(
    left_candidates: list[BoundaryCandidate],
    right_candidates: list[BoundaryCandidate],
    gradient_y: NDArray[np.float32],
    smoothed: NDArray[np.float32],
    search_roi: BoundingBox,
    body_reference_height: float,
    robust_height_range: float,
    plane_box: BoundingBox,
    config: PivotSegmentationConfig,
) -> _BoundaryGroupSelection:
    """
    Jointly select left, right, top and bottom Pivot boundaries.

    The four walls are not selected independently. Every valid combination is
    scored using both the individual wall evidence and the consistency of the
    measured inside heights, outside heights and height steps.
    """
    ranked_left = _rank_boundary_candidates(
        left_candidates,
        "left",
        config.max_candidates_per_boundary,
    )
    ranked_right = _rank_boundary_candidates(
        right_candidates,
        "right",
        config.max_candidates_per_boundary,
    )

    image_height = smoothed.shape[0]
    plane_height = plane_box.height
    best_group: _BoundaryGroupSelection | None = None

    for left_boundary in ranked_left:
        for right_boundary in ranked_right:
            if left_boundary.position >= right_boundary.position:
                continue

            try:
                horizontal_x_indices = (
                    _create_horizontal_boundary_x_indices(
                        left_x=left_boundary.position,
                        right_x=right_boundary.position,
                        plane_box=plane_box,
                        config=config,
                    )
                )
            except ValueError:
                continue

            y_gradient_profile = np.median(
                gradient_y[:, horizontal_x_indices],
                axis=1,
            ).astype(np.float32)

            top_search_start = search_roi.y_min
            top_search_stop = max(
                top_search_start + 1,
                int(
                    np.floor(
                        plane_box.y_min
                        - config.top_exclusion_fraction * plane_height
                    )
                ),
            )

            bottom_search_start = min(
                image_height - 1,
                int(
                    np.ceil(
                        plane_box.y_max
                        + config.bottom_gap_fraction * plane_height
                    )
                ),
            )
            bottom_search_stop = search_roi.y_max

            try:
                top_positions = _find_edge_peaks(
                    profile=y_gradient_profile,
                    start=top_search_start,
                    stop=top_search_stop,
                    config=config,
                )
                bottom_positions = _find_edge_peaks(
                    profile=-y_gradient_profile,
                    start=bottom_search_start,
                    stop=bottom_search_stop,
                    config=config,
                )
            except ValueError:
                continue

            top_candidates = [
                _score_horizontal_boundary(
                    position=position,
                    side="top",
                    profile=y_gradient_profile,
                    smoothed=smoothed,
                    x_indices=horizontal_x_indices,
                    body_reference_height=body_reference_height,
                    robust_height_range=robust_height_range,
                    plane_box=plane_box,
                    config=config,
                )
                for position in top_positions
            ]

            bottom_candidates = [
                _score_horizontal_boundary(
                    position=position,
                    side="bottom",
                    profile=y_gradient_profile,
                    smoothed=smoothed,
                    x_indices=horizontal_x_indices,
                    body_reference_height=body_reference_height,
                    robust_height_range=robust_height_range,
                    plane_box=plane_box,
                    config=config,
                )
                for position in bottom_positions
            ]

            ranked_top = _rank_boundary_candidates(
                top_candidates,
                "top",
                config.max_candidates_per_boundary,
            )
            ranked_bottom = _rank_boundary_candidates(
                bottom_candidates,
                "bottom",
                config.max_candidates_per_boundary,
            )

            for top_boundary in ranked_top:
                for bottom_boundary in ranked_bottom:
                    if top_boundary.position >= bottom_boundary.position:
                        continue

                    selected_boundaries = [
                        left_boundary,
                        right_boundary,
                        top_boundary,
                        bottom_boundary,
                    ]

                    if not _all_boundary_measurements_are_finite(
                        selected_boundaries
                    ):
                        continue

                    consistency = _calculate_boundary_consistency(
                        boundaries=selected_boundaries,
                        robust_height_range=robust_height_range,
                        config=config,
                    )

                    local_boundary_score = float(
                        np.mean(
                            [
                                boundary.score
                                for boundary in selected_boundaries
                            ]
                        )
                    )

                    group_score = float(
                        config.local_boundary_group_weight
                        * local_boundary_score
                        + config.boundary_consistency_weight
                        * consistency.score
                    )

                    candidate_group = _BoundaryGroupSelection(
                        left=left_boundary,
                        right=right_boundary,
                        top=top_boundary,
                        bottom=bottom_boundary,
                        horizontal_x_indices=horizontal_x_indices,
                        y_gradient_profile=y_gradient_profile,
                        local_boundary_score=local_boundary_score,
                        consistency=consistency,
                        score=group_score,
                    )

                    if (
                        best_group is None
                        or candidate_group.score > best_group.score
                    ):
                        best_group = candidate_group

    if best_group is None:
        raise ValueError(
            "No geometrically valid and measurable combination of four "
            "Pivot boundaries was found."
        )

    return best_group


def _rank_boundary_candidates(
    candidates: list[BoundaryCandidate],
    boundary_name: str,
    maximum_count: int,
) -> list[BoundaryCandidate]:
    valid_candidates = [
        candidate
        for candidate in candidates
        if np.isfinite(candidate.score)
    ]

    if not valid_candidates:
        raise ValueError(
            f"No {boundary_name} Pivot-boundary candidates were found."
        )

    return sorted(
        valid_candidates,
        key=lambda candidate: candidate.score,
        reverse=True,
    )[:maximum_count]


def _calculate_boundary_consistency(
    boundaries: list[BoundaryCandidate],
    robust_height_range: float,
    config: PivotSegmentationConfig,
) -> BoundaryConsistency:
    """
    Check that all four walls have similar heights before and after the drop.

    "Inside" means the Pivot-body side of a wall and "outside" means the
    surrounding-background side. Equality is not required; configurable robust
    tolerances are used because the height map contains measurement noise.
    """
    inside_heights = np.asarray(
        [boundary.inside_height for boundary in boundaries],
        dtype=np.float64,
    )
    outside_heights = np.asarray(
        [boundary.outside_height for boundary in boundaries],
        dtype=np.float64,
    )
    height_steps = np.asarray(
        [boundary.height_step for boundary in boundaries],
        dtype=np.float64,
    )

    inside_reference = float(np.median(inside_heights))
    outside_reference = float(np.median(outside_heights))
    step_reference = float(np.median(height_steps))

    inside_deviations = np.abs(inside_heights - inside_reference)
    outside_deviations = np.abs(outside_heights - outside_reference)
    step_deviations = np.abs(height_steps - step_reference)

    inside_mean_deviation = float(np.mean(inside_deviations))
    outside_mean_deviation = float(np.mean(outside_deviations))
    step_mean_deviation = float(np.mean(step_deviations))

    inside_max_deviation = float(np.max(inside_deviations))
    outside_max_deviation = float(np.max(outside_deviations))
    step_max_deviation = float(np.max(step_deviations))

    inside_tolerance = max(
        config.inside_height_tolerance_fraction * robust_height_range,
        1e-8,
    )
    outside_tolerance = max(
        config.outside_height_tolerance_fraction * robust_height_range,
        1e-8,
    )
    step_tolerance = max(
        config.height_step_tolerance_fraction * robust_height_range,
        1e-8,
    )

    inside_score = float(
        np.exp(-inside_mean_deviation / inside_tolerance)
    )
    outside_score = float(
        np.exp(-outside_mean_deviation / outside_tolerance)
    )
    step_score = float(
        np.exp(-step_mean_deviation / step_tolerance)
    )

    consistency_score = float(
        config.inside_consistency_weight * inside_score
        + config.outside_consistency_weight * outside_score
        + config.step_consistency_weight * step_score
    )

    is_consistent = bool(
        inside_max_deviation <= inside_tolerance
        and outside_max_deviation <= outside_tolerance
        and step_max_deviation <= step_tolerance
    )

    return BoundaryConsistency(
        inside_reference_height=inside_reference,
        outside_reference_height=outside_reference,
        reference_height_step=step_reference,
        inside_mean_deviation=inside_mean_deviation,
        outside_mean_deviation=outside_mean_deviation,
        step_mean_deviation=step_mean_deviation,
        inside_max_deviation=inside_max_deviation,
        outside_max_deviation=outside_max_deviation,
        step_max_deviation=step_max_deviation,
        inside_score=inside_score,
        outside_score=outside_score,
        step_score=step_score,
        score=consistency_score,
        is_consistent=is_consistent,
    )


def _all_boundary_measurements_are_finite(
    boundaries: list[BoundaryCandidate],
) -> bool:
    values = np.asarray(
        [
            value
            for boundary in boundaries
            for value in (
                boundary.score,
                boundary.gradient_strength,
                boundary.inside_height,
                boundary.outside_height,
                boundary.height_step,
            )
        ],
        dtype=np.float64,
    )

    return bool(np.isfinite(values).all())


def _validate_final_box(
    pivot_box: BoundingBox,
    plane_box: BoundingBox,
    anchor_x: float,
    anchor_y: float,
) -> None:
    if (
        pivot_box.width <= 0
        or pivot_box.height <= 0
    ):
        raise ValueError(
            "Detected Pivot bounding box has invalid dimensions."
        )

    if not (
        pivot_box.x_min
        <= plane_box.x_min
        and pivot_box.x_max
        >= plane_box.x_max
        and pivot_box.y_min
        <= plane_box.y_min
        and pivot_box.y_max
        >= plane_box.y_max
    ):
        raise ValueError(
            "Detected Pivot box does not contain the lower-plane candidate."
        )

    if not (
        pivot_box.x_min
        <= anchor_x
        < pivot_box.x_max
        and pivot_box.y_min
        <= anchor_y
        < pivot_box.y_max
    ):
        raise ValueError(
            "Detected Pivot box does not contain the selected anchor."
        )


def _translate_box(
    box: BoundingBox,
    dx: int,
    dy: int,
) -> BoundingBox:
    return BoundingBox(
        x_min=box.x_min + dx,
        y_min=box.y_min + dy,
        x_max=box.x_max + dx,
        y_max=box.y_max + dy,
    )


def _validate_inputs(
    height_map: np.ndarray,
    lower_plane_detection: LowerPlaneDetectionResult,
    config: PivotSegmentationConfig,
) -> None:
    if height_map.ndim != 2:
        raise ValueError(
            "Expected a two-dimensional height map, "
            f"received shape {height_map.shape}."
        )

    if not np.issubdtype(
        height_map.dtype,
        np.number,
    ):
        raise TypeError(
            f"Expected numeric height data, received {height_map.dtype}."
        )

    if not np.isfinite(height_map).all():
        raise ValueError(
            "Height map contains NaN or infinite values."
        )

    if (
        lower_plane_detection.label_image.shape
        != height_map.shape
    ):
        raise ValueError(
            "Lower-plane detection and height map must have the same shape."
        )

    if config.gaussian_sigma < 0:
        raise ValueError(
            "gaussian_sigma cannot be negative."
        )

    if config.edge_sample_offset_pixels < 1:
        raise ValueError(
            "edge_sample_offset_pixels must be at least 1."
        )

    if config.peak_min_distance_pixels < 1:
        raise ValueError(
            "peak_min_distance_pixels must be at least 1."
        )

    if not 0 <= config.minimum_side_gap_fraction < 1:
        raise ValueError(
            "minimum_side_gap_fraction must be in [0, 1)."
        )

    if config.max_candidates_per_boundary < 1:
        raise ValueError(
            "max_candidates_per_boundary must be at least 1."
        )

    tolerance_fractions = (
        config.inside_height_tolerance_fraction,
        config.outside_height_tolerance_fraction,
        config.height_step_tolerance_fraction,
    )
    if any(value <= 0 for value in tolerance_fractions):
        raise ValueError(
            "Boundary-consistency tolerance fractions must be positive."
        )

    local_weights = (
        config.inside_height_weight
        + config.height_step_weight
        + config.gradient_strength_weight
        + config.distance_weight
    )

    if not np.isclose(local_weights, 1.0):
        raise ValueError(
            "Pivot-boundary score weights must sum to 1.0."
        )

    group_weights = (
        config.local_boundary_group_weight
        + config.boundary_consistency_weight
    )
    if not np.isclose(group_weights, 1.0):
        raise ValueError(
            "Boundary-group score weights must sum to 1.0."
        )

    consistency_weights = (
        config.inside_consistency_weight
        + config.outside_consistency_weight
        + config.step_consistency_weight
    )
    if not np.isclose(consistency_weights, 1.0):
        raise ValueError(
            "Boundary-consistency component weights must sum to 1.0."
        )
