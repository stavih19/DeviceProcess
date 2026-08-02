from __future__ import annotations

import numpy as np

from AlgoSteps.lower_cross_detection import (
    LowerCrossDetectionResult,
    find_lower_cross_candidates,
    plot_lower_cross_detection,
    print_lower_cross_candidates,
)
from models import Point2D, RawAnalysisResult
from AlgoSteps.pivot_candidates import (
    LowerPlaneDetectionResult,
    find_lower_plane_candidates,
    plot_lower_plane_candidates,
    print_lower_plane_candidates,
)
from AlgoSteps.pivot_segmentation import (
    PivotSegmentationResult,
    plot_pivot_segmentation,
    print_pivot_segmentation,
    segment_pivot,
)
from AlgoSteps.upper_cross_detection import (
    UpperCrossDetectionResult,
    find_upper_cross_candidates,
    plot_upper_cross_detection,
    print_upper_cross_candidates,
)
from AlgoSteps.pivot_plane_height_difference import (
    PivotPlaneHeightDifferenceResult,
    measure_pivot_plane_height_difference,
    plot_pivot_plane_height_measurements,
    print_pivot_plane_height_difference,
)
from AlgoSteps.xpander_segmentation_v4 import (
    XpanderSegmentationResult,
    plot_xpander_segmentation,
    print_xpander_segmentation,
    segment_xpander,
)


def analyze_height_map(
    height_map: np.ndarray,
    pixel_size_um: float,
) -> RawAnalysisResult:
    """
    Analyze one 2D height map.

    Current implementation:
    1. Detect lower-plane candidates.
    2. Select the best lower-plane candidate.
    3. Detect lower-cross candidates inside that plane.
    4. Select the best lower-cross candidate.

    The complete Pivot and Xpander segmentation is not implemented yet.
    """
    del pixel_size_um  # Will be used in later measurement stages.

    # Stage 1: detect the lower Pivot plane.
    lower_plane_detection = get_lower_plane_detection(
        height_map=height_map,
        show_debug=True,
    )

    # Stage 2: detect the lower cross inside the selected plane.
    lower_cross_detection = get_lower_cross_detection(
        height_map=height_map,
        lower_plane_detection=lower_plane_detection,
        show_debug=False,
    )
    best_cross = lower_cross_detection.best_candidate
    cross_centers: list[Point2D] = []
    if best_cross is not None:
        cross_centers.append(
            Point2D(
                x=best_cross.center_x,
                y=best_cross.center_y,
            )
        )

    # Stage 3: label the complete Pivot.
    pivot_segmentation = get_pivot_segmentation(
        height_map=height_map,
        lower_plane_detection=lower_plane_detection,
        lower_cross_detection=lower_cross_detection,
        show_debug=False,
    )

    # Stage 4: detect the upper Pivot cross.
    upper_cross_detection = get_upper_cross_detection(
        height_map=height_map,
        pivot_segmentation=pivot_segmentation,
        lower_plane_detection=lower_plane_detection,
        lower_cross_detection=lower_cross_detection,
        show_debug=False,
    )

    # Stage 5: measure the height difference between the two Pivot surfaces.
    pivot_height_result = get_pivot_plane_height_difference(
        height_map=height_map,
        lower_plane_detection=lower_plane_detection,
        lower_cross_detection=lower_cross_detection,
        pivot_segmentation=pivot_segmentation,
        upper_cross_detection=upper_cross_detection,
        show_debug=False,
    )

    # Stage 6: detect and label the Xpander.
    xpander_segmentation = get_xpander_segmentation(
        height_map=height_map,
        pivot_segmentation=pivot_segmentation,
        show_debug=True,
    )

    # The Xpander label is not available yet, but the Pivot mask is.
    final_label_map = np.zeros(
        height_map.shape,
        dtype=np.uint8,
    )
    final_label_map[
        pivot_segmentation.pivot_mask
    ] = 1
    # final_label_map[
    #     xpander_segmentation.xpander_mask
    # ] = 2

    return RawAnalysisResult(
        status="not_implemented",
        pivot_height_difference_um=pivot_height_result.height_difference,
        pivot_cross_centers_px=cross_centers,
        label_map=final_label_map,
    )


def get_lower_plane_detection(
    height_map: np.ndarray,
    show_debug: bool = False,
) -> LowerPlaneDetectionResult:
    """
    Detect and select the best lower-plane candidate.

    Returns:
        The complete LowerPlaneDetectionResult, including all candidates,
        the selected best candidate, the binary mask and component labels.
    """
    detection_result = find_lower_plane_candidates(
        height_map
    )

    best_candidate = detection_result.best_candidate

    if best_candidate is None:
        raise ValueError(
            "No lower Pivot plane candidate was found."
        )

    print("\nSelected lower-plane candidate:")
    print(f"Component label: {best_candidate.component_label}")
    print(f"Score: {best_candidate.score:.4f}")
    print(f"Bounding box: {best_candidate.bounding_box}")
    print(f"Bounding-box corners: {best_candidate.bounding_box.corners}")
    print(
        "Centroid: "
        f"({best_candidate.centroid_x:.2f}, "
        f"{best_candidate.centroid_y:.2f})"
    )
    print(f"Area: {best_candidate.area_pixels} pixels")
    print(
        f"Rectangularity: "
        f"{best_candidate.rectangularity:.4f}"
    )
    print(
        f"Width/height ratio: "
        f"{best_candidate.width_height_ratio:.4f}"
    )
    print(
        f"Median height: "
        f"{best_candidate.median_height:.4f}"
    )
    print(
        f"Height MAD: "
        f"{best_candidate.height_mad:.4f}"
    )

    print_lower_plane_candidates(
        detection_result
    )

    if show_debug:
        plot_lower_plane_candidates(
            height_map=height_map,
            detection_result=detection_result,
        )

    return detection_result


def get_lower_cross_detection(
    height_map: np.ndarray,
    lower_plane_detection: LowerPlaneDetectionResult,
    show_debug: bool = False,
) -> LowerCrossDetectionResult:
    """
    Detect and select the best lower-cross candidate inside the selected
    lower-plane candidate.

    Returns:
        The complete LowerCrossDetectionResult.
    """
    best_plane = lower_plane_detection.best_candidate

    if best_plane is None:
        raise ValueError(
            "Lower-cross detection requires a valid lower-plane candidate."
        )

    detection_result = find_lower_cross_candidates(
        height_map=height_map,
        lower_plane_detection=lower_plane_detection,
        lower_plane_candidate=best_plane,
    )

    best_candidate = detection_result.best_candidate

    if best_candidate is None:
        raise ValueError(
            "No lower Pivot cross candidate was found."
        )

    # LowerCrossDetectionResult has local/global mask methods.
    # It does not have get_candidate_mask().
    lower_cross_mask = detection_result.get_candidate_mask_global(
        candidate=best_candidate,
        image_shape=height_map.shape,
    )

    # Median raw height is not a stored LowerCrossCandidate field,
    # so calculate it directly from the original height map.
    median_cross_height = float(
        np.median(height_map[lower_cross_mask])
    )

    print("\nSelected lower-cross candidate:")
    print(f"Component label: {best_candidate.component_label}")
    print(f"Score: {best_candidate.score:.4f}")
    print(f"Bounding box: {best_candidate.bounding_box}")
    print(
        f"Bounding-box corners: "
        f"{best_candidate.bounding_box.corners}"
    )
    print(
        "Center: "
        f"({best_candidate.center_x:.2f}, "
        f"{best_candidate.center_y:.2f})"
    )
    print(f"Area: {best_candidate.area_pixels} pixels")
    print(
        f"Area fraction: "
        f"{best_candidate.area_fraction:.6f}"
    )
    print(
        f"Width/height ratio: "
        f"{best_candidate.width_height_ratio:.4f}"
    )
    print(
        f"Fill ratio: "
        f"{best_candidate.fill_ratio:.4f}"
    )
    print(
        f"Horizontal-arm coverage: "
        f"{best_candidate.horizontal_arm_coverage:.4f}"
    )
    print(
        f"Vertical-arm coverage: "
        f"{best_candidate.vertical_arm_coverage:.4f}"
    )
    print(
        f"Corner occupancy: "
        f"{best_candidate.corner_occupancy:.4f}"
    )
    print(
        f"Mean local depth: "
        f"{best_candidate.mean_depth:.4f}"
    )
    print(
        f"Maximum local depth: "
        f"{best_candidate.max_depth:.4f}"
    )
    print(
        f"Median raw cross height: "
        f"{median_cross_height:.4f}"
    )

    print_lower_cross_candidates(
        detection_result
    )

    if show_debug:
        plot_lower_cross_detection(
            height_map=height_map,
            detection_result=detection_result,
        )

    return detection_result

def get_pivot_segmentation(
    height_map: np.ndarray,
    lower_plane_detection: LowerPlaneDetectionResult,
    lower_cross_detection: LowerCrossDetectionResult,
    show_debug: bool = False,
) -> PivotSegmentationResult:
    """
    Segment the Pivot region based on the detected lower-plane and
    lower-cross candidates.

    Returns:
        The complete segmentation result, including the Pivot mask and
        boundary diagnostics needed by subsequent detection stages.
    """
    segmentation_result = segment_pivot(
        height_map=height_map,
        lower_plane_detection=lower_plane_detection,
        lower_cross_detection=lower_cross_detection,
    )

    if show_debug:
        plot_pivot_segmentation(
            height_map=height_map,
            result=segmentation_result,
            lower_plane_detection=lower_plane_detection,
            lower_cross_detection=lower_cross_detection,
        )

    return segmentation_result


def get_upper_cross_detection(
    height_map: np.ndarray,
    pivot_segmentation: PivotSegmentationResult,
    lower_plane_detection: LowerPlaneDetectionResult,
    lower_cross_detection: LowerCrossDetectionResult,
    show_debug: bool = False,
) -> UpperCrossDetectionResult:
    """Detect the upper Pivot cross and optionally display its debug plot."""
    detection_result = find_upper_cross_candidates(
        height_map=height_map,
        pivot_segmentation=pivot_segmentation,
        lower_plane_detection=lower_plane_detection,
        lower_cross_detection=lower_cross_detection,
    )

    print_upper_cross_candidates(detection_result)

    best_candidate = detection_result.best_candidate
    if best_candidate is None:
        print(
            "The Pivot was segmented, but no upper cross "
            "candidate was found."
        )
    else:
        print("\nSelected upper cross:")
        print(
            "Center: "
            f"({best_candidate.center_x:.2f}, "
            f"{best_candidate.center_y:.2f})"
        )
        print(f"Bounding box: {best_candidate.bounding_box}")
        print(f"Score: {best_candidate.score:.4f}")

    if show_debug:
        plot_upper_cross_detection(
            height_map=height_map,
            result=detection_result,
            lower_plane_detection=lower_plane_detection,
            lower_cross_detection=lower_cross_detection,
        )

    return detection_result


def get_pivot_plane_height_difference(
    height_map: np.ndarray,
    lower_plane_detection: LowerPlaneDetectionResult,
    lower_cross_detection: LowerCrossDetectionResult,
    pivot_segmentation: PivotSegmentationResult,
    upper_cross_detection: UpperCrossDetectionResult,
    show_debug: bool = False,
) -> PivotPlaneHeightDifferenceResult:
    """Measure the Pivot plane heights and optionally display the debug plot."""
    measurement_result = measure_pivot_plane_height_difference(
        height_map=height_map,
        lower_plane_detection=lower_plane_detection,
        lower_cross_detection=lower_cross_detection,
        pivot_segmentation=pivot_segmentation,
        upper_cross_detection=upper_cross_detection,
    )

    print_pivot_plane_height_difference(measurement_result)

    if show_debug:
        plot_pivot_plane_height_measurements(
            height_map=height_map,
            result=measurement_result,
        )

    return measurement_result


def get_xpander_segmentation(
    height_map: np.ndarray,
    pivot_segmentation: PivotSegmentationResult,
    show_debug: bool = False,
) -> XpanderSegmentationResult:
    """
    Detect and label the Xpander and optionally display the detection graph.
    """
    segmentation_result = segment_xpander(
        height_map=height_map,
        pivot_segmentation=pivot_segmentation,
    )

    print_xpander_segmentation(
        segmentation_result
    )

    if show_debug:
        plot_xpander_segmentation(
            height_map=height_map,
            pivot_segmentation=pivot_segmentation,
            result=segmentation_result,
        )

    return segmentation_result