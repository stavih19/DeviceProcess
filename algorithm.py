from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from AlgoSteps.step2_lower_cross_detection import (
    get_lower_cross_detection,
)
from AlgoSteps.step1_pivot_candidates import (
    get_lower_plane_detection,
)
from AlgoSteps.step5_pivot_plane_height_difference import (
    get_pivot_plane_height_difference,
)
from AlgoSteps.step3_pivot_segmentation import (
    get_pivot_segmentation,
)
from AlgoSteps.step4_upper_cross_detection import (
    get_upper_cross_detection,
)
from AlgoSteps.step7_xpander_curvature import (
    get_xpander_curvature,
)
from AlgoSteps.step6_xpander_segmentation_v4 import (
    get_xpander_segmentation,
)
from analysis_summary_plot_partial import (
    plot_analysis_summary,
)
from models import Point2D, RawAnalysisResult


def analyze_height_map(
    height_map: np.ndarray,
    pixel_size_um: float,
    input_file_name: str | None = None,
    print_debug: bool = False,
    show_debug: bool = False,
    show_corner_debug: bool = False,
    summary_plot: bool = False,
    summary_output_path: str | Path | None = None,
    show_summary_on_failure: bool = True,
    raise_on_failure: bool = False,
) -> RawAnalysisResult:
    """
    Run the seven analysis stages sequentially.

    If a stage fails:
    - later stages are not executed;
    - all successful earlier results are retained;
    - a partial summary plot can still be displayed or saved;
    - a partial RawAnalysisResult is returned;
    - the original exception is re-raised only when raise_on_failure=True.
    """
    del pixel_size_um  # Stage 7 currently uses the assignment's 0.252 μm pitch.

    lower_plane_detection: Any | None = None
    lower_cross_detection: Any | None = None
    pivot_segmentation: Any | None = None
    upper_cross_detection: Any | None = None
    pivot_height_result: Any | None = None
    xpander_segmentation: Any | None = None
    xpander_curvature: Any | None = None

    completed_stages: list[str] = []
    failed_stage: str | None = None
    failure_message: str | None = None
    caught_error: Exception | None = None

    # Seven-stage analysis summary:
    # 1. Detect the lower Pivot plane.
    # 2. Detect the lower cross inside the selected plane.
    # 3. Label the complete Pivot.
    # 4. Detect the upper Pivot cross.
    # 5. Measure the height difference between the two Pivot surfaces.
    # 6. Detect and label the Xpander.
    # 7. Measure the Xpander radius of curvature.

    try:
        # Stage 1: detect the lower Pivot plane.
        current_stage = "Stage 1 — lower Pivot plane"
        lower_plane_detection = get_lower_plane_detection(
            height_map=height_map,
            print_debug=print_debug,
            show_debug=show_debug,
        )
        completed_stages.append(current_stage)

        # Stage 2: detect the lower cross inside the selected plane.
        current_stage = "Stage 2 — lower Pivot cross"
        lower_cross_detection = get_lower_cross_detection(
            height_map=height_map,
            lower_plane_detection=lower_plane_detection,
            print_debug=print_debug,
            show_debug=show_debug,
        )
        completed_stages.append(current_stage)

        # Stage 3: label the complete Pivot.
        current_stage = "Stage 3 — complete Pivot segmentation"
        pivot_segmentation = get_pivot_segmentation(
            height_map=height_map,
            lower_plane_detection=lower_plane_detection,
            lower_cross_detection=lower_cross_detection,
            print_debug=print_debug,
            show_debug=show_debug,
        )
        completed_stages.append(current_stage)

        # Stage 4: detect the upper Pivot cross.
        current_stage = "Stage 4 — upper Pivot cross"
        upper_cross_detection = get_upper_cross_detection(
            height_map=height_map,
            pivot_segmentation=pivot_segmentation,
            lower_plane_detection=lower_plane_detection,
            lower_cross_detection=lower_cross_detection,
            print_debug=print_debug,
            show_debug=show_debug,
        )

        if upper_cross_detection.best_candidate is None:
            raise ValueError("No upper Pivot cross candidate was selected.")

        completed_stages.append(current_stage)

        # Stage 5: measure the height difference between the two Pivot surfaces.
        current_stage = "Stage 5 — Pivot plane-height difference"
        pivot_height_result = get_pivot_plane_height_difference(
            height_map=height_map,
            lower_plane_detection=lower_plane_detection,
            lower_cross_detection=lower_cross_detection,
            pivot_segmentation=pivot_segmentation,
            upper_cross_detection=upper_cross_detection,
            print_debug=print_debug,
            show_debug=show_debug,
        )
        completed_stages.append(current_stage)

        # Stage 6: detect and label the Xpander.
        current_stage = "Stage 6 — Xpander segmentation"
        xpander_segmentation = get_xpander_segmentation(
            height_map=height_map,
            pivot_segmentation=pivot_segmentation,
            print_debug=print_debug,
            show_debug=show_debug,
            show_corner_debug=show_corner_debug,
        )
        completed_stages.append(current_stage)

        # Stage 7: measure the Xpander radius of curvature.
        current_stage = "Stage 7 — Xpander curvature"
        xpander_curvature = get_xpander_curvature(
            height_map=height_map,
            xpander_segmentation=xpander_segmentation,
            print_debug=print_debug,
            show_debug=show_debug,
        )
        completed_stages.append(current_stage)

    except Exception as error:
        failed_stage = current_stage
        failure_message = str(error)
        caught_error = error

        if print_debug:
            print(
                f"Analysis stopped at {failed_stage}: "
                f"{failure_message}"
            )

    cross_centers = _collect_cross_centers(
        lower_cross_detection=lower_cross_detection,
        upper_cross_detection=upper_cross_detection,
    )

    final_label_map = _build_partial_label_map(
        image_shape=height_map.shape,
        pivot_segmentation=pivot_segmentation,
        xpander_segmentation=xpander_segmentation,
    )

    should_create_summary = (
        summary_plot
        or summary_output_path is not None
        or failed_stage is not None
    )

    if should_create_summary:
        plot_analysis_summary(
            height_map=height_map,
            lower_plane_detection=lower_plane_detection,
            lower_cross_detection=lower_cross_detection,
            pivot_segmentation=pivot_segmentation,
            upper_cross_detection=upper_cross_detection,
            pivot_height_result=pivot_height_result,
            xpander_segmentation=xpander_segmentation,
            xpander_curvature=xpander_curvature,
            completed_stages=completed_stages,
            failed_stage=failed_stage,
            failure_message=failure_message,
            file_name=input_file_name,
            output_path=summary_output_path,
            show=(
                summary_plot
                or (
                    show_summary_on_failure
                    and failed_stage is not None
                )
            ),
        )

    result = RawAnalysisResult(
        status=(
            "completed"
            if failed_stage is None
            else f"failed: {failed_stage}"
        ),
        pivot_height_difference_um=_optional_float(
            pivot_height_result,
            "height_difference",
        ),
        pivot_cross_centers_px=cross_centers,
        xpander_radius_x_um=_optional_float(
            xpander_curvature,
            "radius_x_um",
        ),
        xpander_radius_y_um=_optional_float(
            xpander_curvature,
            "radius_y_um",
        ),
        radius_fit_score_x=_nested_optional_float(
            xpander_curvature,
            "x_axis",
            "confidence",
        ),
        radius_fit_score_y=_nested_optional_float(
            xpander_curvature,
            "y_axis",
            "confidence",
        ),
        radius_fit_score_overall=_optional_float(
            xpander_curvature,
            "confidence",
        ),
        label_map=final_label_map,
    )

    if caught_error is not None and raise_on_failure:
        raise caught_error

    return result


def _collect_cross_centers(
    *,
    lower_cross_detection: Any | None,
    upper_cross_detection: Any | None,
) -> list[Point2D]:
    centers: list[Point2D] = []

    for detection in (
        lower_cross_detection,
        upper_cross_detection,
    ):
        if detection is None:
            continue

        candidate = getattr(detection, "best_candidate", None)
        if candidate is None:
            continue

        centers.append(
            Point2D(
                x=float(candidate.center_x),
                y=float(candidate.center_y),
            )
        )

    return centers


def _build_partial_label_map(
    *,
    image_shape: tuple[int, int],
    pivot_segmentation: Any | None,
    xpander_segmentation: Any | None,
) -> np.ndarray:
    """
    Build the best label map available at the point where processing stopped.

    Labels:
    0 = background
    1 = Pivot
    2 = Xpander
    """
    label_map = np.zeros(image_shape, dtype=np.uint8)

    if xpander_segmentation is not None:
        xpander_mask = getattr(
            xpander_segmentation,
            "xpander_mask",
            None,
        )
        if (
            xpander_mask is not None
            and xpander_mask.shape == image_shape
        ):
            label_map[xpander_mask] = 2

    if pivot_segmentation is not None:
        pivot_mask = getattr(
            pivot_segmentation,
            "pivot_mask",
            None,
        )
        if (
            pivot_mask is not None
            and pivot_mask.shape == image_shape
        ):
            # Pivot wins in the unlikely case of overlap.
            label_map[pivot_mask] = 1

    return label_map


def _optional_float(
    result: Any | None,
    attribute_name: str,
) -> float:
    """
    Return NaN when a sequential stage did not produce the requested value.

    This keeps RawAnalysisResult compatible with float fields without
    fabricating a measurement.
    """
    if result is None:
        return float("nan")

    value = getattr(result, attribute_name, None)
    if value is None:
        return float("nan")

    return float(value)


def _nested_optional_float(
    result: Any | None,
    nested_attribute_name: str,
    value_attribute_name: str,
) -> float:
    if result is None:
        return float("nan")

    nested_result = getattr(
        result,
        nested_attribute_name,
        None,
    )
    return _optional_float(
        nested_result,
        value_attribute_name,
    )
