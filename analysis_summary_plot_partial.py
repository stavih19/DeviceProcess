from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from numpy.typing import NDArray


FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class AnalysisSummaryPlotConfig:
    """Configuration for the final or partial analysis-summary plot."""

    figure_size: tuple[float, float] = (16.0, 9.5)
    colormap: str = "viridis"

    bounding_box_line_width: float = 1.0
    point_edge_line_width: float = 0.7
    text_box_line_width: float = 0.8

    point_size: float = 30.0
    cross_center_size: float = 42.0

    annotation_font_size: float = 8.5
    side_text_font_size: float = 9.0
    title_font_size: float = 13.0

    lower_plane_color: str = "white"
    pivot_color: str = "red"
    xpander_color: str = "cyan"
    lower_cross_color: str = "yellow"
    upper_cross_color: str = "lime"
    measurement_color: str = "white"
    failure_color: str = "orangered"

    right_panel_x: float = 0.735
    plot_right_margin: float = 0.70
    save_dpi: int = 180


def plot_analysis_summary(
    *,
    height_map: FloatArray,
    lower_plane_detection: Any | None = None,
    lower_cross_detection: Any | None = None,
    pivot_segmentation: Any | None = None,
    upper_cross_detection: Any | None = None,
    pivot_height_result: Any | None = None,
    xpander_segmentation: Any | None = None,
    xpander_curvature: Any | None = None,
    completed_stages: Iterable[str] = (),
    failed_stage: str | None = None,
    failure_message: str | None = None,
    file_name: str | None = None,
    output_path: str | Path | None = None,
    show: bool = True,
    config: AnalysisSummaryPlotConfig | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot every result that is available.

    All stage-result arguments are optional. If a sequential stage fails, pass
    the completed results together with ``failed_stage`` and
    ``failure_message``. The plot will show the input and all detections that
    succeeded before the failure.
    """
    config = config or AnalysisSummaryPlotConfig()
    height_map = _validate_height_map(height_map)

    lower_plane_candidate = _best_candidate(lower_plane_detection)
    lower_cross = _best_candidate(lower_cross_detection)
    upper_cross = _best_candidate(upper_cross_detection)

    figure, axis = plt.subplots(figsize=config.figure_size)
    figure.subplots_adjust(right=config.plot_right_margin)

    image = axis.imshow(
        height_map,
        cmap=config.colormap,
        origin="upper",
        interpolation="nearest",
    )
    colorbar = figure.colorbar(
        image,
        ax=axis,
        fraction=0.035,
        pad=0.02,
    )
    colorbar.set_label("Height (μm)")

    legend_items: list[tuple[str, str]] = []

    # Show the lower plane only while the complete Pivot is unavailable.
    if lower_plane_candidate is not None and pivot_segmentation is None:
        lower_plane_box = _candidate_bounding_box(lower_plane_candidate)
        if lower_plane_box is not None:
            _draw_bounding_box(
                axis=axis,
                box=lower_plane_box,
                color=config.lower_plane_color,
                label="Lower plane",
                line_width=config.bounding_box_line_width,
                text_font_size=config.annotation_font_size,
                line_style="--",
            )
            centroid_x = getattr(lower_plane_candidate, "centroid_x", None)
            centroid_y = getattr(lower_plane_candidate, "centroid_y", None)
            if centroid_x is not None and centroid_y is not None:
                _draw_points(
                    axis=axis,
                    points=((float(centroid_x), float(centroid_y)),),
                    names=("LP",),
                    color=config.lower_plane_color,
                    marker="x",
                    size=config.cross_center_size,
                    edge_line_width=config.point_edge_line_width,
                    font_size=config.annotation_font_size,
                )
            legend_items.append(("Lower plane", config.lower_plane_color))

    pivot_box = None
    if pivot_segmentation is not None:
        pivot_box = getattr(pivot_segmentation, "bounding_box", None)
        if pivot_box is not None:
            _draw_bounding_box(
                axis=axis,
                box=pivot_box,
                color=config.pivot_color,
                label="Pivot BB",
                line_width=config.bounding_box_line_width,
                text_font_size=config.annotation_font_size,
            )
            _draw_points(
                axis=axis,
                points=_bounding_box_points(pivot_box),
                names=("P1", "P2", "P3", "P4"),
                color=config.pivot_color,
                marker="o",
                size=config.point_size,
                edge_line_width=config.point_edge_line_width,
                font_size=config.annotation_font_size,
            )
            legend_items.append(("Pivot", config.pivot_color))

    xpander_box = None
    if xpander_segmentation is not None:
        xpander_box = getattr(xpander_segmentation, "bounding_box", None)
        if xpander_box is not None:
            _draw_bounding_box(
                axis=axis,
                box=xpander_box,
                color=config.xpander_color,
                label="Xpander BB",
                line_width=config.bounding_box_line_width,
                text_font_size=config.annotation_font_size,
            )

        xpander_points = _xpander_corner_points(xpander_segmentation)
        if xpander_points is not None:
            _draw_points(
                axis=axis,
                points=xpander_points,
                names=("X1", "X2", "X3", "X4"),
                color=config.xpander_color,
                marker="s",
                size=config.point_size,
                edge_line_width=config.point_edge_line_width,
                font_size=config.annotation_font_size,
            )
        legend_items.append(("Xpander", config.xpander_color))

    if lower_cross is not None:
        lower_cross_box = _candidate_bounding_box(lower_cross)
        if lower_cross_box is not None:
            _draw_bounding_box(
                axis=axis,
                box=lower_cross_box,
                color=config.lower_cross_color,
                label="Lower cross",
                line_width=config.bounding_box_line_width,
                text_font_size=config.annotation_font_size,
            )

        center = _candidate_center(lower_cross)
        if center is not None:
            _draw_points(
                axis=axis,
                points=(center,),
                names=("LC",),
                color=config.lower_cross_color,
                marker="+",
                size=config.cross_center_size,
                edge_line_width=config.point_edge_line_width,
                font_size=config.annotation_font_size,
            )
        legend_items.append(("Lower cross", config.lower_cross_color))

    if upper_cross is not None:
        upper_cross_box = _candidate_bounding_box(upper_cross)
        if upper_cross_box is not None:
            _draw_bounding_box(
                axis=axis,
                box=upper_cross_box,
                color=config.upper_cross_color,
                label="Upper cross",
                line_width=config.bounding_box_line_width,
                text_font_size=config.annotation_font_size,
            )

        center = _candidate_center(upper_cross)
        if center is not None:
            _draw_points(
                axis=axis,
                points=(center,),
                names=("UC",),
                color=config.upper_cross_color,
                marker="+",
                size=config.cross_center_size,
                edge_line_width=config.point_edge_line_width,
                font_size=config.annotation_font_size,
            )
        legend_items.append(("Upper cross", config.upper_cross_color))

    if pivot_height_result is not None and pivot_box is not None:
        height_difference = getattr(
            pivot_height_result,
            "height_difference",
            None,
        )
        if height_difference is not None:
            pivot_center_x, _ = _box_center(pivot_box)
            axis.text(
                pivot_center_x,
                pivot_box.y_max + 8,
                f"ΔH = {float(height_difference):.4f} μm",
                color=config.pivot_color,
                fontsize=config.annotation_font_size + 1,
                ha="center",
                va="top",
                bbox={
                    "facecolor": "black",
                    "alpha": 0.55,
                    "edgecolor": config.pivot_color,
                    "linewidth": config.text_box_line_width,
                    "pad": 3,
                },
            )

    if xpander_curvature is not None and xpander_box is not None:
        radius_x = getattr(xpander_curvature, "radius_x_um", None)
        radius_y = getattr(xpander_curvature, "radius_y_um", None)
        if radius_x is not None and radius_y is not None:
            xpander_center_x, xpander_center_y = _box_center(xpander_box)
            axis.text(
                xpander_center_x,
                xpander_center_y,
                (
                    f"Rₓ = {float(radius_x):.3f} μm\n"
                    f"Rᵧ = {float(radius_y):.3f} μm"
                ),
                color=config.measurement_color,
                fontsize=config.annotation_font_size + 1,
                ha="center",
                va="center",
                bbox={
                    "facecolor": "black",
                    "alpha": 0.60,
                    "edgecolor": config.xpander_color,
                    "linewidth": config.text_box_line_width,
                    "pad": 4,
                },
            )

    image_height, image_width = height_map.shape
    axis.set_xlim(-0.5, image_width - 0.5)
    axis.set_ylim(image_height - 0.5, -0.5)
    axis.set_aspect("equal")

    is_partial = failed_stage is not None
    title = (
        "Partial analysis results"
        if is_partial
        else "Final component detection and measurements"
    )
    if file_name:
        title += f" — {file_name}"
    axis.set_title(title, fontsize=config.title_font_size)
    axis.set_xlabel("X coordinate (pixel)")
    axis.set_ylabel("Y coordinate (pixel)")

    side_text = _build_side_text(
        lower_plane_detection=lower_plane_detection,
        lower_cross_detection=lower_cross_detection,
        pivot_segmentation=pivot_segmentation,
        upper_cross_detection=upper_cross_detection,
        pivot_height_result=pivot_height_result,
        xpander_segmentation=xpander_segmentation,
        xpander_curvature=xpander_curvature,
        completed_stages=tuple(completed_stages),
        failed_stage=failed_stage,
        failure_message=failure_message,
    )
    figure.text(
        config.right_panel_x,
        0.94,
        side_text,
        va="top",
        ha="left",
        fontsize=config.side_text_font_size,
        family="monospace",
        linespacing=1.20,
    )

    if legend_items:
        used: set[str] = set()
        for label, color in legend_items:
            if label in used:
                continue
            used.add(label)
            axis.plot(
                [],
                [],
                color=color,
                linewidth=1.5,
                label=label,
            )
        axis.legend(loc="lower left", framealpha=0.8)

    if failed_stage is not None:
        axis.text(
            0.015,
            0.02,
            f"FAILED: {failed_stage}",
            transform=axis.transAxes,
            color="white",
            fontsize=config.annotation_font_size + 1,
            ha="left",
            va="bottom",
            bbox={
                "facecolor": config.failure_color,
                "alpha": 0.85,
                "edgecolor": "black",
                "linewidth": config.text_box_line_width,
                "pad": 4,
            },
        )

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            output,
            dpi=config.save_dpi,
            bbox_inches="tight",
        )

    if show:
        plt.show()
    else:
        plt.close(figure)

    return figure, axis


def _build_side_text(
    *,
    lower_plane_detection: Any | None,
    lower_cross_detection: Any | None,
    pivot_segmentation: Any | None,
    upper_cross_detection: Any | None,
    pivot_height_result: Any | None,
    xpander_segmentation: Any | None,
    xpander_curvature: Any | None,
    completed_stages: tuple[str, ...],
    failed_stage: str | None,
    failure_message: str | None,
) -> str:
    lines = ["ANALYSIS STATUS", "=" * 38]

    if failed_stage is None:
        lines.append("status: completed")
    else:
        lines.append("status: failed")
        lines.append(f"failed stage: {failed_stage}")
        if failure_message:
            lines.extend(_wrapped_lines("error: ", failure_message, 52))

    if completed_stages:
        lines.append("")
        lines.append("COMPLETED STAGES")
        lines.extend(f"✓ {stage}" for stage in completed_stages)

    lower_plane = _best_candidate(lower_plane_detection)
    lower_cross = _best_candidate(lower_cross_detection)
    upper_cross = _best_candidate(upper_cross_detection)

    lines.extend(["", "LOWER PIVOT PLANE"])
    if lower_plane is None:
        lines.append("not available")
    else:
        box = _candidate_bounding_box(lower_plane)
        if box is not None:
            lines.append(f"BB: {_format_box(box)}")
        centroid = _candidate_centroid(lower_plane)
        if centroid is not None:
            lines.append(
                f"center: ({centroid[0]:.2f}, {centroid[1]:.2f}) px"
            )
        score = getattr(lower_plane, "score", None)
        if score is not None:
            lines.append(f"score: {float(score):.4f}")

    lines.extend(["", "PIVOT"])
    if pivot_segmentation is None:
        lines.append("not available")
    else:
        box = getattr(pivot_segmentation, "bounding_box", None)
        if box is not None:
            lines.append(f"BB: {_format_box(box)}")
            lines.extend(
                _format_named_points(
                    ("P1", "P2", "P3", "P4"),
                    _bounding_box_points(box),
                )
            )
        confidence = getattr(pivot_segmentation, "confidence", None)
        if confidence is not None:
            lines.append(f"confidence: {float(confidence):.4f}")

    lines.extend(["", "LOWER CROSS"])
    lines.extend(_cross_side_lines(lower_cross, prefix="L"))

    lines.extend(["", "UPPER CROSS"])
    lines.extend(_cross_side_lines(upper_cross, prefix="U"))

    lines.extend(["", "PIVOT HEIGHT"])
    if pivot_height_result is None:
        lines.append("not available")
    else:
        plane_1 = getattr(pivot_height_result, "plane_1_height", None)
        plane_2 = getattr(pivot_height_result, "plane_2_height", None)
        difference = getattr(pivot_height_result, "height_difference", None)
        if plane_1 is not None:
            lines.append(f"plane 1: {float(plane_1):.6f} μm")
        if plane_2 is not None:
            lines.append(f"plane 2: {float(plane_2):.6f} μm")
        if difference is not None:
            lines.append(f"ΔH (P2-P1): {float(difference):.6f} μm")

    lines.extend(["", "XPANDER"])
    if xpander_segmentation is None:
        lines.append("not available")
    else:
        box = getattr(xpander_segmentation, "bounding_box", None)
        if box is not None:
            lines.append(f"BB: {_format_box(box)}")
        points = _xpander_corner_points(xpander_segmentation)
        if points is not None:
            lines.extend(
                _format_named_points(
                    ("X1", "X2", "X3", "X4"),
                    points,
                )
            )
        confidence = getattr(xpander_segmentation, "confidence", None)
        if confidence is not None:
            lines.append(f"confidence: {float(confidence):.4f}")

    lines.extend(["", "XPANDER CURVATURE"])
    if xpander_curvature is None:
        lines.append("not available")
    else:
        radius_x = getattr(xpander_curvature, "radius_x_um", None)
        radius_y = getattr(xpander_curvature, "radius_y_um", None)
        if radius_x is not None:
            lines.append(f"R_x: {float(radius_x):.6f} μm")
        if radius_y is not None:
            lines.append(f"R_y: {float(radius_y):.6f} μm")

        x_axis = getattr(xpander_curvature, "x_axis", None)
        y_axis = getattr(xpander_curvature, "y_axis", None)
        if x_axis is not None:
            confidence = getattr(x_axis, "confidence", None)
            if confidence is not None:
                lines.append(f"fit X: {float(confidence):.4f}")
        if y_axis is not None:
            confidence = getattr(y_axis, "confidence", None)
            if confidence is not None:
                lines.append(f"fit Y: {float(confidence):.4f}")

        confidence = getattr(xpander_curvature, "confidence", None)
        if confidence is not None:
            lines.append(f"overall: {float(confidence):.4f}")

    return "\n".join(lines)


def _cross_side_lines(
    candidate: Any | None,
    *,
    prefix: str,
) -> list[str]:
    if candidate is None:
        return ["not available"]

    lines: list[str] = []
    box = _candidate_bounding_box(candidate)
    if box is not None:
        lines.append(f"BB: {_format_box(box)}")
        lines.extend(
            _format_named_points(
                (
                    f"{prefix}1",
                    f"{prefix}2",
                    f"{prefix}3",
                    f"{prefix}4",
                ),
                _bounding_box_points(box),
            )
        )

    center = _candidate_center(candidate)
    if center is not None:
        lines.append(f"center: ({center[0]:.2f}, {center[1]:.2f}) px")

    score = getattr(candidate, "score", None)
    if score is not None:
        lines.append(f"score: {float(score):.4f}")

    return lines


def _validate_height_map(height_map: FloatArray) -> FloatArray:
    array = np.asarray(height_map)
    if array.ndim != 2:
        raise ValueError(
            f"Expected a 2-D height map, received shape {array.shape}."
        )
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(
            f"Expected a numeric height map, received {array.dtype}."
        )
    if not np.isfinite(array).all():
        raise ValueError("Height map contains NaN or infinite values.")
    return array


def _best_candidate(result: Any | None) -> Any | None:
    if result is None:
        return None
    return getattr(result, "best_candidate", None)


def _candidate_bounding_box(candidate: Any) -> Any | None:
    return (
        getattr(candidate, "bounding_box", None)
        or getattr(candidate, "outer_bounding_box", None)
    )


def _candidate_center(candidate: Any) -> tuple[float, float] | None:
    x_value = getattr(candidate, "center_x", None)
    y_value = getattr(candidate, "center_y", None)
    if x_value is None or y_value is None:
        return None
    return float(x_value), float(y_value)


def _candidate_centroid(candidate: Any) -> tuple[float, float] | None:
    x_value = getattr(candidate, "centroid_x", None)
    y_value = getattr(candidate, "centroid_y", None)
    if x_value is None or y_value is None:
        return None
    return float(x_value), float(y_value)


def _xpander_corner_points(
    segmentation: Any,
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
] | None:
    names = ("top_left", "top_right", "bottom_right", "bottom_left")
    point_objects = [getattr(segmentation, name, None) for name in names]
    if any(point is None for point in point_objects):
        return None

    return tuple(
        (float(point.x), float(point.y))
        for point in point_objects
    )  # type: ignore[return-value]


def _draw_bounding_box(
    *,
    axis: Axes,
    box: Any,
    color: str,
    label: str,
    line_width: float,
    text_font_size: float,
    line_style: str = "-",
) -> None:
    axis.add_patch(
        Rectangle(
            (box.x_min, box.y_min),
            int(box.x_max - box.x_min),
            int(box.y_max - box.y_min),
            fill=False,
            edgecolor=color,
            linewidth=line_width,
            linestyle=line_style,
            zorder=6,
        )
    )
    axis.text(
        box.x_min,
        max(0, box.y_min - 4),
        label,
        color=color,
        fontsize=text_font_size,
        ha="left",
        va="bottom",
        bbox={
            "facecolor": "black",
            "alpha": 0.40,
            "edgecolor": "none",
            "pad": 1.5,
        },
        zorder=7,
    )


def _draw_points(
    *,
    axis: Axes,
    points: tuple[tuple[float, float], ...],
    names: tuple[str, ...],
    color: str,
    marker: str,
    size: float,
    edge_line_width: float,
    font_size: float,
) -> None:
    for (x_value, y_value), name in zip(points, names, strict=True):
        if marker in {"+", "x"}:
            axis.scatter(
                [x_value],
                [y_value],
                s=size,
                marker=marker,
                color=color,
                linewidths=edge_line_width,
                zorder=8,
            )
        else:
            axis.scatter(
                [x_value],
                [y_value],
                s=size,
                marker=marker,
                facecolors="none",
                edgecolors=color,
                linewidths=edge_line_width,
                zorder=8,
            )

        axis.annotate(
            name,
            xy=(x_value, y_value),
            xytext=(4, -5),
            textcoords="offset points",
            color=color,
            fontsize=font_size,
            fontweight="bold",
            zorder=9,
        )


def _bounding_box_points(
    box: Any,
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]:
    return (
        (float(box.x_min), float(box.y_min)),
        (float(box.x_max - 1), float(box.y_min)),
        (float(box.x_max - 1), float(box.y_max - 1)),
        (float(box.x_min), float(box.y_max - 1)),
    )


def _box_center(box: Any) -> tuple[float, float]:
    return (
        0.5 * (box.x_min + box.x_max - 1),
        0.5 * (box.y_min + box.y_max - 1),
    )


def _format_box(box: Any) -> str:
    return (
        f"x={box.x_min}:{box.x_max}, "
        f"y={box.y_min}:{box.y_max}"
    )


def _format_named_points(
    names: tuple[str, ...],
    points: tuple[tuple[float, float], ...],
) -> list[str]:
    return [
        f"{name}: ({x_value:.1f}, {y_value:.1f}) px"
        for name, (x_value, y_value) in zip(names, points, strict=True)
    ]


def _wrapped_lines(prefix: str, text: str, width: int) -> list[str]:
    words = text.split()
    if not words:
        return [prefix.rstrip()]

    lines: list[str] = []
    current = prefix
    continuation_prefix = " " * len(prefix)

    for word in words:
        separator = "" if current.endswith(" ") else " "
        candidate = current + separator + word
        if len(candidate) <= width:
            current = candidate
            continue

        lines.append(current.rstrip())
        current = continuation_prefix + word

    lines.append(current.rstrip())
    return lines
