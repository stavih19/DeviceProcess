from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray


AnalysisStatus = Literal["not_implemented", "ok", "failed"]
ElementName = Literal["pivot", "xpander", "unknown"]


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float


@dataclass(frozen=True)
class Fault:
    element: ElementName
    reason: str


@dataclass
class RawAnalysisResult:
    """
    Internal result returned by the algorithm.

    It may contain large NumPy arrays. These arrays are saved to files by the
    output layer and are not kept in the final lightweight AnalysisResult.
    """

    status: AnalysisStatus

    # Task 1 and Task 2 measurements
    pivot_bounding_box_px: list[Point2D] = field(default_factory=list)
    xpander_bounding_box_px: list[Point2D] = field(default_factory=list)
    pivot_height_difference_um: float | None = None
    pivot_cross_centers_px: list[Point2D] = field(default_factory=list)

    xpander_radius_x_um: float | None = None
    xpander_radius_y_um: float | None = None

    radius_fit_score_x: float | None = None
    radius_fit_score_y: float | None = None
    radius_fit_score_overall: float | None = None

    # Mainly required for Task 2
    tilt_x_deg: float | None = None
    tilt_y_deg: float | None = None
    rotation_deg: float | None = None

    faults: list[Fault] = field(default_factory=list)

    # Label convention:
    # 0 = background
    # 1 = Pivot
    # 2 = Xpander
    label_map: NDArray[np.uint8] | None = None


@dataclass(frozen=True)
class OutputFileNames:
    """Only file names are stored here, not arrays or absolute paths."""

    result_json: str
    label_map_npy: str
    label_image_png: str


@dataclass(frozen=True)
class AnalysisResult:
    """
    Lightweight result for one input file.

    Numeric/list outputs are kept in the object. Generated arrays and images
    are saved on disk, and the object stores only their file names.
    """

    input_file_name: str
    status: AnalysisStatus

    pivot_bounding_box_px: list[Point2D]
    xpander_bounding_box_px: list[Point2D]

    pivot_height_difference_um: float | None
    pivot_cross_centers_px: list[Point2D]

    xpander_radius_x_um: float | None
    xpander_radius_y_um: float | None

    radius_fit_score_x: float | None
    radius_fit_score_y: float | None
    radius_fit_score_overall: float | None

    tilt_x_deg: float | None
    tilt_y_deg: float | None
    rotation_deg: float | None

    faults: list[Fault]
    output_files: OutputFileNames
