from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


FloatArray = NDArray[np.floating]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.integer]


@dataclass(frozen=True)
class BoundingBox:
    """
    Axis-aligned bounding box.

    x_max and y_max are exclusive, following NumPy slicing conventions.
    """

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    @property
    def corners(self) -> tuple[
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
    ]:
        """
        Return corners in:
        top-left, top-right, bottom-right, bottom-left order.
        """
        return (
            (self.x_min, self.y_min),
            (self.x_max - 1, self.y_min),
            (self.x_max - 1, self.y_max - 1),
            (self.x_min, self.y_max - 1),
        )


@dataclass(frozen=True)
class LowerPlaneCandidate:
    """
    Measurements describing one possible lower Pivot plane.
    """

    component_label: int
    bounding_box: BoundingBox

    centroid_x: float
    centroid_y: float

    area_pixels: int
    rectangularity: float
    width_height_ratio: float

    median_height: float
    height_mad: float

    touches_border: bool
    score: float


@dataclass
class LowerPlaneDetectionResult:
    """
    Output of the lower-plane candidate detection stage.

    label_image contains connected-component IDs.
    A candidate's exact mask is obtained by comparing label_image with
    candidate.component_label.
    """

    threshold: float
    high_region_mask: BoolArray
    label_image: IntArray
    candidates: list[LowerPlaneCandidate] = field(default_factory=list)

    @property
    def best_candidate(self) -> LowerPlaneCandidate | None:
        if not self.candidates:
            return None

        return self.candidates[0]

    def get_candidate_mask(
        self,
        candidate: LowerPlaneCandidate,
    ) -> BoolArray:
        return self.label_image == candidate.component_label


@dataclass(frozen=True)
class LowerPlaneDetectionConfig:
    """
    Initial configurable parameters.

    These values are starting points and should be validated against all
    supplied datasets.
    """

    gaussian_sigma: float = 1.0

    # Keep approximately the highest 3% of smoothed height values.
    height_quantile: float = 97.0

    opening_size: int = 3
    closing_size: int = 5

    min_area_fraction: float = 0.001
    max_area_fraction: float = 0.03

    min_rectangularity: float = 0.65

    # The lower Pivot plane is expected to be wider than it is tall.
    min_width_height_ratio: float = 1.15
    max_width_height_ratio: float = 3.0

    reject_border_components: bool = True


def find_lower_plane_candidates(
    height_map: FloatArray,
    config: LowerPlaneDetectionConfig | None = None,
) -> LowerPlaneDetectionResult:
    """
    Find candidate regions for the lower flat plane of the Pivot.

    This function does not yet label the complete Pivot. It detects and ranks
    high, approximately rectangular, horizontally oriented flat regions.

    Args:
        height_map:
            Two-dimensional array whose values represent measured height.

        config:
            Detection parameters. Defaults are used when omitted.

    Returns:
        LowerPlaneDetectionResult containing the connected-component label
        image and the ranked candidate list.
    """
    if config is None:
        config = LowerPlaneDetectionConfig()

    _validate_height_map(height_map)
    _validate_config(config)

    height_map = height_map.astype(np.float32, copy=False)

    # Step 1: suppress small local measurement noise.
    smoothed = ndi.gaussian_filter(
        height_map,
        sigma=config.gaussian_sigma,
    )

    # Step 2: use a relative threshold instead of a fixed physical height.
    threshold = float(
        np.percentile(smoothed, config.height_quantile)
    )

    high_region_mask = smoothed >= threshold

    # Step 3: remove isolated pixels and very small protrusions.
    opening_structure = np.ones(
        (config.opening_size, config.opening_size),
        dtype=bool,
    )
    high_region_mask = ndi.binary_opening(
        high_region_mask,
        structure=opening_structure,
    )

    # Step 4: connect small gaps in the high region.
    closing_structure = np.ones(
        (config.closing_size, config.closing_size),
        dtype=bool,
    )
    high_region_mask = ndi.binary_closing(
        high_region_mask,
        structure=closing_structure,
    )

    # The cross may appear as a hole inside the lower plane.
    # At this stage, we want the complete candidate region.
    high_region_mask = ndi.binary_fill_holes(high_region_mask)

    # Step 5: label spatially disconnected regions.
    label_image, component_count = ndi.label(high_region_mask)

    image_height, image_width = height_map.shape
    image_area = image_height * image_width

    min_area = max(
        1,
        int(image_area * config.min_area_fraction),
    )
    max_area = int(image_area * config.max_area_fraction)

    # Used to normalize the flatness measurement.
    robust_height_range = float(
        np.percentile(height_map, 99)
        - np.percentile(height_map, 1)
    )
    robust_height_range = max(robust_height_range, 1e-8)

    component_slices = ndi.find_objects(label_image)
    candidates: list[LowerPlaneCandidate] = []

    for component_label in range(1, component_count + 1):
        component_slice = component_slices[component_label - 1]

        if component_slice is None:
            continue

        y_slice, x_slice = component_slice

        bounding_box = BoundingBox(
            x_min=x_slice.start,
            y_min=y_slice.start,
            x_max=x_slice.stop,
            y_max=y_slice.stop,
        )

        component_mask = label_image == component_label
        area_pixels = int(np.count_nonzero(component_mask))

        if area_pixels < min_area or area_pixels > max_area:
            continue

        bounding_box_area = (
            bounding_box.width * bounding_box.height
        )

        if bounding_box_area == 0:
            continue

        rectangularity = area_pixels / bounding_box_area

        width_height_ratio = (
            bounding_box.width / bounding_box.height
        )

        touches_border = (
            bounding_box.x_min == 0
            or bounding_box.y_min == 0
            or bounding_box.x_max == image_width
            or bounding_box.y_max == image_height
        )

        if (
            config.reject_border_components
            and touches_border
        ):
            continue

        if rectangularity < config.min_rectangularity:
            continue

        if not (
            config.min_width_height_ratio
            <= width_height_ratio
            <= config.max_width_height_ratio
        ):
            continue

        y_coordinates, x_coordinates = np.nonzero(
            component_mask
        )

        centroid_x = float(np.mean(x_coordinates))
        centroid_y = float(np.mean(y_coordinates))

        component_heights = height_map[component_mask]

        median_height = float(
            np.median(component_heights)
        )

        # Median Absolute Deviation is more robust than standard deviation.
        height_mad = float(
            np.median(
                np.abs(component_heights - median_height)
            )
        )

        score = _calculate_candidate_score(
            rectangularity=rectangularity,
            width_height_ratio=width_height_ratio,
            height_mad=height_mad,
            robust_height_range=robust_height_range,
            config=config,
        )

        candidates.append(
            LowerPlaneCandidate(
                component_label=component_label,
                bounding_box=bounding_box,
                centroid_x=centroid_x,
                centroid_y=centroid_y,
                area_pixels=area_pixels,
                rectangularity=float(rectangularity),
                width_height_ratio=float(width_height_ratio),
                median_height=median_height,
                height_mad=height_mad,
                touches_border=touches_border,
                score=score,
            )
        )

    # The first element is always the strongest candidate.
    candidates.sort(
        key=lambda candidate: candidate.score,
        reverse=True,
    )

    return LowerPlaneDetectionResult(
        threshold=threshold,
        high_region_mask=high_region_mask,
        label_image=label_image,
        candidates=candidates,
    )


def _calculate_candidate_score(
    rectangularity: float,
    width_height_ratio: float,
    height_mad: float,
    robust_height_range: float,
    config: LowerPlaneDetectionConfig,
) -> float:
    """
    Produce a score in approximately the [0, 1] range.

    The score is not a physical measurement. It is used only to rank
    candidate regions.
    """
    rectangularity_score = np.clip(
        (
            rectangularity
            - config.min_rectangularity
        )
        / (
            1.0
            - config.min_rectangularity
        ),
        0.0,
        1.0,
    )

    # A wider horizontal region receives a better score, up to ratio 2.
    aspect_score = np.clip(
        (
            width_height_ratio
            - config.min_width_height_ratio
        )
        / (
            2.0
            - config.min_width_height_ratio
        ),
        0.0,
        1.0,
    )

    normalized_mad = height_mad / robust_height_range

    # Low variation means the region is relatively flat.
    flatness_score = float(
        np.exp(-normalized_mad / 0.1)
    )

    score = (
        0.45 * rectangularity_score
        + 0.25 * aspect_score
        + 0.30 * flatness_score
    )

    return float(score)


def _validate_height_map(height_map: np.ndarray) -> None:
    if height_map.ndim != 2:
        raise ValueError(
            "Expected a two-dimensional height map, "
            f"received shape {height_map.shape}."
        )

    if not np.issubdtype(height_map.dtype, np.number):
        raise TypeError(
            f"Expected numeric data, received {height_map.dtype}."
        )

    if not np.isfinite(height_map).all():
        raise ValueError(
            "Height map contains NaN or infinite values."
        )


def _validate_config(
    config: LowerPlaneDetectionConfig,
) -> None:
    if not 0 < config.height_quantile < 100:
        raise ValueError(
            "height_quantile must be between 0 and 100."
        )

    if config.opening_size < 1:
        raise ValueError(
            "opening_size must be at least 1."
        )

    if config.closing_size < 1:
        raise ValueError(
            "closing_size must be at least 1."
        )

    if (
        config.min_area_fraction < 0
        or config.max_area_fraction <= 0
        or config.min_area_fraction
        >= config.max_area_fraction
    ):
        raise ValueError(
            "Invalid component-area fractions."
        )

def print_lower_plane_candidates(
    detection_result: LowerPlaneDetectionResult,
) -> None:
    """
    Print all candidates in descending score order.
    """
    if not detection_result.candidates:
        print("No lower-plane candidates were found.")
        return

    print("\nAll lower-plane candidates:")
    print("-" * 100)

    for rank, candidate in enumerate(
        detection_result.candidates,
        start=1,
    ):
        bbox = candidate.bounding_box
        selected_marker = " <-- SELECTED" if rank == 1 else ""

        print(
            f"Rank {rank}: "
            f"label={candidate.component_label}, "
            f"score={candidate.score:.4f}, "
            f"bbox=("
            f"x={bbox.x_min}:{bbox.x_max}, "
            f"y={bbox.y_min}:{bbox.y_max}"
            f"), "
            f"size={bbox.width}x{bbox.height}, "
            f"centroid=("
            f"{candidate.centroid_x:.2f}, "
            f"{candidate.centroid_y:.2f}"
            f"), "
            f"area={candidate.area_pixels}, "
            f"rectangularity={candidate.rectangularity:.4f}, "
            f"aspect_ratio={candidate.width_height_ratio:.4f}, "
            f"median_height={candidate.median_height:.4f}, "
            f"height_mad={candidate.height_mad:.4f}"
            f"{selected_marker}"
        )

    print("-" * 100)


def plot_lower_plane_candidates(
    height_map: FloatArray,
    detection_result: LowerPlaneDetectionResult,
) -> None:
    """
    Display the height map with candidate bounding boxes.

    The selected candidate is drawn with a thick red rectangle.
    Other candidates are drawn with thinner white rectangles.
    """
    if not detection_result.candidates:
        print("Cannot plot candidates because no candidates were found.")
        return

    figure, axis = plt.subplots(figsize=(10, 12))

    image = axis.imshow(
        height_map,
        cmap="viridis",
        aspect="auto",
    )

    figure.colorbar(
        image,
        ax=axis,
        label="Height",
    )

    for rank, candidate in enumerate(
        detection_result.candidates,
        start=1,
    ):
        bbox = candidate.bounding_box
        is_best_candidate = rank == 1

        rectangle = Rectangle(
            (bbox.x_min, bbox.y_min),
            bbox.width,
            bbox.height,
            fill=False,
            edgecolor="red" if is_best_candidate else "white",
            linewidth=3 if is_best_candidate else 1,
        )

        axis.add_patch(rectangle)

        axis.text(
            bbox.x_min,
            max(0, bbox.y_min - 8),
            (
                f"#{rank} "
                f"score={candidate.score:.3f}"
            ),
            fontsize=9,
            color="red" if is_best_candidate else "white",
            bbox={
                "facecolor": "black",
                "alpha": 0.65,
                "pad": 2,
            },
        )

    best_candidate = detection_result.best_candidate
    assert best_candidate is not None

    axis.scatter(
        best_candidate.centroid_x,
        best_candidate.centroid_y,
        marker="x",
        s=100,
        linewidths=3,
        color="red",
        label="Selected candidate centroid",
    )

    axis.set_title(
        "Lower Pivot Plane Candidates\n"
        "Selected candidate is marked in red"
    )
    axis.set_xlabel("X coordinate [pixels]")
    axis.set_ylabel("Y coordinate [pixels]")
    axis.legend()

    figure.tight_layout()
    plt.show()


def get_lower_plane_detection(
    height_map: FloatArray,
    print_debug: bool = False,
    show_debug: bool = False,
) -> LowerPlaneDetectionResult:
    """Detect the lower Pivot plane and optionally display its debug plot."""
    detection_result = find_lower_plane_candidates(height_map)

    if detection_result.best_candidate is None:
        raise ValueError("No lower Pivot plane candidate was found.")

    if print_debug:
        print_lower_plane_candidates(detection_result)

    if show_debug:
        plot_lower_plane_candidates(
            height_map=height_map,
            detection_result=detection_result,
        )

    return detection_result
