from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from AlgoSteps.lower_cross_detection import (
    LowerCrossCandidate,
    LowerCrossDetectionResult,
)
from AlgoSteps.pivot_candidates import (
    BoolArray,
    BoundingBox,
    FloatArray,
    LowerPlaneCandidate,
    LowerPlaneDetectionResult,
)
from AlgoSteps.pivot_segmentation import PivotSegmentationResult
from AlgoSteps.upper_cross_detection import (
    UpperCrossCandidate,
    UpperCrossDetectionResult,
)


@dataclass(frozen=True)
class PivotPlaneHeightConfig:
    """
    Configuration for Stage 5: measure the height difference between the two
    flat Pivot surfaces.

    For each detected cross:

    1. Use the detected cross bounding box.
    2. Extend the measurement region only along X.
    3. Extend by half of the cross width on each side by default.
    4. Keep the original cross Y range unchanged.
    5. Exclude the complete rectangular cross bounding box.
    6. Exclude an additional horizontal clearance beyond both BB edges, so
       pixels on the cross slopes or halo are not measured.
    7. Measure only the left and right strips that remain.
    8. Use every permitted pixel directly, without statistical filtering.

    The same method is applied to the upper and lower Pivot surfaces.
    """

    # Outer horizontal extension from the original cross BB, per side.
    horizontal_margin_fraction: float = 0.5
    minimum_horizontal_margin_pixels: int = 2

    # Extra exclusion beyond the left/right edges of the cross BB.
    # For example, 0.10 removes an additional 10% of the cross width per side.
    cross_clearance_fraction: float = 0.10
    minimum_cross_clearance_pixels: int = 1

    minimum_strip_width_pixels: int = 1
    minimum_pixels_per_strip: int = 4
    minimum_measurement_pixels: int = 8

    # Assignment convention used by this module:
    # plane 1 = upper surface, plane 2 = lower surface.
    plane_1_name: str = "upper"
    plane_2_name: str = "lower"


@dataclass
class LocalPlaneHeightMeasurement:
    """
    Local height measurement around one cross.

    The height is measured only in two side strips:

        outer measurement BB
        minus
        cross BB expanded horizontally by the configured clearance

    No pixels from the excluded central rectangle are used.
    """

    surface_name: str

    cross_bounding_box: BoundingBox
    exclusion_bounding_box: BoundingBox
    measurement_bounding_box: BoundingBox
    left_strip_bounding_box: BoundingBox
    right_strip_bounding_box: BoundingBox

    # Full-size masks in original image coordinates.
    cross_mask: BoolArray
    measurement_box_mask: BoolArray
    exclusion_box_mask: BoolArray
    allowed_surface_mask: BoolArray
    left_strip_mask: BoolArray
    right_strip_mask: BoolArray
    measurement_mask: BoolArray

    horizontal_margin_pixels: int
    cross_clearance_pixels: int

    pixel_count: int
    left_pixel_count: int
    right_pixel_count: int

    # The direct mean of all remaining measurement pixels is used.
    mean_height: float

    # Diagnostics only. No values are removed based on these statistics.
    median_height: float
    standard_deviation: float
    minimum_height: float
    maximum_height: float

    left_mean_height: float
    right_mean_height: float

    @property
    def left_right_height_difference(self) -> float:
        return abs(
            self.left_mean_height
            - self.right_mean_height
        )

    @property
    def side_balanced_mean_height(self) -> float:
        """
        Diagnostic value that gives equal weight to both strips.

        The final Stage-5 result still uses mean_height, which is the direct
        mean of all included pixels.
        """
        return (
            self.left_mean_height
            + self.right_mean_height
        ) / 2.0


@dataclass
class PivotPlaneHeightDifferenceResult:
    """Complete Stage-5 output."""

    upper_surface: LocalPlaneHeightMeasurement
    lower_surface: LocalPlaneHeightMeasurement

    plane_1_name: str
    plane_2_name: str
    plane_1_height: float
    plane_2_height: float

    # Required convention: plane 2 - plane 1.
    height_difference: float

    @property
    def upper_minus_lower(self) -> float:
        return (
            self.upper_surface.mean_height
            - self.lower_surface.mean_height
        )

    @property
    def lower_minus_upper(self) -> float:
        return (
            self.lower_surface.mean_height
            - self.upper_surface.mean_height
        )


def measure_pivot_plane_height_difference(
    height_map: FloatArray,
    lower_plane_detection: LowerPlaneDetectionResult,
    lower_cross_detection: LowerCrossDetectionResult,
    pivot_segmentation: PivotSegmentationResult,
    upper_cross_detection: UpperCrossDetectionResult,
    config: PivotPlaneHeightConfig | None = None,
) -> PivotPlaneHeightDifferenceResult:
    """
    Calculate the height difference between the two flat Pivot surfaces.

    For both the upper and lower cross, the measurement uses only the two
    horizontal strips beside the cross:

        left outer boundary  -> left of expanded exclusion BB
        right of expanded exclusion BB -> right outer boundary

    The original cross BB and an additional clearance beyond its left/right
    edges are excluded. The Y range is not expanded.

    No MAD filtering, clipping, trimming, gradient filtering, or other
    statistical outlier removal is performed.
    """
    if config is None:
        config = PivotPlaneHeightConfig()

    _validate_inputs(
        height_map=height_map,
        lower_plane_detection=lower_plane_detection,
        lower_cross_detection=lower_cross_detection,
        pivot_segmentation=pivot_segmentation,
        upper_cross_detection=upper_cross_detection,
        config=config,
    )

    lower_plane_candidate = (
        lower_plane_detection.best_candidate
    )
    lower_cross_candidate = (
        lower_cross_detection.best_candidate
    )
    upper_cross_candidate = (
        upper_cross_detection.best_candidate
    )

    if lower_plane_candidate is None:
        raise ValueError(
            "Stage 5 requires a selected lower-plane candidate."
        )

    if lower_cross_candidate is None:
        raise ValueError(
            "Stage 5 requires a selected lower-cross candidate."
        )

    if upper_cross_candidate is None:
        raise ValueError(
            "Stage 5 requires a selected upper-cross candidate."
        )

    image_shape = height_map.shape

    lower_cross_mask = (
        lower_cross_detection.get_candidate_mask_global(
            candidate=lower_cross_candidate,
            image_shape=image_shape,
        )
    )

    upper_cross_mask = (
        upper_cross_detection.get_candidate_mask_global(
            candidate=upper_cross_candidate,
            image_shape=image_shape,
        )
    )

    lower_surface_mask = (
        lower_plane_detection.get_candidate_mask(
            lower_plane_candidate
        )
    ).astype(bool, copy=False)

    upper_surface_mask = (
        pivot_segmentation.pivot_mask
    ).astype(bool, copy=False)

    lower_plane_box = _get_lower_plane_box(
        lower_plane_candidate
    )

    lower_measurement = _measure_surface_from_side_strips(
        height_map=height_map,
        surface_name="lower",
        cross_candidate=lower_cross_candidate,
        cross_mask=lower_cross_mask,
        allowed_surface_mask=lower_surface_mask,
        allowed_bounding_box=lower_plane_box,
        config=config,
    )

    upper_measurement = _measure_surface_from_side_strips(
        height_map=height_map,
        surface_name="upper",
        cross_candidate=upper_cross_candidate,
        cross_mask=upper_cross_mask,
        allowed_surface_mask=upper_surface_mask,
        allowed_bounding_box=pivot_segmentation.bounding_box,
        config=config,
    )

    measurement_by_name = {
        "upper": upper_measurement,
        "lower": lower_measurement,
    }

    plane_1_height = measurement_by_name[
        config.plane_1_name
    ].mean_height
    plane_2_height = measurement_by_name[
        config.plane_2_name
    ].mean_height

    return PivotPlaneHeightDifferenceResult(
        upper_surface=upper_measurement,
        lower_surface=lower_measurement,
        plane_1_name=config.plane_1_name,
        plane_2_name=config.plane_2_name,
        plane_1_height=plane_1_height,
        plane_2_height=plane_2_height,
        height_difference=(
            plane_2_height
            - plane_1_height
        ),
    )


def print_pivot_plane_height_difference(
    result: PivotPlaneHeightDifferenceResult,
) -> None:
    """Print the Stage-5 measurements."""
    print("\nPivot plane height measurement:")
    print("-" * 88)

    for measurement in (
        result.upper_surface,
        result.lower_surface,
    ):
        print(
            f"{measurement.surface_name.capitalize()} surface:"
        )
        print(
            f"  Cross BB: "
            f"{measurement.cross_bounding_box}"
        )
        print(
            f"  Exclusion BB: "
            f"{measurement.exclusion_bounding_box}"
        )
        print(
            f"  Measurement BB: "
            f"{measurement.measurement_bounding_box}"
        )
        print(
            f"  Horizontal outer margin: "
            f"{measurement.horizontal_margin_pixels} px per side"
        )
        print(
            f"  Additional cross clearance: "
            f"{measurement.cross_clearance_pixels} px per side"
        )
        print(
            f"  Left strip BB: "
            f"{measurement.left_strip_bounding_box}"
        )
        print(
            f"  Right strip BB: "
            f"{measurement.right_strip_bounding_box}"
        )
        print(
            f"  Used pixels: "
            f"{measurement.pixel_count}"
        )
        print(
            f"  Mean height: "
            f"{measurement.mean_height:.6f}"
        )
        print(
            f"  Median height (diagnostic): "
            f"{measurement.median_height:.6f}"
        )
        print(
            f"  Standard deviation (diagnostic): "
            f"{measurement.standard_deviation:.6f}"
        )
        print(
            f"  Left mean: "
            f"{measurement.left_mean_height:.6f} "
            f"({measurement.left_pixel_count} pixels)"
        )
        print(
            f"  Right mean: "
            f"{measurement.right_mean_height:.6f} "
            f"({measurement.right_pixel_count} pixels)"
        )
        print(
            f"  Left/right difference: "
            f"{measurement.left_right_height_difference:.6f}"
        )
        print(
            f"  Side-balanced mean (diagnostic): "
            f"{measurement.side_balanced_mean_height:.6f}"
        )

    print("-" * 88)
    print(
        f"Plane 1 ({result.plane_1_name}) height: "
        f"{result.plane_1_height:.6f}"
    )
    print(
        f"Plane 2 ({result.plane_2_name}) height: "
        f"{result.plane_2_height:.6f}"
    )
    print(
        f"Plane 2 - Plane 1: "
        f"{result.height_difference:.6f}"
    )
    print("-" * 88)





def plot_pivot_plane_height_measurements(
    height_map: FloatArray,
    result: PivotPlaneHeightDifferenceResult,
) -> None:
    """
    Draw exactly two measurement rectangles for each Pivot surface:

    - left_strip_bounding_box
    - right_strip_bounding_box

    The central cross BB and its horizontal safety clearance are excluded.
    The outer measurement BB is intentionally not drawn, so it cannot be
    confused with the two strips that actually contribute to the mean.
    """
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(16, 7),
    )

    measurements = (
        result.upper_surface,
        result.lower_surface,
    )

    for axis, measurement in zip(
        axes,
        measurements,
    ):
        crop_box = measurement.measurement_bounding_box

        crop = height_map[
            crop_box.y_min:crop_box.y_max,
            crop_box.x_min:crop_box.x_max,
        ]

        left_mask_crop = measurement.left_strip_mask[
            crop_box.y_min:crop_box.y_max,
            crop_box.x_min:crop_box.x_max,
        ]
        right_mask_crop = measurement.right_strip_mask[
            crop_box.y_min:crop_box.y_max,
            crop_box.x_min:crop_box.x_max,
        ]
        exclusion_mask_crop = measurement.exclusion_box_mask[
            crop_box.y_min:crop_box.y_max,
            crop_box.x_min:crop_box.x_max,
        ]

        height_image = axis.imshow(
            crop,
            aspect="equal",
            interpolation="nearest",
        )
        figure.colorbar(
            height_image,
            ax=axis,
            label="Height",
        )

        # Make the exact pixels used by each strip visible.
        axis.imshow(
            np.ma.masked_where(
                ~left_mask_crop,
                left_mask_crop,
            ),
            alpha=0.35,
            aspect="equal",
            interpolation="nearest",
        )
        axis.imshow(
            np.ma.masked_where(
                ~right_mask_crop,
                right_mask_crop,
            ),
            alpha=0.35,
            aspect="equal",
            interpolation="nearest",
        )

        # Light overlay for the complete excluded central region.
        axis.imshow(
            np.ma.masked_where(
                ~exclusion_mask_crop,
                exclusion_mask_crop,
            ),
            alpha=0.16,
            aspect="equal",
            interpolation="nearest",
        )

        left_box = _translate_box(
            measurement.left_strip_bounding_box,
            dx=-crop_box.x_min,
            dy=-crop_box.y_min,
        )
        right_box = _translate_box(
            measurement.right_strip_bounding_box,
            dx=-crop_box.x_min,
            dy=-crop_box.y_min,
        )
        exclusion_box = _translate_box(
            measurement.exclusion_bounding_box,
            dx=-crop_box.x_min,
            dy=-crop_box.y_min,
        )
        cross_box = _translate_box(
            measurement.cross_bounding_box,
            dx=-crop_box.x_min,
            dy=-crop_box.y_min,
        )

        # These are the two rectangles the algorithm actually measures.
        left_rectangle = Rectangle(
            (
                left_box.x_min - 0.5,
                left_box.y_min - 0.5,
            ),
            left_box.width,
            left_box.height,
            fill=False,
            linewidth=4.0,
            linestyle="-",
            hatch="////",
            label="Left measured strip",
        )
        right_rectangle = Rectangle(
            (
                right_box.x_min - 0.5,
                right_box.y_min - 0.5,
            ),
            right_box.width,
            right_box.height,
            fill=False,
            linewidth=4.0,
            linestyle="--",
            hatch="\\\\",
            label="Right measured strip",
        )

        axis.add_patch(left_rectangle)
        axis.add_patch(right_rectangle)

        # Central excluded area: cross BB plus the extra safety clearance.
        axis.add_patch(
            Rectangle(
                (
                    exclusion_box.x_min - 0.5,
                    exclusion_box.y_min - 0.5,
                ),
                exclusion_box.width,
                exclusion_box.height,
                fill=False,
                linewidth=2.5,
                linestyle=":",
                label="Excluded cross + clearance",
            )
        )

        # Original cross BB, shown only as a reference.
        axis.add_patch(
            Rectangle(
                (
                    cross_box.x_min - 0.5,
                    cross_box.y_min - 0.5,
                ),
                cross_box.width,
                cross_box.height,
                fill=False,
                linewidth=1.5,
                linestyle="-.",
                label="Original cross BB",
            )
        )

        left_center_x = (
            left_box.x_min
            + left_box.width / 2.0
            - 0.5
        )
        right_center_x = (
            right_box.x_min
            + right_box.width / 2.0
            - 0.5
        )
        center_y = (
            left_box.y_min
            + left_box.height / 2.0
            - 0.5
        )

        axis.text(
            left_center_x,
            center_y,
            (
                "LEFT STRIP\n"
                f"{measurement.left_pixel_count} px\n"
                f"mean={measurement.left_mean_height:.6f}"
            ),
            ha="center",
            va="center",
        )
        axis.text(
            right_center_x,
            center_y,
            (
                "RIGHT STRIP\n"
                f"{measurement.right_pixel_count} px\n"
                f"mean={measurement.right_mean_height:.6f}"
            ),
            ha="center",
            va="center",
        )

        axis.set_xlim(
            -0.5,
            crop.shape[1] - 0.5,
        )
        axis.set_ylim(
            crop.shape[0] - 0.5,
            -0.5,
        )

        axis.set_title(
            f"{measurement.surface_name.capitalize()} surface — "
            "TWO MEASURED SIDE STRIPS\n"
            f"combined mean={measurement.mean_height:.6f}"
        )
        axis.set_xlabel("Local X [pixels]")
        axis.set_ylabel("Local Y [pixels]")
        axis.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.13),
        )

    figure.suptitle(
        "Stage 5 — Two side-strip rectangles for each Pivot surface\n"
        f"Plane 2 - Plane 1 = "
        f"{result.height_difference:.6f}"
    )
    figure.tight_layout()
    plt.show()


def _measure_surface_from_side_strips(
    height_map: FloatArray,
    surface_name: str,
    cross_candidate: LowerCrossCandidate | UpperCrossCandidate,
    cross_mask: BoolArray,
    allowed_surface_mask: BoolArray,
    allowed_bounding_box: BoundingBox,
    config: PivotPlaneHeightConfig,
) -> LocalPlaneHeightMeasurement:
    """
    Measure one surface from two horizontal strips beside a detected cross.

    Geometry:

        outer_left ... exclusion_left | excluded area |
        exclusion_right ... outer_right

    The excluded area contains the complete cross BB plus a small horizontal
    clearance beyond both BB edges.
    """
    image_height, image_width = height_map.shape
    cross_box = cross_candidate.bounding_box

    horizontal_margin = max(
        config.minimum_horizontal_margin_pixels,
        int(
            round(
                cross_box.width
                * config.horizontal_margin_fraction
            )
        ),
    )

    cross_clearance = max(
        config.minimum_cross_clearance_pixels,
        int(
            round(
                cross_box.width
                * config.cross_clearance_fraction
            )
        ),
    )

    if horizontal_margin <= cross_clearance:
        raise ValueError(
            f"The {surface_name} horizontal margin "
            f"({horizontal_margin}px) must be greater than the "
            f"cross clearance ({cross_clearance}px)."
        )

    # Outer measurement BB: expand only along X.
    measurement_box = BoundingBox(
        x_min=max(
            0,
            allowed_bounding_box.x_min,
            cross_box.x_min - horizontal_margin,
        ),
        y_min=max(
            0,
            allowed_bounding_box.y_min,
            cross_box.y_min,
        ),
        x_max=min(
            image_width,
            allowed_bounding_box.x_max,
            cross_box.x_max + horizontal_margin,
        ),
        y_max=min(
            image_height,
            allowed_bounding_box.y_max,
            cross_box.y_max,
        ),
    )

    # Central excluded rectangle: complete cross BB plus a small horizontal
    # clearance. Its Y range remains identical to the original cross BB.
    exclusion_box = BoundingBox(
        x_min=max(
            measurement_box.x_min,
            cross_box.x_min - cross_clearance,
        ),
        y_min=measurement_box.y_min,
        x_max=min(
            measurement_box.x_max,
            cross_box.x_max + cross_clearance,
        ),
        y_max=measurement_box.y_max,
    )

    left_strip_box = BoundingBox(
        x_min=measurement_box.x_min,
        y_min=measurement_box.y_min,
        x_max=exclusion_box.x_min,
        y_max=measurement_box.y_max,
    )

    right_strip_box = BoundingBox(
        x_min=exclusion_box.x_max,
        y_min=measurement_box.y_min,
        x_max=measurement_box.x_max,
        y_max=measurement_box.y_max,
    )

    _validate_measurement_geometry(
        surface_name=surface_name,
        measurement_box=measurement_box,
        exclusion_box=exclusion_box,
        left_strip_box=left_strip_box,
        right_strip_box=right_strip_box,
        config=config,
    )

    measurement_box_mask = _box_to_mask(
        box=measurement_box,
        image_shape=height_map.shape,
    )
    exclusion_box_mask = _box_to_mask(
        box=exclusion_box,
        image_shape=height_map.shape,
    )
    left_strip_box_mask = _box_to_mask(
        box=left_strip_box,
        image_shape=height_map.shape,
    )
    right_strip_box_mask = _box_to_mask(
        box=right_strip_box,
        image_shape=height_map.shape,
    )

    # The surface mask is used only as a geometric guard. No height-based or
    # statistical filtering is applied.
    left_strip_mask = (
        left_strip_box_mask
        & allowed_surface_mask
    )
    right_strip_mask = (
        right_strip_box_mask
        & allowed_surface_mask
    )

    measurement_mask = (
        left_strip_mask
        | right_strip_mask
    )

    # Defensive assertion: no central excluded pixel may enter the measurement.
    if np.any(
        measurement_mask
        & exclusion_box_mask
    ):
        raise RuntimeError(
            "Measurement mask overlaps the excluded cross region."
        )

    left_values = np.asarray(
        height_map[left_strip_mask],
        dtype=np.float64,
    )
    right_values = np.asarray(
        height_map[right_strip_mask],
        dtype=np.float64,
    )
    values = np.asarray(
        height_map[measurement_mask],
        dtype=np.float64,
    )

    if (
        left_values.size
        < config.minimum_pixels_per_strip
    ):
        raise ValueError(
            f"Only {left_values.size} pixels remain in the "
            f"{surface_name} left measurement strip. At least "
            f"{config.minimum_pixels_per_strip} are required."
        )

    if (
        right_values.size
        < config.minimum_pixels_per_strip
    ):
        raise ValueError(
            f"Only {right_values.size} pixels remain in the "
            f"{surface_name} right measurement strip. At least "
            f"{config.minimum_pixels_per_strip} are required."
        )

    if (
        values.size
        < config.minimum_measurement_pixels
    ):
        raise ValueError(
            f"Only {values.size} total pixels remain for the "
            f"{surface_name} surface. At least "
            f"{config.minimum_measurement_pixels} are required."
        )

    return LocalPlaneHeightMeasurement(
        surface_name=surface_name,
        cross_bounding_box=cross_box,
        exclusion_bounding_box=exclusion_box,
        measurement_bounding_box=measurement_box,
        left_strip_bounding_box=left_strip_box,
        right_strip_bounding_box=right_strip_box,
        cross_mask=cross_mask.astype(
            bool,
            copy=False,
        ),
        measurement_box_mask=measurement_box_mask,
        exclusion_box_mask=exclusion_box_mask,
        allowed_surface_mask=allowed_surface_mask.astype(
            bool,
            copy=False,
        ),
        left_strip_mask=left_strip_mask,
        right_strip_mask=right_strip_mask,
        measurement_mask=measurement_mask,
        horizontal_margin_pixels=horizontal_margin,
        cross_clearance_pixels=cross_clearance,
        pixel_count=int(values.size),
        left_pixel_count=int(left_values.size),
        right_pixel_count=int(right_values.size),
        mean_height=float(
            np.mean(values)
        ),
        median_height=float(
            np.median(values)
        ),
        standard_deviation=float(
            np.std(values)
        ),
        minimum_height=float(
            np.min(values)
        ),
        maximum_height=float(
            np.max(values)
        ),
        left_mean_height=float(
            np.mean(left_values)
        ),
        right_mean_height=float(
            np.mean(right_values)
        ),
    )


def _validate_measurement_geometry(
    surface_name: str,
    measurement_box: BoundingBox,
    exclusion_box: BoundingBox,
    left_strip_box: BoundingBox,
    right_strip_box: BoundingBox,
    config: PivotPlaneHeightConfig,
) -> None:
    if (
        measurement_box.width <= 0
        or measurement_box.height <= 0
    ):
        raise ValueError(
            f"The {surface_name} measurement bounding box is empty."
        )

    if (
        exclusion_box.width <= 0
        or exclusion_box.height <= 0
    ):
        raise ValueError(
            f"The {surface_name} exclusion bounding box is empty."
        )

    if (
        left_strip_box.width
        < config.minimum_strip_width_pixels
    ):
        raise ValueError(
            f"The {surface_name} left strip is only "
            f"{left_strip_box.width}px wide. At least "
            f"{config.minimum_strip_width_pixels}px are required."
        )

    if (
        right_strip_box.width
        < config.minimum_strip_width_pixels
    ):
        raise ValueError(
            f"The {surface_name} right strip is only "
            f"{right_strip_box.width}px wide. At least "
            f"{config.minimum_strip_width_pixels}px are required."
        )


def _box_to_mask(
    box: BoundingBox,
    image_shape: tuple[int, int],
) -> BoolArray:
    mask = np.zeros(
        image_shape,
        dtype=bool,
    )
    mask[
        box.y_min:box.y_max,
        box.x_min:box.x_max,
    ] = True
    return mask


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


def _get_lower_plane_box(
    candidate: LowerPlaneCandidate,
) -> BoundingBox:
    """
    Support both lower-plane candidate interfaces used during development.
    """
    bounding_box = getattr(
        candidate,
        "bounding_box",
        None,
    )

    if bounding_box is not None:
        return bounding_box

    outer_bounding_box = getattr(
        candidate,
        "outer_bounding_box",
        None,
    )

    if outer_bounding_box is not None:
        return outer_bounding_box

    raise AttributeError(
        "LowerPlaneCandidate must expose either "
        "'bounding_box' or 'outer_bounding_box'."
    )


def _validate_inputs(
    height_map: np.ndarray,
    lower_plane_detection: LowerPlaneDetectionResult,
    lower_cross_detection: LowerCrossDetectionResult,
    pivot_segmentation: PivotSegmentationResult,
    upper_cross_detection: UpperCrossDetectionResult,
    config: PivotPlaneHeightConfig,
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
            f"Expected numeric height data, received "
            f"{height_map.dtype}."
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

    if (
        pivot_segmentation.pivot_mask.shape
        != height_map.shape
    ):
        raise ValueError(
            "Pivot segmentation and height map must have the same shape."
        )

    if (
        lower_cross_detection.best_candidate
        is None
    ):
        raise ValueError(
            "No lower-cross candidate was selected."
        )

    if (
        upper_cross_detection.best_candidate
        is None
    ):
        raise ValueError(
            "No upper-cross candidate was selected."
        )

    if config.horizontal_margin_fraction <= 0:
        raise ValueError(
            "horizontal_margin_fraction must be positive."
        )

    if config.minimum_horizontal_margin_pixels < 1:
        raise ValueError(
            "minimum_horizontal_margin_pixels must be at least 1."
        )

    if config.cross_clearance_fraction < 0:
        raise ValueError(
            "cross_clearance_fraction cannot be negative."
        )

    if config.minimum_cross_clearance_pixels < 0:
        raise ValueError(
            "minimum_cross_clearance_pixels cannot be negative."
        )

    if config.minimum_strip_width_pixels < 1:
        raise ValueError(
            "minimum_strip_width_pixels must be at least 1."
        )

    if config.minimum_pixels_per_strip < 1:
        raise ValueError(
            "minimum_pixels_per_strip must be at least 1."
        )

    if config.minimum_measurement_pixels < 2:
        raise ValueError(
            "minimum_measurement_pixels must be at least 2."
        )

    valid_plane_names = {
        "upper",
        "lower",
    }

    if config.plane_1_name not in valid_plane_names:
        raise ValueError(
            "plane_1_name must be either 'upper' or 'lower'."
        )

    if config.plane_2_name not in valid_plane_names:
        raise ValueError(
            "plane_2_name must be either 'upper' or 'lower'."
        )

    if config.plane_1_name == config.plane_2_name:
        raise ValueError(
            "plane_1_name and plane_2_name must refer to different surfaces."
        )