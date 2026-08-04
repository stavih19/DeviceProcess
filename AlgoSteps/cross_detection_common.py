from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi

from AlgoSteps.step1_pivot_candidates import (
    BoolArray,
    BoundingBox,
    FloatArray,
    IntArray,
)


@dataclass(frozen=True)
class CrossDetectionCoreConfig:
    """Parameters shared by lower- and upper-cross detection."""

    small_gaussian_sigma: float = 0.8
    large_gaussian_sigma: float = 8.0
    threshold_mad_multiplier: float = 3.0

    opening_size: int = 1
    closing_size: int = 3

    min_area_fraction: float = 0.0005
    max_area_fraction: float = 0.08
    min_width_pixels: int = 3
    min_height_pixels: int = 3


@dataclass(frozen=True)
class CrossComponent:
    """Common measurements for one thresholded cross candidate."""

    component_label: int
    bounding_box: BoundingBox
    center_x: float
    center_y: float

    area_pixels: int
    area_fraction: float
    width_height_ratio: float
    fill_ratio: float

    horizontal_arm_coverage: float
    vertical_arm_coverage: float
    corner_occupancy: float

    mean_depth: float
    max_depth: float
    median_raw_height: float

    shape_score: float
    contrast_score: float


@dataclass
class CrossDetectionCoreResult:
    """Raw shared detector output in crop-local coordinates."""

    crop_origin_x: int
    crop_origin_y: int
    crop_shape: tuple[int, int]

    threshold: float
    robust_response_sigma: float

    response_map: NDArray[np.float32]
    search_mask: BoolArray
    threshold_mask: BoolArray
    label_image: IntArray

    components: list[CrossComponent] = field(default_factory=list)

    def get_component_mask_local(
        self,
        component: CrossComponent,
    ) -> BoolArray:
        return self.label_image == component.component_label

    def get_component_mask_global(
        self,
        component: CrossComponent,
        image_shape: tuple[int, int],
    ) -> BoolArray:
        global_mask = np.zeros(image_shape, dtype=bool)
        crop_height, crop_width = self.crop_shape

        global_mask[
            self.crop_origin_y:self.crop_origin_y + crop_height,
            self.crop_origin_x:self.crop_origin_x + crop_width,
        ] = self.get_component_mask_local(component)

        return global_mask


def detect_cross_components(
    height_crop: FloatArray,
    search_mask: BoolArray,
    crop_origin_x: int,
    crop_origin_y: int,
    config: CrossDetectionCoreConfig,
) -> CrossDetectionCoreResult:
    """
    Run the detector shared by both Pivot crosses.

    The cross is treated as a local depression, so the response is:

        large-scale local background - small-scale height map
    """
    _validate_core_inputs(
        height_crop=height_crop,
        search_mask=search_mask,
        config=config,
    )

    height_crop = height_crop.astype(np.float32, copy=False)
    search_mask = search_mask.astype(bool, copy=False)

    small_scale = ndi.gaussian_filter(
        height_crop,
        sigma=config.small_gaussian_sigma,
    )
    large_scale = ndi.gaussian_filter(
        height_crop,
        sigma=config.large_gaussian_sigma,
    )

    response_map = (large_scale - small_scale).astype(np.float32)

    response_values = response_map[search_mask]
    response_median = float(np.median(response_values))
    response_mad = float(
        np.median(np.abs(response_values - response_median))
    )
    robust_response_sigma = max(1.4826 * response_mad, 1e-8)

    threshold = float(
        response_median
        + config.threshold_mad_multiplier * robust_response_sigma
    )

    threshold_mask = (response_map >= threshold) & search_mask

    if config.opening_size > 1:
        threshold_mask = ndi.binary_opening(
            threshold_mask,
            structure=np.ones(
                (config.opening_size, config.opening_size),
                dtype=bool,
            ),
        )

    if config.closing_size > 1:
        threshold_mask = ndi.binary_closing(
            threshold_mask,
            structure=np.ones(
                (config.closing_size, config.closing_size),
                dtype=bool,
            ),
        )

    label_image, component_count = ndi.label(threshold_mask)
    label_image = label_image.astype(np.int32, copy=False)

    search_area = max(1, int(np.count_nonzero(search_mask)))
    min_area = max(
        1,
        int(round(search_area * config.min_area_fraction)),
    )
    max_area = max(
        min_area,
        int(round(search_area * config.max_area_fraction)),
    )

    components: list[CrossComponent] = []
    component_slices = ndi.find_objects(label_image)

    for component_label in range(1, component_count + 1):
        component_slice = component_slices[component_label - 1]
        if component_slice is None:
            continue

        local_component_mask = label_image == component_label
        area_pixels = int(np.count_nonzero(local_component_mask))

        if area_pixels < min_area or area_pixels > max_area:
            continue

        y_slice, x_slice = component_slice
        local_box = BoundingBox(
            x_min=int(x_slice.start),
            y_min=int(y_slice.start),
            x_max=int(x_slice.stop),
            y_max=int(y_slice.stop),
        )

        if (
            local_box.width < config.min_width_pixels
            or local_box.height < config.min_height_pixels
        ):
            continue

        local_y, local_x = np.nonzero(local_component_mask)
        center_x_local = float(np.mean(local_x))
        center_y_local = float(np.mean(local_y))

        global_box = BoundingBox(
            x_min=crop_origin_x + local_box.x_min,
            y_min=crop_origin_y + local_box.y_min,
            x_max=crop_origin_x + local_box.x_max,
            y_max=crop_origin_y + local_box.y_max,
        )

        area_fraction = area_pixels / search_area
        width_height_ratio = local_box.width / local_box.height

        shape = measure_cross_shape(
            local_component_mask=local_component_mask,
            local_box=local_box,
        )

        component_response = response_map[local_component_mask]
        mean_depth = float(np.mean(component_response))
        max_depth = float(np.max(component_response))
        median_raw_height = float(
            np.median(height_crop[local_component_mask])
        )

        shape_score = calculate_shape_score(
            width_height_ratio=width_height_ratio,
            fill_ratio=shape["fill_ratio"],
            horizontal_arm_coverage=shape[
                "horizontal_arm_coverage"
            ],
            vertical_arm_coverage=shape[
                "vertical_arm_coverage"
            ],
            corner_occupancy=shape["corner_occupancy"],
        )

        contrast_score = float(
            np.clip(
                (mean_depth - threshold)
                / (2.0 * robust_response_sigma),
                0.0,
                1.0,
            )
        )

        components.append(
            CrossComponent(
                component_label=component_label,
                bounding_box=global_box,
                center_x=crop_origin_x + center_x_local,
                center_y=crop_origin_y + center_y_local,
                area_pixels=area_pixels,
                area_fraction=float(area_fraction),
                width_height_ratio=float(width_height_ratio),
                fill_ratio=shape["fill_ratio"],
                horizontal_arm_coverage=shape[
                    "horizontal_arm_coverage"
                ],
                vertical_arm_coverage=shape[
                    "vertical_arm_coverage"
                ],
                corner_occupancy=shape["corner_occupancy"],
                mean_depth=mean_depth,
                max_depth=max_depth,
                median_raw_height=median_raw_height,
                shape_score=shape_score,
                contrast_score=contrast_score,
            )
        )

    return CrossDetectionCoreResult(
        crop_origin_x=crop_origin_x,
        crop_origin_y=crop_origin_y,
        crop_shape=height_crop.shape,
        threshold=threshold,
        robust_response_sigma=robust_response_sigma,
        response_map=response_map,
        search_mask=search_mask,
        threshold_mask=threshold_mask,
        label_image=label_image,
        components=components,
    )


def create_interior_search_mask(
    component_mask: BoolArray,
    edge_margin_pixels: int,
) -> BoolArray:
    """Remove a boundary band without creating a persistent inner box."""
    if component_mask.ndim != 2:
        raise ValueError("component_mask must be two-dimensional.")
    if edge_margin_pixels < 0:
        raise ValueError("edge_margin_pixels cannot be negative.")

    if edge_margin_pixels == 0:
        return component_mask.astype(bool, copy=True)

    distance_to_boundary = ndi.distance_transform_edt(component_mask)
    return component_mask & (distance_to_boundary >= edge_margin_pixels)


def measure_cross_shape(
    local_component_mask: BoolArray,
    local_box: BoundingBox,
) -> dict[str, float]:
    """Measure template-free plus-sign geometry."""
    crop = local_component_mask[
        local_box.y_min:local_box.y_max,
        local_box.x_min:local_box.x_max,
    ]

    if crop.size == 0 or not np.any(crop):
        return {
            "fill_ratio": 0.0,
            "horizontal_arm_coverage": 0.0,
            "vertical_arm_coverage": 0.0,
            "corner_occupancy": 1.0,
        }

    height, width = crop.shape
    fill_ratio = float(np.mean(crop))

    local_y, local_x = np.nonzero(crop)
    center_y = float(np.mean(local_y))
    center_x = float(np.mean(local_x))

    horizontal_half_band = max(1, int(round(height * 0.125)))
    vertical_half_band = max(1, int(round(width * 0.125)))

    horizontal_start = max(
        0,
        int(round(center_y)) - horizontal_half_band,
    )
    horizontal_stop = min(
        height,
        int(round(center_y)) + horizontal_half_band + 1,
    )
    vertical_start = max(
        0,
        int(round(center_x)) - vertical_half_band,
    )
    vertical_stop = min(
        width,
        int(round(center_x)) + vertical_half_band + 1,
    )

    horizontal_band = crop[horizontal_start:horizontal_stop, :]
    vertical_band = crop[:, vertical_start:vertical_stop]

    horizontal_arm_coverage = float(
        np.mean(np.any(horizontal_band, axis=0))
    )
    vertical_arm_coverage = float(
        np.mean(np.any(vertical_band, axis=1))
    )

    corner_height = max(1, height // 3)
    corner_width = max(1, width // 3)
    corner_pixels = np.concatenate(
        [
            crop[:corner_height, :corner_width].ravel(),
            crop[:corner_height, -corner_width:].ravel(),
            crop[-corner_height:, :corner_width].ravel(),
            crop[-corner_height:, -corner_width:].ravel(),
        ]
    )
    corner_occupancy = float(np.mean(corner_pixels))

    return {
        "fill_ratio": fill_ratio,
        "horizontal_arm_coverage": horizontal_arm_coverage,
        "vertical_arm_coverage": vertical_arm_coverage,
        "corner_occupancy": corner_occupancy,
    }


def calculate_shape_score(
    width_height_ratio: float,
    fill_ratio: float,
    horizontal_arm_coverage: float,
    vertical_arm_coverage: float,
    corner_occupancy: float,
) -> float:
    """Return a [0, 1] score for a plus-sign-like component."""
    aspect_score = float(
        np.exp(
            -abs(np.log(max(width_height_ratio, 1e-8))) / 0.7
        )
    )
    fill_score = float(
        np.exp(-0.5 * ((fill_ratio - 0.45) / 0.25) ** 2)
    )

    score = (
        0.20 * aspect_score
        + 0.25 * horizontal_arm_coverage
        + 0.25 * vertical_arm_coverage
        + 0.20 * (1.0 - corner_occupancy)
        + 0.10 * fill_score
    )
    return float(np.clip(score, 0.0, 1.0))


def calculate_center_score(
    center_x: float,
    center_y: float,
    expected_center_x: float,
    expected_center_y: float,
    reference_width: int,
    reference_height: int,
    tolerance_fraction: float,
) -> float:
    """Score normalized two-dimensional distance from an expected centre."""
    normalized_dx = (
        center_x - expected_center_x
    ) / max(reference_width / 2.0, 1.0)
    normalized_dy = (
        center_y - expected_center_y
    ) / max(reference_height / 2.0, 1.0)

    normalized_distance = float(np.hypot(normalized_dx, normalized_dy))
    tolerance = max(tolerance_fraction, 1e-8)

    return float(
        np.exp(-0.5 * (normalized_distance / tolerance) ** 2)
    )


def calculate_axis_alignment_score(
    candidate_position: float,
    expected_position: float,
    reference_size: int,
    tolerance_fraction: float,
) -> float:
    """Score distance from an expected position along one axis."""
    normalized_distance = (
        abs(candidate_position - expected_position)
        / max(reference_size / 2.0, 1.0)
    )
    tolerance = max(tolerance_fraction, 1e-8)

    return float(
        np.exp(-0.5 * (normalized_distance / tolerance) ** 2)
    )


def calculate_area_score(
    area_fraction: float,
    expected_area_fraction: float,
    tolerance_factor: float,
) -> float:
    return log_ratio_score(
        actual=area_fraction,
        expected=expected_area_fraction,
        tolerance_factor=tolerance_factor,
    )


def calculate_size_similarity_score(
    area_pixels: int,
    width: int,
    height: int,
    expected_area_pixels: int,
    expected_width: int,
    expected_height: int,
    tolerance_factor: float,
) -> float:
    """Compare candidate dimensions with a previously detected cross."""
    scores = (
        log_ratio_score(
            actual=float(area_pixels),
            expected=float(expected_area_pixels),
            tolerance_factor=tolerance_factor,
        ),
        log_ratio_score(
            actual=float(width),
            expected=float(expected_width),
            tolerance_factor=tolerance_factor,
        ),
        log_ratio_score(
            actual=float(height),
            expected=float(expected_height),
            tolerance_factor=tolerance_factor,
        ),
    )
    return float(np.mean(scores))


def log_ratio_score(
    actual: float,
    expected: float,
    tolerance_factor: float,
) -> float:
    if actual <= 0 or expected <= 0:
        return 0.0
    if tolerance_factor <= 1:
        raise ValueError("tolerance_factor must be greater than 1.")

    logarithmic_distance = abs(np.log(actual / expected))
    logarithmic_tolerance = max(np.log(tolerance_factor), 1e-8)

    return float(
        np.exp(
            -0.5
            * (logarithmic_distance / logarithmic_tolerance) ** 2
        )
    )


def _validate_core_inputs(
    height_crop: np.ndarray,
    search_mask: np.ndarray,
    config: CrossDetectionCoreConfig,
) -> None:
    if height_crop.ndim != 2:
        raise ValueError(
            "Expected a two-dimensional height crop, "
            f"received shape {height_crop.shape}."
        )
    if search_mask.shape != height_crop.shape:
        raise ValueError(
            "search_mask and height_crop must have the same shape."
        )
    if not np.issubdtype(height_crop.dtype, np.number):
        raise TypeError(
            f"Expected numeric height data, received {height_crop.dtype}."
        )
    if not np.isfinite(height_crop).all():
        raise ValueError("Height crop contains NaN or infinite values.")
    if not np.any(search_mask):
        raise ValueError("search_mask does not contain any searchable pixels.")

    if config.small_gaussian_sigma < 0:
        raise ValueError("small_gaussian_sigma cannot be negative.")
    if config.large_gaussian_sigma <= config.small_gaussian_sigma:
        raise ValueError(
            "large_gaussian_sigma must be greater than "
            "small_gaussian_sigma."
        )
    if config.threshold_mad_multiplier <= 0:
        raise ValueError("threshold_mad_multiplier must be positive.")
    if config.opening_size < 1 or config.closing_size < 1:
        raise ValueError("Morphological sizes must be at least 1.")
    if (
        config.min_area_fraction < 0
        or config.max_area_fraction <= 0
        or config.min_area_fraction >= config.max_area_fraction
    ):
        raise ValueError("Invalid cross component area fractions.")
    if config.min_width_pixels < 1 or config.min_height_pixels < 1:
        raise ValueError("Minimum component dimensions must be positive.")
