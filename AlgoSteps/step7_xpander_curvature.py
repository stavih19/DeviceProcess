from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from numpy.typing import NDArray
from scipy import ndimage as ndi
from scipy.optimize import least_squares

try:
    from AlgoSteps.step6_xpander_segmentation_v4 import XpanderSegmentationResult
except ImportError:
    from AlgoSteps.step6_xpander_segmentation_v4 import XpanderSegmentationResult


FloatArray = NDArray[np.floating]
BoolArray = NDArray[np.bool_]
AxisName = Literal["X", "Y"]


@dataclass(frozen=True)
class XpanderCurvatureConfig:
    """
    Configuration for Stage 7.

    Unit assumption used by this stage:
    - X and Y coordinates are converted from pixels to micrometres using
      ``pixel_pitch_um``.
    - Height-map values are treated as Z heights already expressed in
      micrometres, because no separate Z conversion factor is provided.

    The detector measures curvature around the geometric centre of the
    Stage-6 Xpander bounding box. Several parallel profiles are fitted in a
    narrow central band. The final radius for each axis is the median radius
    of the valid central profiles. A circle fitted to the median profile is
    retained as an additional diagnostic.
    """

    pixel_pitch_um: float = 0.252

    # Mild smoothing reduces pixel-scale measurement noise without flattening
    # the large Xpander curvature.
    gaussian_sigma_pixels: float = 0.75

    # Width of the band perpendicular to the measured axis.
    central_band_fraction: float = 0.08
    min_band_half_width_pixels: int = 2
    max_band_half_width_pixels: int = 15

    # Fraction of the Xpander width/height used around the centre. Avoiding the
    # outer part reduces contamination from the separator, halo and corners.
    x_fit_span_fraction: float = 0.70
    y_fit_span_fraction: float = 0.70

    # Optional erosion keeps fitting points away from the Stage-6 mask edge.
    # Keep this small because the current Stage-6 mask is intentionally
    # conservative.
    mask_erosion_pixels: int = 1

    min_profile_points: int = 30
    min_lateral_span_um: float = 5.0
    min_height_span_um: float = 0.015

    # Remove only isolated profile spikes relative to a local median. This is
    # not a height clipping operation and therefore preserves the curvature.
    profile_median_filter_size: int = 5
    profile_spike_mad_multiplier: float = 5.0

    # Robust nonlinear circle fit.
    robust_loss: Literal["linear", "soft_l1", "huber", "cauchy", "arctan"] = (
        "soft_l1"
    )
    max_fit_evaluations: int = 5000

    # Reject clearly unstable profile fits.
    max_normalized_radial_rmse: float = 0.30
    minimum_fit_r_squared: float = 0.50

    # Robust filtering of radii estimated from neighbouring central profiles.
    radius_outlier_mad_multiplier: float = 3.5
    minimum_successful_profiles: int = 3
    maximum_radius_coefficient_of_variation: float = 0.35

    confidence_threshold: float = 0.55

    # Debug visualization.
    detection_line_width: float = 1.0
    figure_size: tuple[float, float] = (15.0, 10.0)


@dataclass(frozen=True)
class CircleFitResult:
    """Circle fitted to one X-Z or Y-Z profile, all values in micrometres."""

    center_lateral_um: float
    center_z_um: float
    radius_um: float

    radial_rmse_um: float
    profile_rmse_um: float
    normalized_radial_rmse: float
    r_squared: float

    lateral_span_um: float
    height_span_um: float
    point_count: int
    branch_sign: int

    success: bool
    message: str


@dataclass(frozen=True)
class CurvatureProfileResult:
    """Measurements and fit for one central row or column."""

    axis: AxisName
    line_index_pixel: int
    lateral_pixels: NDArray[np.int64]
    lateral_um: NDArray[np.float64]
    heights_um: NDArray[np.float64]
    fitted_heights_um: NDArray[np.float64]
    circle: CircleFitResult
    retained_for_final_radius: bool


@dataclass(frozen=True)
class AxisCurvatureResult:
    """Curvature measurement for one lateral axis."""

    axis: AxisName
    radius_um: float
    radius_std_um: float
    radius_mad_um: float
    radius_min_um: float
    radius_max_um: float

    central_line_pixel: int
    band_start_pixel: int
    band_stop_pixel: int
    fit_start_pixel: int
    fit_stop_pixel: int

    attempted_profile_count: int
    successful_profile_count: int
    retained_profile_count: int

    aggregate_profile: CurvatureProfileResult
    profile_results: tuple[CurvatureProfileResult, ...]

    confidence: float
    is_confident: bool


@dataclass(frozen=True)
class XpanderCurvatureResult:
    """Complete Stage-7 result."""

    radius_x_um: float
    radius_y_um: float

    x_axis: AxisCurvatureResult
    y_axis: AxisCurvatureResult

    xpander_center_x_pixel: float
    xpander_center_y_pixel: float
    pixel_pitch_um: float

    confidence: float
    is_confident: bool
    smoothed_height_map: NDArray[np.float64]


def measure_xpander_curvature(
    height_map: FloatArray,
    xpander_segmentation: XpanderSegmentationResult,
    config: XpanderCurvatureConfig | None = None,
) -> XpanderCurvatureResult:
    """
    Measure the Xpander radius of curvature around its centre.

    ``R_x`` is measured from central X-Z profiles (rows), and ``R_y`` from
    central Y-Z profiles (columns). X/Y are converted using the configured
    pixel pitch; Z values are used directly as micrometres.
    """
    if config is None:
        config = XpanderCurvatureConfig()

    _validate_inputs(height_map, xpander_segmentation, config)

    height_float = np.asarray(height_map, dtype=np.float64)
    smoothed = ndi.gaussian_filter(
        height_float,
        sigma=config.gaussian_sigma_pixels,
    ).astype(np.float64)

    mask = np.asarray(xpander_segmentation.xpander_mask, dtype=bool)
    fitting_mask = _prepare_fitting_mask(mask, config.mask_erosion_pixels)

    box = xpander_segmentation.bounding_box
    center_x = 0.5 * (box.x_min + box.x_max - 1)
    center_y = 0.5 * (box.y_min + box.y_max - 1)

    x_axis = _measure_axis_curvature(
        axis="X",
        smoothed=smoothed,
        fitting_mask=fitting_mask,
        center_x=center_x,
        center_y=center_y,
        box=box,
        config=config,
    )
    y_axis = _measure_axis_curvature(
        axis="Y",
        smoothed=smoothed,
        fitting_mask=fitting_mask,
        center_x=center_x,
        center_y=center_y,
        box=box,
        config=config,
    )

    confidence = float(np.sqrt(x_axis.confidence * y_axis.confidence))
    is_confident = bool(
        x_axis.is_confident
        and y_axis.is_confident
        and confidence >= config.confidence_threshold
    )

    return XpanderCurvatureResult(
        radius_x_um=x_axis.radius_um,
        radius_y_um=y_axis.radius_um,
        x_axis=x_axis,
        y_axis=y_axis,
        xpander_center_x_pixel=float(center_x),
        xpander_center_y_pixel=float(center_y),
        pixel_pitch_um=config.pixel_pitch_um,
        confidence=confidence,
        is_confident=is_confident,
        smoothed_height_map=smoothed,
    )


def _validate_inputs(
    height_map: FloatArray,
    xpander_segmentation: XpanderSegmentationResult,
    config: XpanderCurvatureConfig,
) -> None:
    if not isinstance(height_map, np.ndarray):
        raise TypeError("height_map must be a NumPy array.")
    if height_map.ndim != 2:
        raise ValueError("height_map must be two-dimensional.")
    if not np.issubdtype(height_map.dtype, np.number):
        raise TypeError("height_map must contain numeric values.")
    if not np.all(np.isfinite(height_map)):
        raise ValueError("height_map contains NaN or infinite values.")

    mask = np.asarray(xpander_segmentation.xpander_mask)
    if mask.shape != height_map.shape:
        raise ValueError(
            "xpander_mask must have the same shape as height_map. "
            f"Received {mask.shape} and {height_map.shape}."
        )
    if np.count_nonzero(mask) < config.min_profile_points:
        raise ValueError("The Xpander mask contains too few pixels.")

    box = xpander_segmentation.bounding_box
    height, width = height_map.shape
    if not (
        0 <= box.x_min < box.x_max <= width
        and 0 <= box.y_min < box.y_max <= height
    ):
        raise ValueError(f"Invalid Xpander bounding box: {box}.")

    if config.pixel_pitch_um <= 0:
        raise ValueError("pixel_pitch_um must be positive.")
    if config.gaussian_sigma_pixels < 0:
        raise ValueError("gaussian_sigma_pixels cannot be negative.")
    if not 0 < config.central_band_fraction <= 1:
        raise ValueError("central_band_fraction must be in (0, 1].")
    if not 0 < config.x_fit_span_fraction <= 1:
        raise ValueError("x_fit_span_fraction must be in (0, 1].")
    if not 0 < config.y_fit_span_fraction <= 1:
        raise ValueError("y_fit_span_fraction must be in (0, 1].")
    if config.min_profile_points < 3:
        raise ValueError("min_profile_points must be at least 3.")


def _prepare_fitting_mask(mask: BoolArray, erosion_pixels: int) -> BoolArray:
    if erosion_pixels <= 0:
        return mask.copy()

    eroded = ndi.binary_erosion(mask, iterations=erosion_pixels)

    # Do not allow erosion to destroy a conservative Stage-6 mask.
    if np.count_nonzero(eroded) < 0.50 * np.count_nonzero(mask):
        return mask.copy()
    return np.asarray(eroded, dtype=bool)


def _measure_axis_curvature(
    *,
    axis: AxisName,
    smoothed: NDArray[np.float64],
    fitting_mask: BoolArray,
    center_x: float,
    center_y: float,
    box: object,
    config: XpanderCurvatureConfig,
) -> AxisCurvatureResult:
    if axis == "X":
        central_line = int(round(center_y))
        orthogonal_size = int(box.y_max - box.y_min)
        band_half_width = _band_half_width(orthogonal_size, config)
        band_start = max(int(box.y_min), central_line - band_half_width)
        band_stop = min(int(box.y_max), central_line + band_half_width + 1)

        fit_start, fit_stop = _centered_interval(
            low=int(box.x_min),
            high=int(box.x_max),
            center=center_x,
            fraction=config.x_fit_span_fraction,
        )
        line_indices = range(band_start, band_stop)
    else:
        central_line = int(round(center_x))
        orthogonal_size = int(box.x_max - box.x_min)
        band_half_width = _band_half_width(orthogonal_size, config)
        band_start = max(int(box.x_min), central_line - band_half_width)
        band_stop = min(int(box.x_max), central_line + band_half_width + 1)

        fit_start, fit_stop = _centered_interval(
            low=int(box.y_min),
            high=int(box.y_max),
            center=center_y,
            fraction=config.y_fit_span_fraction,
        )
        line_indices = range(band_start, band_stop)

    profile_results: list[CurvatureProfileResult] = []

    for line_index in line_indices:
        extracted = _extract_profile(
            axis=axis,
            line_index=line_index,
            fit_start=fit_start,
            fit_stop=fit_stop,
            smoothed=smoothed,
            fitting_mask=fitting_mask,
            central_lateral_position=(center_x if axis == "X" else center_y),
            config=config,
        )
        if extracted is None:
            continue

        lateral_pixels, lateral_um, heights_um = extracted
        circle, fitted_heights = _fit_circle_to_profile(
            lateral_um=lateral_um,
            heights_um=heights_um,
            config=config,
        )
        if not circle.success:
            continue

        profile_results.append(
            CurvatureProfileResult(
                axis=axis,
                line_index_pixel=int(line_index),
                lateral_pixels=lateral_pixels,
                lateral_um=lateral_um,
                heights_um=heights_um,
                fitted_heights_um=fitted_heights,
                circle=circle,
                retained_for_final_radius=False,
            )
        )

    aggregate_profile = _build_and_fit_aggregate_profile(
        axis=axis,
        band_start=band_start,
        band_stop=band_stop,
        fit_start=fit_start,
        fit_stop=fit_stop,
        smoothed=smoothed,
        fitting_mask=fitting_mask,
        central_lateral_position=(center_x if axis == "X" else center_y),
        config=config,
    )

    if not profile_results and not aggregate_profile.circle.success:
        raise ValueError(
            f"No valid central {axis}-axis profiles could be fitted to a circle."
        )

    retained_indices = _select_consistent_profile_indices(
        profile_results,
        config.radius_outlier_mad_multiplier,
    )

    updated_profiles: list[CurvatureProfileResult] = []
    retained_radii: list[float] = []
    for index, profile in enumerate(profile_results):
        retained = index in retained_indices
        if retained:
            retained_radii.append(profile.circle.radius_um)
        updated_profiles.append(
            CurvatureProfileResult(
                axis=profile.axis,
                line_index_pixel=profile.line_index_pixel,
                lateral_pixels=profile.lateral_pixels,
                lateral_um=profile.lateral_um,
                heights_um=profile.heights_um,
                fitted_heights_um=profile.fitted_heights_um,
                circle=profile.circle,
                retained_for_final_radius=retained,
            )
        )

    if retained_radii:
        radii = np.asarray(retained_radii, dtype=np.float64)
    elif aggregate_profile.circle.success:
        radii = np.asarray([aggregate_profile.circle.radius_um], dtype=np.float64)
    else:
        raise ValueError(f"No consistent {axis}-axis curvature radii remained.")

    radius = float(np.median(radii))
    radius_std = float(np.std(radii))
    radius_mad = float(np.median(np.abs(radii - radius)))
    radius_min = float(np.min(radii))
    radius_max = float(np.max(radii))

    attempted_count = band_stop - band_start
    successful_count = len(profile_results)
    retained_count = int(radii.size)

    profile_fit_quality = _mean_profile_quality(
        [updated_profiles[index] for index in sorted(retained_indices)]
    )
    if not retained_indices and aggregate_profile.circle.success:
        profile_fit_quality = _circle_quality(aggregate_profile.circle, config)

    success_ratio = min(
        1.0,
        successful_count / max(config.minimum_successful_profiles, attempted_count),
    )
    retained_ratio = retained_count / max(1, successful_count)

    coefficient_of_variation = radius_std / max(radius, 1e-12)
    consistency_score = float(
        np.exp(
            -coefficient_of_variation
            / max(config.maximum_radius_coefficient_of_variation, 1e-12)
        )
    )

    aggregate_score = (
        _circle_quality(aggregate_profile.circle, config)
        if aggregate_profile.circle.success
        else 0.0
    )

    confidence = float(
        np.clip(
            0.40 * profile_fit_quality
            + 0.20 * success_ratio
            + 0.15 * retained_ratio
            + 0.15 * consistency_score
            + 0.10 * aggregate_score,
            0.0,
            1.0,
        )
    )

    enough_profiles = (
        retained_count >= config.minimum_successful_profiles
        or (
            aggregate_profile.circle.success
            and successful_count > 0
            and retained_count >= 1
        )
    )
    is_confident = bool(
        enough_profiles
        and coefficient_of_variation
        <= config.maximum_radius_coefficient_of_variation
        and confidence >= config.confidence_threshold
    )

    return AxisCurvatureResult(
        axis=axis,
        radius_um=radius,
        radius_std_um=radius_std,
        radius_mad_um=radius_mad,
        radius_min_um=radius_min,
        radius_max_um=radius_max,
        central_line_pixel=central_line,
        band_start_pixel=band_start,
        band_stop_pixel=band_stop,
        fit_start_pixel=fit_start,
        fit_stop_pixel=fit_stop,
        attempted_profile_count=attempted_count,
        successful_profile_count=successful_count,
        retained_profile_count=retained_count,
        aggregate_profile=aggregate_profile,
        profile_results=tuple(updated_profiles),
        confidence=confidence,
        is_confident=is_confident,
    )


def _band_half_width(
    orthogonal_size: int,
    config: XpanderCurvatureConfig,
) -> int:
    requested = int(round(0.5 * config.central_band_fraction * orthogonal_size))
    return int(
        np.clip(
            requested,
            config.min_band_half_width_pixels,
            config.max_band_half_width_pixels,
        )
    )


def _centered_interval(
    *,
    low: int,
    high: int,
    center: float,
    fraction: float,
) -> tuple[int, int]:
    available = high - low
    length = max(3, int(round(fraction * available)))
    length = min(length, available)

    start = int(round(center - 0.5 * length))
    start = max(low, min(start, high - length))
    stop = start + length
    return start, stop


def _extract_profile(
    *,
    axis: AxisName,
    line_index: int,
    fit_start: int,
    fit_stop: int,
    smoothed: NDArray[np.float64],
    fitting_mask: BoolArray,
    central_lateral_position: float,
    config: XpanderCurvatureConfig,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]] | None:
    if axis == "X":
        lateral_pixels = np.arange(fit_start, fit_stop, dtype=np.int64)
        valid = fitting_mask[line_index, fit_start:fit_stop]
        heights = smoothed[line_index, fit_start:fit_stop]
    else:
        lateral_pixels = np.arange(fit_start, fit_stop, dtype=np.int64)
        valid = fitting_mask[fit_start:fit_stop, line_index]
        heights = smoothed[fit_start:fit_stop, line_index]

    run = _largest_contiguous_true_run_near_center(
        valid=np.asarray(valid, dtype=bool),
        absolute_positions=lateral_pixels,
        center=central_lateral_position,
    )
    if run is None:
        return None

    run_start, run_stop = run
    lateral_pixels = lateral_pixels[run_start:run_stop]
    heights = np.asarray(heights[run_start:run_stop], dtype=np.float64)

    if lateral_pixels.size < config.min_profile_points:
        return None

    keep = _remove_isolated_profile_spikes(heights, config)
    lateral_pixels = lateral_pixels[keep]
    heights = heights[keep]

    if lateral_pixels.size < config.min_profile_points:
        return None

    lateral_um = (
        lateral_pixels.astype(np.float64)
        - float(np.mean(lateral_pixels.astype(np.float64)))
    ) * config.pixel_pitch_um

    if np.ptp(lateral_um) < config.min_lateral_span_um:
        return None
    if np.ptp(heights) < config.min_height_span_um:
        return None

    return lateral_pixels, lateral_um, heights


def _largest_contiguous_true_run_near_center(
    *,
    valid: BoolArray,
    absolute_positions: NDArray[np.int64],
    center: float,
) -> tuple[int, int] | None:
    if valid.size == 0 or not np.any(valid):
        return None

    padded = np.pad(valid.astype(np.int8), (1, 1), constant_values=0)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)

    if starts.size == 0:
        return None

    best: tuple[float, int, int] | None = None
    for start, stop in zip(starts, stops, strict=True):
        run_positions = absolute_positions[start:stop]
        if run_positions.size == 0:
            continue
        distance = float(np.min(np.abs(run_positions.astype(float) - center)))
        length = int(stop - start)
        key = (distance, -length, int(start))
        if best is None or key < (best[0], -best[1], best[2]):
            best = (distance, length, int(start))
            best_stop = int(stop)

    if best is None:
        return None
    return best[2], best_stop


def _remove_isolated_profile_spikes(
    heights: NDArray[np.float64],
    config: XpanderCurvatureConfig,
) -> BoolArray:
    size = max(1, int(config.profile_median_filter_size))
    if size % 2 == 0:
        size += 1
    if size <= 1 or heights.size < size:
        return np.ones(heights.shape, dtype=bool)

    local_median = ndi.median_filter(heights, size=size, mode="nearest")
    residual = heights - local_median
    residual_median = float(np.median(residual))
    residual_mad = float(np.median(np.abs(residual - residual_median)))
    robust_sigma = 1.4826 * residual_mad

    if robust_sigma <= 1e-12:
        return np.ones(heights.shape, dtype=bool)

    return np.abs(residual - residual_median) <= (
        config.profile_spike_mad_multiplier * robust_sigma
    )


def _fit_circle_to_profile(
    *,
    lateral_um: NDArray[np.float64],
    heights_um: NDArray[np.float64],
    config: XpanderCurvatureConfig,
) -> tuple[CircleFitResult, NDArray[np.float64]]:
    if lateral_um.size < 3:
        return _failed_circle("At least three points are required."), np.full_like(
            heights_um, np.nan
        )

    x_offset = float(np.mean(lateral_um))
    z_offset = float(np.mean(heights_um))
    x = lateral_um - x_offset
    z = heights_um - z_offset

    initial = _algebraic_circle_initialization(x, z)
    if initial is None:
        return _failed_circle("Algebraic circle initialization failed."), np.full_like(
            heights_um, np.nan
        )

    center_x_0, center_z_0, radius_0 = initial
    if not np.isfinite(radius_0) or radius_0 <= 0:
        return _failed_circle("Invalid initial radius."), np.full_like(
            heights_um, np.nan
        )

    local_noise = _estimate_profile_noise(heights_um)
    f_scale = max(local_noise, 1e-4)

    def residuals(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        center_x, center_z, log_radius = parameters
        radius = float(np.exp(log_radius))
        distances = np.hypot(x - center_x, z - center_z)
        return distances - radius

    try:
        optimization = least_squares(
            residuals,
            x0=np.asarray(
                [center_x_0, center_z_0, np.log(max(radius_0, 1e-9))],
                dtype=np.float64,
            ),
            loss=config.robust_loss,
            f_scale=f_scale,
            max_nfev=config.max_fit_evaluations,
            x_scale="jac",
        )
    except (FloatingPointError, ValueError, RuntimeError) as error:
        return _failed_circle(f"Circle optimization failed: {error}"), np.full_like(
            heights_um, np.nan
        )

    center_x = float(optimization.x[0])
    center_z = float(optimization.x[1])
    radius = float(np.exp(optimization.x[2]))

    if not all(np.isfinite([center_x, center_z, radius])) or radius <= 0:
        return _failed_circle("The optimized circle parameters are invalid."), np.full_like(
            heights_um, np.nan
        )

    radial_residuals = residuals(optimization.x)
    radial_rmse = float(np.sqrt(np.mean(np.square(radial_residuals))))

    fitted_center_x_global = center_x + x_offset
    fitted_center_z_global = center_z + z_offset
    fitted_heights, branch_sign = _circle_branch_predictions(
        lateral_um=lateral_um,
        observed_heights_um=heights_um,
        center_lateral_um=fitted_center_x_global,
        center_z_um=fitted_center_z_global,
        radius_um=radius,
    )

    if not np.all(np.isfinite(fitted_heights)):
        return _failed_circle("The fitted circle does not cover the profile span."), np.full_like(
            heights_um, np.nan
        )

    profile_residuals = heights_um - fitted_heights
    profile_rmse = float(np.sqrt(np.mean(np.square(profile_residuals))))
    height_span = float(np.ptp(heights_um))
    lateral_span = float(np.ptp(lateral_um))
    normalization = max(height_span, 3.0 * local_noise, 1e-6)
    normalized_radial_rmse = radial_rmse / normalization

    total_sum_squares = float(
        np.sum(np.square(heights_um - float(np.mean(heights_um))))
    )
    residual_sum_squares = float(np.sum(np.square(profile_residuals)))
    r_squared = (
        1.0 - residual_sum_squares / total_sum_squares
        if total_sum_squares > 1e-12
        else 0.0
    )

    minimum_possible_radius = 0.5 * lateral_span
    accepted = bool(
        optimization.success
        and radius >= minimum_possible_radius
        and normalized_radial_rmse <= config.max_normalized_radial_rmse
        and r_squared >= config.minimum_fit_r_squared
    )

    message = str(optimization.message)
    if not accepted:
        message = (
            "Circle fit was numerically completed but failed quality checks: "
            f"normalized_rmse={normalized_radial_rmse:.4f}, "
            f"r_squared={r_squared:.4f}."
        )

    return (
        CircleFitResult(
            center_lateral_um=fitted_center_x_global,
            center_z_um=fitted_center_z_global,
            radius_um=radius,
            radial_rmse_um=radial_rmse,
            profile_rmse_um=profile_rmse,
            normalized_radial_rmse=normalized_radial_rmse,
            r_squared=float(r_squared),
            lateral_span_um=lateral_span,
            height_span_um=height_span,
            point_count=int(lateral_um.size),
            branch_sign=branch_sign,
            success=accepted,
            message=message,
        ),
        fitted_heights,
    )


def _algebraic_circle_initialization(
    x: NDArray[np.float64],
    z: NDArray[np.float64],
) -> tuple[float, float, float] | None:
    # Solve x^2 + z^2 = 2*a*x + 2*b*z + c.
    design = np.column_stack((2.0 * x, 2.0 * z, np.ones_like(x)))
    target = np.square(x) + np.square(z)

    try:
        solution, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    except np.linalg.LinAlgError:
        return None

    if rank < 3:
        # For a very shallow arc, initialize from a quadratic curvature fit.
        return _quadratic_circle_initialization(x, z)

    center_x = float(solution[0])
    center_z = float(solution[1])
    radius_squared = float(solution[2] + center_x**2 + center_z**2)
    if radius_squared <= 0 or not np.isfinite(radius_squared):
        return _quadratic_circle_initialization(x, z)

    radius = float(np.sqrt(radius_squared))
    return center_x, center_z, radius


def _quadratic_circle_initialization(
    x: NDArray[np.float64],
    z: NDArray[np.float64],
) -> tuple[float, float, float] | None:
    try:
        coefficient_2, coefficient_1, coefficient_0 = np.polyfit(x, z, deg=2)
    except (np.linalg.LinAlgError, ValueError):
        return None

    if abs(coefficient_2) <= 1e-12:
        return None

    vertex_x = float(-coefficient_1 / (2.0 * coefficient_2))
    vertex_z = float(
        coefficient_2 * vertex_x**2
        + coefficient_1 * vertex_x
        + coefficient_0
    )
    radius = float(1.0 / (2.0 * abs(coefficient_2)))

    # The circle centre is on the concave side of the fitted parabola.
    center_z = vertex_z + np.sign(coefficient_2) * radius
    return vertex_x, center_z, radius


def _estimate_profile_noise(heights_um: NDArray[np.float64]) -> float:
    if heights_um.size < 3:
        return 1e-4
    second_difference = np.diff(heights_um, n=2)
    if second_difference.size == 0:
        return 1e-4
    median = float(np.median(second_difference))
    mad = float(np.median(np.abs(second_difference - median)))
    # The exact conversion is not critical; this is used only as robust-loss
    # scale rather than as a physical uncertainty estimate.
    return max(1.4826 * mad, 1e-4)


def _circle_branch_predictions(
    *,
    lateral_um: NDArray[np.float64],
    observed_heights_um: NDArray[np.float64],
    center_lateral_um: float,
    center_z_um: float,
    radius_um: float,
) -> tuple[NDArray[np.float64], int]:
    inside = radius_um**2 - np.square(lateral_um - center_lateral_um)
    tolerance = max(radius_um**2 * 1e-10, 1e-12)
    if np.any(inside < -tolerance):
        return np.full_like(observed_heights_um, np.nan), 0

    root = np.sqrt(np.maximum(inside, 0.0))
    upper = center_z_um + root
    lower = center_z_um - root

    upper_rmse = float(np.sqrt(np.mean(np.square(observed_heights_um - upper))))
    lower_rmse = float(np.sqrt(np.mean(np.square(observed_heights_um - lower))))
    if upper_rmse <= lower_rmse:
        return upper, 1
    return lower, -1


def _failed_circle(message: str) -> CircleFitResult:
    return CircleFitResult(
        center_lateral_um=float("nan"),
        center_z_um=float("nan"),
        radius_um=float("nan"),
        radial_rmse_um=float("inf"),
        profile_rmse_um=float("inf"),
        normalized_radial_rmse=float("inf"),
        r_squared=float("-inf"),
        lateral_span_um=0.0,
        height_span_um=0.0,
        point_count=0,
        branch_sign=0,
        success=False,
        message=message,
    )


def _build_and_fit_aggregate_profile(
    *,
    axis: AxisName,
    band_start: int,
    band_stop: int,
    fit_start: int,
    fit_stop: int,
    smoothed: NDArray[np.float64],
    fitting_mask: BoolArray,
    central_lateral_position: float,
    config: XpanderCurvatureConfig,
) -> CurvatureProfileResult:
    if axis == "X":
        values = smoothed[band_start:band_stop, fit_start:fit_stop]
        valid = fitting_mask[band_start:band_stop, fit_start:fit_stop]
        line_index = int(round(0.5 * (band_start + band_stop - 1)))
    else:
        values = smoothed[fit_start:fit_stop, band_start:band_stop].T
        valid = fitting_mask[fit_start:fit_stop, band_start:band_stop].T
        line_index = int(round(0.5 * (band_start + band_stop - 1)))

    masked_values = np.where(valid, values, np.nan)
    with np.errstate(all="ignore"):
        aggregate_heights = np.nanmedian(masked_values, axis=0)

    lateral_pixels = np.arange(fit_start, fit_stop, dtype=np.int64)
    finite = np.isfinite(aggregate_heights)
    run = _largest_contiguous_true_run_near_center(
        valid=finite,
        absolute_positions=lateral_pixels,
        center=central_lateral_position,
    )

    if run is None:
        return _failed_profile(axis, line_index, "No aggregate profile segment.")

    run_start, run_stop = run
    lateral_pixels = lateral_pixels[run_start:run_stop]
    heights = np.asarray(aggregate_heights[run_start:run_stop], dtype=np.float64)

    if lateral_pixels.size < config.min_profile_points:
        return _failed_profile(axis, line_index, "Aggregate profile is too short.")

    keep = _remove_isolated_profile_spikes(heights, config)
    lateral_pixels = lateral_pixels[keep]
    heights = heights[keep]

    lateral_um = (
        lateral_pixels.astype(np.float64)
        - float(np.mean(lateral_pixels.astype(np.float64)))
    ) * config.pixel_pitch_um

    if (
        lateral_pixels.size < config.min_profile_points
        or np.ptp(lateral_um) < config.min_lateral_span_um
        or np.ptp(heights) < config.min_height_span_um
    ):
        return _failed_profile(
            axis,
            line_index,
            "Aggregate profile failed size or variation requirements.",
        )

    circle, fitted = _fit_circle_to_profile(
        lateral_um=lateral_um,
        heights_um=heights,
        config=config,
    )
    return CurvatureProfileResult(
        axis=axis,
        line_index_pixel=line_index,
        lateral_pixels=lateral_pixels,
        lateral_um=lateral_um,
        heights_um=heights,
        fitted_heights_um=fitted,
        circle=circle,
        retained_for_final_radius=circle.success,
    )


def _failed_profile(
    axis: AxisName,
    line_index: int,
    message: str,
) -> CurvatureProfileResult:
    empty_int = np.asarray([], dtype=np.int64)
    empty_float = np.asarray([], dtype=np.float64)
    return CurvatureProfileResult(
        axis=axis,
        line_index_pixel=line_index,
        lateral_pixels=empty_int,
        lateral_um=empty_float,
        heights_um=empty_float,
        fitted_heights_um=empty_float,
        circle=_failed_circle(message),
        retained_for_final_radius=False,
    )


def _select_consistent_profile_indices(
    profiles: Sequence[CurvatureProfileResult],
    mad_multiplier: float,
) -> set[int]:
    if not profiles:
        return set()

    radii = np.asarray([profile.circle.radius_um for profile in profiles])
    median = float(np.median(radii))
    mad = float(np.median(np.abs(radii - median)))

    if mad <= 1e-12:
        return set(range(len(profiles)))

    robust_sigma = 1.4826 * mad
    keep = np.abs(radii - median) <= mad_multiplier * robust_sigma
    return set(np.flatnonzero(keep).astype(int).tolist())


def _circle_quality(
    circle: CircleFitResult,
    config: XpanderCurvatureConfig,
) -> float:
    if not circle.success:
        return 0.0

    rmse_score = float(
        np.exp(
            -circle.normalized_radial_rmse
            / max(config.max_normalized_radial_rmse, 1e-12)
        )
    )
    r_squared_score = float(np.clip(circle.r_squared, 0.0, 1.0))
    return float(0.55 * rmse_score + 0.45 * r_squared_score)


def _mean_profile_quality(
    profiles: Sequence[CurvatureProfileResult],
) -> float:
    if not profiles:
        return 0.0
    qualities = [
        float(
            0.55
            * np.exp(-profile.circle.normalized_radial_rmse / 0.30)
            + 0.45 * np.clip(profile.circle.r_squared, 0.0, 1.0)
        )
        for profile in profiles
    ]
    return float(np.mean(qualities))


def print_xpander_curvature(result: XpanderCurvatureResult) -> None:
    """Print the Stage-7 curvature measurements and diagnostics."""
    print("\nXpander radius-of-curvature measurement:")
    print("-" * 88)
    _print_axis_result(result.x_axis)
    _print_axis_result(result.y_axis)
    print("-" * 88)
    print(f"R_x: {result.radius_x_um:.6f} um")
    print(f"R_y: {result.radius_y_um:.6f} um")
    print(f"Pixel pitch: {result.pixel_pitch_um:.6f} um/pixel")
    print(f"Overall confidence: {result.confidence:.4f}")
    print(f"Confident: {result.is_confident}")
    print("-" * 88)


def _print_axis_result(axis_result: AxisCurvatureResult) -> None:
    print(f"{axis_result.axis}-axis curvature:")
    print(f"  Radius: {axis_result.radius_um:.6f} um")
    print(f"  Radius std: {axis_result.radius_std_um:.6f} um")
    print(f"  Radius MAD: {axis_result.radius_mad_um:.6f} um")
    print(
        "  Radius range: "
        f"[{axis_result.radius_min_um:.6f}, "
        f"{axis_result.radius_max_um:.6f}] um"
    )
    print(
        "  Profiles: "
        f"attempted={axis_result.attempted_profile_count}, "
        f"successful={axis_result.successful_profile_count}, "
        f"retained={axis_result.retained_profile_count}"
    )
    print(
        "  Aggregate fit: "
        f"R={axis_result.aggregate_profile.circle.radius_um:.6f} um, "
        f"RMSE={axis_result.aggregate_profile.circle.profile_rmse_um:.6f} um, "
        f"R^2={axis_result.aggregate_profile.circle.r_squared:.6f}"
    )
    print(f"  Confidence: {axis_result.confidence:.4f}")
    print(f"  Confident: {axis_result.is_confident}")


def plot_xpander_curvature(
    height_map: FloatArray,
    xpander_segmentation: XpanderSegmentationResult,
    result: XpanderCurvatureResult,
    config: XpanderCurvatureConfig | None = None,
) -> None:
    """Display the central fitting bands and the fitted X/Y curvature arcs."""
    if config is None:
        config = XpanderCurvatureConfig(pixel_pitch_um=result.pixel_pitch_um)

    box = xpander_segmentation.bounding_box
    crop_margin_x = max(10, int(round(0.08 * (box.x_max - box.x_min))))
    crop_margin_y = max(10, int(round(0.08 * (box.y_max - box.y_min))))

    height, width = height_map.shape
    crop_x_min = max(0, box.x_min - crop_margin_x)
    crop_x_max = min(width, box.x_max + crop_margin_x)
    crop_y_min = max(0, box.y_min - crop_margin_y)
    crop_y_max = min(height, box.y_max + crop_margin_y)

    figure, axes = plt.subplots(2, 2, figsize=config.figure_size)
    image_axis = axes[0, 0]
    x_axis = axes[0, 1]
    y_axis = axes[1, 0]
    radii_axis = axes[1, 1]

    crop = np.asarray(height_map)[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
    image = image_axis.imshow(crop, cmap="viridis", origin="upper")
    figure.colorbar(image, ax=image_axis, label="Height (um)")

    line_width = config.detection_line_width
    image_axis.add_patch(
        Rectangle(
            (box.x_min - crop_x_min, box.y_min - crop_y_min),
            box.x_max - box.x_min,
            box.y_max - box.y_min,
            fill=False,
            edgecolor="red",
            linewidth=line_width,
            label="Xpander BB",
        )
    )

    x_result = result.x_axis
    y_result = result.y_axis

    image_axis.add_patch(
        Rectangle(
            (
                x_result.fit_start_pixel - crop_x_min,
                x_result.band_start_pixel - crop_y_min,
            ),
            x_result.fit_stop_pixel - x_result.fit_start_pixel,
            x_result.band_stop_pixel - x_result.band_start_pixel,
            fill=False,
            edgecolor="white",
            linewidth=line_width,
            linestyle="--",
            label="X-curvature band",
        )
    )
    image_axis.add_patch(
        Rectangle(
            (
                y_result.band_start_pixel - crop_x_min,
                y_result.fit_start_pixel - crop_y_min,
            ),
            y_result.band_stop_pixel - y_result.band_start_pixel,
            y_result.fit_stop_pixel - y_result.fit_start_pixel,
            fill=False,
            edgecolor="cyan",
            linewidth=line_width,
            linestyle="--",
            label="Y-curvature band",
        )
    )
    image_axis.scatter(
        [result.xpander_center_x_pixel - crop_x_min],
        [result.xpander_center_y_pixel - crop_y_min],
        marker="+",
        s=80,
        linewidths=line_width,
        label="Xpander centre",
    )
    image_axis.set_title("Stage 7 central curvature measurement regions")
    image_axis.set_xlabel("X pixel")
    image_axis.set_ylabel("Y pixel")
    image_axis.legend(loc="best")

    _plot_axis_profile(x_axis, x_result, "X-Z median central profile")
    _plot_axis_profile(y_axis, y_result, "Y-Z median central profile")
    _plot_profile_radii(radii_axis, x_result, y_result)

    figure.tight_layout()
    plt.show()


def get_xpander_curvature(
    height_map: FloatArray,
    xpander_segmentation: XpanderSegmentationResult,
    print_debug: bool = False,
    show_debug: bool = False,
) -> XpanderCurvatureResult:
    """Measure Xpander curvature and optionally display the debug plot."""
    curvature_result = measure_xpander_curvature(
        height_map=height_map,
        xpander_segmentation=xpander_segmentation,
    )

    if print_debug:
        print_xpander_curvature(curvature_result)

    if show_debug:
        plot_xpander_curvature(
            height_map=height_map,
            xpander_segmentation=xpander_segmentation,
            result=curvature_result,
        )

    return curvature_result


def _plot_axis_profile(
    axis: plt.Axes,
    axis_result: AxisCurvatureResult,
    title: str,
) -> None:
    profile = axis_result.aggregate_profile
    if not profile.circle.success:
        axis.text(
            0.5,
            0.5,
            "Aggregate circle fit unavailable",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_title(title)
        return

    order = np.argsort(profile.lateral_um)
    axis.scatter(
        profile.lateral_um,
        profile.heights_um,
        s=12,
        label="Median measured profile",
    )
    axis.plot(
        profile.lateral_um[order],
        profile.fitted_heights_um[order],
        linewidth=1.0,
        label=(
            f"Circle fit: R={profile.circle.radius_um:.3f} um, "
            f"R^2={profile.circle.r_squared:.4f}"
        ),
    )
    axis.set_title(title)
    axis.set_xlabel(f"{axis_result.axis} relative position (um)")
    axis.set_ylabel("Z height (um)")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")


def _plot_profile_radii(
    axis: plt.Axes,
    x_result: AxisCurvatureResult,
    y_result: AxisCurvatureResult,
) -> None:
    for axis_result, marker in ((x_result, "o"), (y_result, "s")):
        profiles = axis_result.profile_results
        if not profiles:
            continue
        line_indices = np.asarray(
            [profile.line_index_pixel for profile in profiles], dtype=float
        )
        radii = np.asarray([profile.circle.radius_um for profile in profiles])
        retained = np.asarray(
            [profile.retained_for_final_radius for profile in profiles], dtype=bool
        )

        axis.scatter(
            line_indices[~retained],
            radii[~retained],
            marker=marker,
            alpha=0.35,
            label=f"{axis_result.axis} rejected",
        )
        axis.scatter(
            line_indices[retained],
            radii[retained],
            marker=marker,
            label=f"{axis_result.axis} retained",
        )
        axis.axhline(
            axis_result.radius_um,
            linewidth=1.0,
            linestyle="--",
            label=f"{axis_result.axis} median R",
        )

    axis.set_title("Central-profile radius consistency")
    axis.set_xlabel("Row/column index (pixel)")
    axis.set_ylabel("Radius (um)")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
