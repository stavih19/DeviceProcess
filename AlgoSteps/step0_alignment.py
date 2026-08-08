from __future__ import annotations

from AlgoSteps.debug_utils import debug_print_context

"""Stage 0: tilt and in-plane rotation analysis/correction for 2-D height maps.

The module is intentionally independent of the seven Task-1 stages.  Its main
entry point is ``analyze_and_correct_alignment``.  Feed the returned
``corrected_height_map`` into the existing pipeline.

Dependencies:
    numpy
    scipy
    matplotlib  # only for the optional diagnostic plot
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import argparse
import json

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi


FloatArray = NDArray[np.floating[Any]]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class AlignmentConfig:
    # Physical calibration. ``height_unit_scale`` converts one input Z unit to
    # the same physical unit used by pixel_size_x/pixel_size_y.
    pixel_size_x: float = 1.0
    pixel_size_y: float = 1.0
    height_unit_scale: float = 1.0

    # Mild filtering used only for estimating masks and edge directions.
    gaussian_sigma: float = 1.0
    local_std_window: int = 9

    # Background sampling for robust plane fitting.
    border_fraction: float = 0.14
    background_gradient_percentile: float = 55.0
    background_local_std_percentile: float = 60.0
    minimum_background_pixels: int = 1500
    maximum_plane_fit_samples: int = 120_000

    # Robust plane fit.
    plane_fit_max_iterations: int = 20
    plane_fit_huber_delta: float = 1.5
    plane_fit_outlier_sigma: float = 3.5
    plane_fit_tolerance: float = 1e-10

    # Decide whether a measured global tilt is large/reliable enough to remove.
    minimum_tilt_degrees: float = 0.03
    minimum_tilt_confidence: float = 0.25
    remove_residual_tilt_after_rotation: bool = True

    # Rotation estimation from coherent, strong local edges.
    edge_magnitude_percentile: float = 88.0
    minimum_edge_coherence: float = 0.55
    structure_tensor_sigma: float = 2.0
    orientation_histogram_bin_degrees: float = 0.25
    orientation_histogram_smoothing_degrees: float = 1.0
    orientation_refine_window_degrees: float = 4.0
    orientation_support_window_degrees: float = 2.0
    minimum_rotation_edge_pixels: int = 500

    # Decide whether rotation correction should be applied.
    minimum_rotation_degrees: float = 0.20
    minimum_rotation_confidence: float = 0.20
    maximum_abs_rotation_degrees: float = 30.0

    # Rotation resampling. Linear interpolation is a good compromise for height
    # maps: less blurring than high-order splines and smoother than nearest.
    rotation_interpolation_order: int = 1
    rotation_reshape: bool = False

    # If lateral pixels are strongly anisotropic, a simple image-grid rotation
    # is not a true physical rotation. Keep this strict unless you explicitly
    # resample the map to isotropic spacing first.
    maximum_pixel_anisotropy_fraction: float = 0.02

    random_seed: int = 17


@dataclass(frozen=True)
class PlaneFitResult:
    # z_physical = slope_x * x_physical + slope_y * y_physical + intercept
    slope_x: float
    slope_y: float
    intercept: float
    rmse_input_units: float
    constant_rmse_input_units: float
    robust_scale_input_units: float
    inlier_fraction: float
    sample_count: int
    confidence: float

    @property
    def tilt_x_degrees(self) -> float:
        return float(np.degrees(np.arctan(self.slope_x)))

    @property
    def tilt_y_degrees(self) -> float:
        return float(np.degrees(np.arctan(self.slope_y)))

    @property
    def tilt_total_degrees(self) -> float:
        return float(
            np.degrees(
                np.arctan(np.hypot(self.slope_x, self.slope_y))
            )
        )

    @property
    def tilt_azimuth_degrees(self) -> float:
        return float(np.degrees(np.arctan2(self.slope_y, self.slope_x)))


@dataclass(frozen=True)
class RotationEstimate:
    angle_degrees: float
    confidence: float
    angle_mad_degrees: float
    concentration: float
    support_fraction: float
    edge_pixel_count: int


@dataclass
class AlignmentResult:
    corrected_height_map: NDArray[np.float32]
    valid_mask: BoolArray
    tilt_corrected_before_rotation: NDArray[np.float32]

    raw_plane: PlaneFitResult
    residual_plane_before_second_correction: PlaneFitResult
    final_plane: PlaneFitResult

    raw_rotation: RotationEstimate
    final_rotation: RotationEstimate

    tilt_applied: bool
    rotation_applied: bool
    residual_tilt_applied: bool

    applied_rotation_degrees: float

    raw_background_mask: BoolArray
    final_background_mask: BoolArray

    # Maps points [x, y, 1] from raw coordinates to corrected coordinates.
    forward_xy_transform: NDArray[np.float64]
    inverse_xy_transform: NDArray[np.float64]

    input_shape: tuple[int, int]
    output_shape: tuple[int, int]

    estimated_tilt_height_change_x: float
    estimated_tilt_height_change_y: float

    @property
    def applied_tilt_x_degrees(self) -> float:
        """Tilt correction applied on X; zero means no X correction."""
        return -self.raw_plane.tilt_x_degrees if self.tilt_applied else 0.0

    @property
    def applied_tilt_y_degrees(self) -> float:
        """Tilt correction applied on Y; zero means no Y correction."""
        return -self.raw_plane.tilt_y_degrees if self.tilt_applied else 0.0

    @property
    def applied_tilt_total_degrees(self) -> float:
        """Magnitude of the primary tilt correction, or zero if unchanged."""
        if not self.tilt_applied:
            return 0.0
        return self.raw_plane.tilt_total_degrees

    @property
    def applied_residual_tilt_degrees(self) -> float:
        """Residual tilt correction magnitude, or zero if unchanged."""
        if not self.residual_tilt_applied:
            return 0.0
        return self.residual_plane_before_second_correction.tilt_total_degrees

    def report_dict(self) -> dict[str, Any]:
        return {
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "tilt_applied": self.tilt_applied,
            "rotation_applied": self.rotation_applied,
            "residual_tilt_applied": self.residual_tilt_applied,
            "applied_tilt_x_degrees": self.applied_tilt_x_degrees,
            "applied_tilt_y_degrees": self.applied_tilt_y_degrees,
            "applied_tilt_total_degrees": self.applied_tilt_total_degrees,
            "applied_residual_tilt_degrees": (
                self.applied_residual_tilt_degrees
            ),
            "measured_rotation_degrees": self.raw_rotation.angle_degrees,
            "applied_rotation_degrees": self.applied_rotation_degrees,
            "residual_rotation_degrees": self.final_rotation.angle_degrees,
            "raw_tilt_x_degrees": self.raw_plane.tilt_x_degrees,
            "raw_tilt_y_degrees": self.raw_plane.tilt_y_degrees,
            "raw_tilt_total_degrees": self.raw_plane.tilt_total_degrees,
            "raw_tilt_azimuth_degrees": self.raw_plane.tilt_azimuth_degrees,
            "residual_tilt_before_second_correction_degrees": (
                self.residual_plane_before_second_correction.tilt_total_degrees
            ),
            "final_residual_tilt_degrees": self.final_plane.tilt_total_degrees,
            "background_constant_rmse_before": (
                self.raw_plane.constant_rmse_input_units
            ),
            "background_plane_rmse_before": self.raw_plane.rmse_input_units,
            "background_constant_rmse_after": (
                self.final_plane.constant_rmse_input_units
            ),
            "background_plane_rmse_after": self.final_plane.rmse_input_units,
            "tilt_confidence": self.raw_plane.confidence,
            "rotation_confidence": self.raw_rotation.confidence,
            "rotation_angle_mad_degrees": self.raw_rotation.angle_mad_degrees,
            "rotation_edge_pixel_count": self.raw_rotation.edge_pixel_count,
            "estimated_tilt_height_change_x": self.estimated_tilt_height_change_x,
            "estimated_tilt_height_change_y": self.estimated_tilt_height_change_y,
            "forward_xy_transform": self.forward_xy_transform.tolist(),
            "inverse_xy_transform": self.inverse_xy_transform.tolist(),
        }


def get_alignment_correction(
    height_map: FloatArray,
    pixel_size_um: float = 0.252,
    print_debug: bool = False,
    show_debug: bool = False,
    config: AlignmentConfig | None = None,
) -> AlignmentResult:
    if config is None:
        if pixel_size_um <= 0:
            raise ValueError(
                f"pixel_size_um must be positive, received {pixel_size_um}."
            )

        config = AlignmentConfig(
            pixel_size_x=float(pixel_size_um),
            pixel_size_y=float(pixel_size_um),
            height_unit_scale=1.0,
        )

    with debug_print_context(print_debug):
        result = analyze_and_correct_alignment(
            height_map=height_map,
            config=config,
        )

    if print_debug:
        print_alignment_report(result)

    if show_debug:
        plot_alignment_diagnostics(
            raw_height_map=height_map,
            result=result,
            show=True,
        )

    return result


def analyze_and_correct_alignment(
    height_map: FloatArray,
    config: AlignmentConfig | None = None,
) -> AlignmentResult:
    """Analyze and correct global Z tilt and XY rotation.

    Processing order:
        1. Build a likely-background mask.
        2. Robustly fit z = ax + by + c.
        3. Remove only the plane slope, keeping the central height unchanged.
        4. Estimate dominant deviation from the nearest horizontal/vertical axis.
        5. Rotate by the opposite angle.
        6. Fit/remove a small residual plane and validate residual alignment.
    """
    if config is None:
        config = AlignmentConfig()

    _validate_input(height_map, config)

    raw = np.asarray(height_map, dtype=np.float32)
    raw_valid = np.isfinite(raw)
    filled_raw = _fill_invalid(raw, raw_valid)

    raw_background_mask = _build_background_mask(
        filled_raw,
        raw_valid,
        config,
    )
    raw_plane = _fit_plane_robust(
        filled_raw,
        raw_background_mask,
        config,
    )

    tilt_applied = bool(
        raw_plane.tilt_total_degrees >= config.minimum_tilt_degrees
        and raw_plane.confidence >= config.minimum_tilt_confidence
    )

    if tilt_applied:
        untilted = _remove_plane_slope(filled_raw, raw_plane, config)
    else:
        untilted = filled_raw.copy()

    raw_rotation = _estimate_rotation(untilted, raw_valid, config)
    rotation_applied = bool(
        abs(raw_rotation.angle_degrees) >= config.minimum_rotation_degrees
        and raw_rotation.confidence >= config.minimum_rotation_confidence
        and abs(raw_rotation.angle_degrees)
        <= config.maximum_abs_rotation_degrees
    )

    if rotation_applied:
        _validate_rotation_pixel_spacing(config)
        applied_rotation_degrees = raw_rotation.angle_degrees
        rotated, rotated_valid = _rotate_height_map(
            untilted,
            raw_valid,
            applied_rotation_degrees,
            config,
        )
        forward_transform = _rotation_transform_matrix(
            input_shape=raw.shape,
            output_shape=rotated.shape,
            angle_degrees=applied_rotation_degrees,
        )
    else:
        applied_rotation_degrees = 0.0
        rotated = untilted.copy()
        rotated_valid = raw_valid.copy()
        forward_transform = np.eye(3, dtype=np.float64)

    inverse_transform = np.linalg.inv(forward_transform)

    residual_background_mask = _build_background_mask(
        rotated,
        rotated_valid,
        config,
    )
    residual_plane = _fit_plane_robust(
        rotated,
        residual_background_mask,
        config,
    )

    residual_tilt_applied = bool(
        config.remove_residual_tilt_after_rotation
        and residual_plane.confidence >= config.minimum_tilt_confidence
        and residual_plane.tilt_total_degrees >= config.minimum_tilt_degrees
    )

    if residual_tilt_applied:
        corrected = _remove_plane_slope(rotated, residual_plane, config)
    else:
        corrected = rotated.copy()

    # Keep artificial rotation-canvas pixels at a neutral background height.
    final_background_value = float(
        np.median(corrected[residual_background_mask])
    )
    corrected = corrected.astype(np.float32, copy=False)
    corrected[~rotated_valid] = final_background_value

    final_background_mask = _build_background_mask(
        corrected,
        rotated_valid,
        config,
    )
    final_plane = _fit_plane_robust(
        corrected,
        final_background_mask,
        config,
    )
    final_rotation = _estimate_rotation(corrected, rotated_valid, config)

    physical_width = max(raw.shape[1] - 1, 0) * config.pixel_size_x
    physical_height = max(raw.shape[0] - 1, 0) * config.pixel_size_y
    height_scale = config.height_unit_scale

    estimated_change_x = float(
        raw_plane.slope_x * physical_width / height_scale
    )
    estimated_change_y = float(
        raw_plane.slope_y * physical_height / height_scale
    )

    return AlignmentResult(
        corrected_height_map=corrected,
        valid_mask=rotated_valid,
        tilt_corrected_before_rotation=untilted.astype(np.float32),
        raw_plane=raw_plane,
        residual_plane_before_second_correction=residual_plane,
        final_plane=final_plane,
        raw_rotation=raw_rotation,
        final_rotation=final_rotation,
        tilt_applied=tilt_applied,
        rotation_applied=rotation_applied,
        residual_tilt_applied=residual_tilt_applied,
        applied_rotation_degrees=float(applied_rotation_degrees),
        raw_background_mask=raw_background_mask,
        final_background_mask=final_background_mask,
        forward_xy_transform=forward_transform,
        inverse_xy_transform=inverse_transform,
        input_shape=tuple(int(v) for v in raw.shape),
        output_shape=tuple(int(v) for v in corrected.shape),
        estimated_tilt_height_change_x=estimated_change_x,
        estimated_tilt_height_change_y=estimated_change_y,
    )


def transform_points(
    points_xy: NDArray[np.floating[Any]],
    transform: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Apply a homogeneous 3x3 transform to an array shaped (N, 2)."""
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_xy must have shape (N, 2).")
    homogeneous = np.column_stack(
        [points, np.ones(points.shape[0], dtype=np.float64)]
    )
    transformed = homogeneous @ transform.T
    return transformed[:, :2] / transformed[:, 2:3]


def print_alignment_report(result: AlignmentResult) -> None:
    report = result.report_dict()
    print("\n" + "=" * 76)
    print("STAGE 0 ALIGNMENT REPORT")
    print("=" * 76)
    print(
        "Raw tilt: "
        f"X={report['raw_tilt_x_degrees']:.6f} deg, "
        f"Y={report['raw_tilt_y_degrees']:.6f} deg, "
        f"total={report['raw_tilt_total_degrees']:.6f} deg, "
        f"azimuth={report['raw_tilt_azimuth_degrees']:.3f} deg"
    )
    print(
        "Rotation: "
        f"measured={report['measured_rotation_degrees']:.4f} deg, "
        f"applied={report['applied_rotation_degrees']:.4f} deg, "
        f"confidence={report['rotation_confidence']:.3f}, "
        f"MAD={report['rotation_angle_mad_degrees']:.3f} deg"
    )
    print(
        "Residuals: "
        f"tilt={report['final_residual_tilt_degrees']:.6f} deg, "
        f"rotation={report['residual_rotation_degrees']:.4f} deg"
    )
    print(
        "Background constant-RMSE: "
        f"before={report['background_constant_rmse_before']:.6f}, "
        f"after={report['background_constant_rmse_after']:.6f}"
    )
    print(
        "Estimated raw tilt height change: "
        f"across X={report['estimated_tilt_height_change_x']:.6f}, "
        f"across Y={report['estimated_tilt_height_change_y']:.6f}"
    )
    print(
        "Corrections: "
        f"tilt={result.tilt_applied}, "
        f"rotation={result.rotation_applied}, "
        f"residual_tilt={result.residual_tilt_applied}"
    )
    print(f"Shape: {result.input_shape} -> {result.output_shape}")
    print("-" * 76)
    print("STAGE 0 APPLIED-CORRECTION SUMMARY")
    print(
        "Tilt correction: "
        f"X={result.applied_tilt_x_degrees:.6f} deg, "
        f"Y={result.applied_tilt_y_degrees:.6f} deg, "
        f"total={result.applied_tilt_total_degrees:.6f} deg"
    )
    print(
        "Residual tilt correction: "
        f"{result.applied_residual_tilt_degrees:.6f} deg"
    )
    print(
        "Rotation correction: "
        f"{result.applied_rotation_degrees:.6f} deg"
    )
    print(
        "A displayed correction of 0.000000 deg means that axis was not changed."
    )
    print("=" * 76 + "\n")


def plot_alignment_diagnostics(
    raw_height_map: FloatArray,
    result: AlignmentResult,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot raw/corrected maps and the masks used for plane fitting."""
    import matplotlib.pyplot as plt

    raw = np.asarray(raw_height_map)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    image = axes[0, 0].imshow(raw)
    axes[0, 0].set_title("Raw height map")
    fig.colorbar(image, ax=axes[0, 0], fraction=0.046)

    image = axes[0, 1].imshow(result.tilt_corrected_before_rotation)
    axes[0, 1].set_title("After global tilt removal")
    fig.colorbar(image, ax=axes[0, 1], fraction=0.046)

    image = axes[1, 0].imshow(result.corrected_height_map)
    axes[1, 0].set_title(
        "Final corrected map\n"
        f"rotation={result.raw_rotation.angle_degrees:.3f} deg, "
        f"residual={result.final_rotation.angle_degrees:.3f} deg"
    )
    fig.colorbar(image, ax=axes[1, 0], fraction=0.046)

    axes[1, 1].imshow(result.corrected_height_map)
    mask_overlay = np.ma.masked_where(
        ~result.final_background_mask,
        result.final_background_mask,
    )
    axes[1, 1].imshow(mask_overlay, alpha=0.5)
    axes[1, 1].set_title("Final background pixels used for plane fit")

    for axis in axes.ravel():
        axis.set_xlabel("X [pixel]")
        axis.set_ylabel("Y [pixel]")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _validate_input(height_map: FloatArray, config: AlignmentConfig) -> None:
    array = np.asarray(height_map)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D height map, got shape {array.shape}.")
    if min(array.shape) < 32:
        raise ValueError("The height map is too small for reliable alignment.")
    if np.count_nonzero(np.isfinite(array)) < 0.8 * array.size:
        raise ValueError("At least 80% of height-map values must be finite.")
    if config.pixel_size_x <= 0 or config.pixel_size_y <= 0:
        raise ValueError("Pixel sizes must be positive.")
    if config.height_unit_scale <= 0:
        raise ValueError("height_unit_scale must be positive.")
    if not 0 < config.border_fraction < 0.5:
        raise ValueError("border_fraction must be between 0 and 0.5.")
    if config.local_std_window < 3:
        raise ValueError("local_std_window must be at least 3.")
    if not 0 <= config.rotation_interpolation_order <= 5:
        raise ValueError("rotation_interpolation_order must be in [0, 5].")


def _fill_invalid(height_map: NDArray[np.float32], valid: BoolArray) -> NDArray[np.float32]:
    if np.all(valid):
        return height_map.astype(np.float32, copy=True)
    fill_value = float(np.median(height_map[valid]))
    output = height_map.astype(np.float32, copy=True)
    output[~valid] = fill_value
    return output


def _gradient_and_local_std(
    height_map: NDArray[np.float32],
    config: AlignmentConfig,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    smoothed = ndi.gaussian_filter(
        height_map,
        sigma=config.gaussian_sigma,
        mode="reflect",
    ).astype(np.float32)
    gradient_x = (ndi.sobel(smoothed, axis=1, mode="reflect") / 8.0).astype(
        np.float32
    )
    gradient_y = (ndi.sobel(smoothed, axis=0, mode="reflect") / 8.0).astype(
        np.float32
    )
    magnitude = np.hypot(gradient_x, gradient_y).astype(np.float32)

    window = int(config.local_std_window)
    if window % 2 == 0:
        window += 1
    mean = ndi.uniform_filter(smoothed, size=window, mode="reflect")
    mean_square = ndi.uniform_filter(smoothed * smoothed, size=window, mode="reflect")
    variance = np.maximum(mean_square - mean * mean, 0.0)
    local_std = np.sqrt(variance).astype(np.float32)
    return magnitude, local_std


def _build_background_mask(
    height_map: NDArray[np.float32],
    valid_mask: BoolArray,
    config: AlignmentConfig,
) -> BoolArray:
    height, width = height_map.shape
    border_y = max(2, int(round(height * config.border_fraction)))
    border_x = max(2, int(round(width * config.border_fraction)))

    border = np.zeros_like(valid_mask, dtype=bool)
    border[:border_y, :] = True
    border[-border_y:, :] = True
    border[:, :border_x] = True
    border[:, -border_x:] = True
    border &= valid_mask

    magnitude, local_std = _gradient_and_local_std(height_map, config)
    reference_mask = border if np.count_nonzero(border) > 50 else valid_mask

    gradient_threshold = float(
        np.percentile(
            magnitude[reference_mask],
            config.background_gradient_percentile,
        )
    )
    std_threshold = float(
        np.percentile(
            local_std[reference_mask],
            config.background_local_std_percentile,
        )
    )

    candidate = (
        border
        & (magnitude <= gradient_threshold)
        & (local_std <= std_threshold)
    )

    # Fallback: retain the same low-detail conditions across the image. The
    # robust plane fit will reject flat component surfaces that do not follow
    # the dominant background plane.
    if np.count_nonzero(candidate) < config.minimum_background_pixels:
        candidate = (
            valid_mask
            & (magnitude <= gradient_threshold)
            & (local_std <= std_threshold)
        )

    if np.count_nonzero(candidate) < 30:
        raise ValueError("Could not find enough likely-background pixels.")

    return candidate


def _fit_plane_robust(
    height_map: NDArray[np.float32],
    mask: BoolArray,
    config: AlignmentConfig,
) -> PlaneFitResult:
    y_pixels, x_pixels = np.nonzero(mask)
    z_input = height_map[mask].astype(np.float64)

    if z_input.size < 3:
        raise ValueError("At least three points are required for plane fitting.")

    if z_input.size > config.maximum_plane_fit_samples:
        rng = np.random.default_rng(config.random_seed)
        selected = rng.choice(
            z_input.size,
            size=config.maximum_plane_fit_samples,
            replace=False,
        )
        x_pixels = x_pixels[selected]
        y_pixels = y_pixels[selected]
        z_input = z_input[selected]

    x_physical = x_pixels.astype(np.float64) * config.pixel_size_x
    y_physical = y_pixels.astype(np.float64) * config.pixel_size_y
    z_physical = z_input * config.height_unit_scale

    design = np.column_stack(
        [x_physical, y_physical, np.ones_like(x_physical)]
    )
    beta, *_ = np.linalg.lstsq(design, z_physical, rcond=None)

    weights = np.ones_like(z_physical)
    robust_scale_physical = 0.0

    for _ in range(config.plane_fit_max_iterations):
        residuals = z_physical - design @ beta
        residual_center = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - residual_center)))
        robust_scale_physical = max(1.4826 * mad, 1e-12)
        cutoff = config.plane_fit_huber_delta * robust_scale_physical
        absolute = np.abs(residuals - residual_center)
        new_weights = np.ones_like(weights)
        outside = absolute > cutoff
        new_weights[outside] = cutoff / np.maximum(absolute[outside], 1e-12)

        weighted_design = design * np.sqrt(new_weights)[:, None]
        weighted_z = z_physical * np.sqrt(new_weights)
        new_beta, *_ = np.linalg.lstsq(weighted_design, weighted_z, rcond=None)

        if np.linalg.norm(new_beta - beta) <= config.plane_fit_tolerance:
            beta = new_beta
            weights = new_weights
            break
        beta = new_beta
        weights = new_weights

    residuals = z_physical - design @ beta
    residual_center = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - residual_center)))
    robust_scale_physical = max(1.4826 * mad, 1e-12)
    inliers = (
        np.abs(residuals - residual_center)
        <= config.plane_fit_outlier_sigma * robust_scale_physical
    )

    if np.count_nonzero(inliers) >= 3:
        beta, *_ = np.linalg.lstsq(
            design[inliers],
            z_physical[inliers],
            rcond=None,
        )
        residuals = z_physical - design @ beta

    rmse_physical = float(np.sqrt(np.mean(residuals[inliers] ** 2)))
    constant_residuals = z_physical - np.median(z_physical)
    constant_rmse_physical = float(np.sqrt(np.mean(constant_residuals**2)))

    inlier_fraction = float(np.mean(inliers))
    dynamic_range = max(
        float(np.percentile(z_physical, 95) - np.percentile(z_physical, 5)),
        1e-12,
    )
    noise_score = float(np.exp(-rmse_physical / dynamic_range))
    confidence = float(np.clip(inlier_fraction * noise_score, 0.0, 1.0))

    return PlaneFitResult(
        slope_x=float(beta[0]),
        slope_y=float(beta[1]),
        intercept=float(beta[2]),
        rmse_input_units=rmse_physical / config.height_unit_scale,
        constant_rmse_input_units=(
            constant_rmse_physical / config.height_unit_scale
        ),
        robust_scale_input_units=(
            robust_scale_physical / config.height_unit_scale
        ),
        inlier_fraction=inlier_fraction,
        sample_count=int(z_input.size),
        confidence=confidence,
    )


def _remove_plane_slope(
    height_map: NDArray[np.float32],
    plane: PlaneFitResult,
    config: AlignmentConfig,
) -> NDArray[np.float32]:
    height, width = height_map.shape
    x = np.arange(width, dtype=np.float64) * config.pixel_size_x
    y = np.arange(height, dtype=np.float64) * config.pixel_size_y
    x_center = 0.5 * (x[0] + x[-1])
    y_center = 0.5 * (y[0] + y[-1])

    slope_surface_physical = (
        plane.slope_x * (x[None, :] - x_center)
        + plane.slope_y * (y[:, None] - y_center)
    )
    corrected = (
        height_map.astype(np.float64)
        - slope_surface_physical / config.height_unit_scale
    )
    return corrected.astype(np.float32)


def _estimate_rotation(
    height_map: NDArray[np.float32],
    valid_mask: BoolArray,
    config: AlignmentConfig,
) -> RotationEstimate:
    smoothed = ndi.gaussian_filter(
        height_map,
        sigma=config.gaussian_sigma,
        mode="reflect",
    ).astype(np.float64)

    gradient_x = ndi.sobel(smoothed, axis=1, mode="reflect") / 8.0
    gradient_y = ndi.sobel(smoothed, axis=0, mode="reflect") / 8.0
    magnitude = np.hypot(gradient_x, gradient_y)

    tensor_xx = ndi.gaussian_filter(
        gradient_x * gradient_x,
        sigma=config.structure_tensor_sigma,
        mode="reflect",
    )
    tensor_yy = ndi.gaussian_filter(
        gradient_y * gradient_y,
        sigma=config.structure_tensor_sigma,
        mode="reflect",
    )
    tensor_xy = ndi.gaussian_filter(
        gradient_x * gradient_y,
        sigma=config.structure_tensor_sigma,
        mode="reflect",
    )
    coherence = np.sqrt(
        (tensor_xx - tensor_yy) ** 2 + 4.0 * tensor_xy**2
    ) / np.maximum(tensor_xx + tensor_yy, 1e-12)

    core = valid_mask.copy()
    border = max(2, int(round(min(height_map.shape) * 0.01)))
    core[:border, :] = False
    core[-border:, :] = False
    core[:, :border] = False
    core[:, -border:] = False

    if np.count_nonzero(core) < 10:
        return RotationEstimate(0.0, 0.0, 45.0, 0.0, 0.0, 0)

    magnitude_threshold = float(
        np.percentile(magnitude[core], config.edge_magnitude_percentile)
    )
    selected = (
        core
        & (magnitude >= magnitude_threshold)
        & (coherence >= config.minimum_edge_coherence)
    )

    if np.count_nonzero(selected) < config.minimum_rotation_edge_pixels:
        relaxed_threshold = float(np.percentile(magnitude[core], 78.0))
        selected = (
            core
            & (magnitude >= relaxed_threshold)
            & (coherence >= max(0.35, config.minimum_edge_coherence - 0.15))
        )

    edge_count = int(np.count_nonzero(selected))
    if edge_count < 20:
        return RotationEstimate(0.0, 0.0, 45.0, 0.0, 0.0, edge_count)

    # Gradient angle is normal to an edge. Adding 90 degrees gives the local
    # edge tangent. Axis-aligned directions are periodic every 90 degrees.
    tangent_degrees = np.degrees(
        np.arctan2(gradient_y[selected], gradient_x[selected])
    ) + 90.0
    deviations = _normalize_axis_angle(tangent_degrees)
    weights = (
        magnitude[selected]
        * np.maximum(coherence[selected], 1e-6) ** 2
    ).astype(np.float64)

    bin_size = config.orientation_histogram_bin_degrees
    bin_edges = np.arange(-45.0, 45.0 + bin_size, bin_size)
    histogram, _ = np.histogram(deviations, bins=bin_edges, weights=weights)
    smoothing_sigma_bins = max(
        config.orientation_histogram_smoothing_degrees / bin_size,
        0.0,
    )
    histogram = ndi.gaussian_filter1d(
        histogram.astype(np.float64),
        sigma=smoothing_sigma_bins,
        mode="wrap",
    )
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    peak_angle = float(centers[int(np.argmax(histogram))])

    peak_delta = _normalize_axis_angle(deviations - peak_angle)
    refine = np.abs(peak_delta) <= config.orientation_refine_window_degrees
    if np.count_nonzero(refine) >= 5:
        correction = _weighted_median(peak_delta[refine], weights[refine])
        angle = float(_normalize_axis_angle(np.array([peak_angle + correction]))[0])
    else:
        angle = peak_angle

    residuals = _normalize_axis_angle(deviations - angle)
    angle_mad = _weighted_median(np.abs(residuals), weights)
    support = np.abs(residuals) <= config.orientation_support_window_degrees
    support_fraction = float(
        np.sum(weights[support]) / max(np.sum(weights), 1e-12)
    )

    # Period 90 degrees -> multiply angles by four for a normal circular mean.
    phase = np.deg2rad(4.0 * deviations)
    concentration = float(
        np.abs(np.sum(weights * np.exp(1j * phase)))
        / max(np.sum(weights), 1e-12)
    )
    spread_score = float(np.exp(-angle_mad / 4.0))
    confidence = float(
        np.clip(
            0.45 * concentration
            + 0.35 * support_fraction
            + 0.20 * spread_score,
            0.0,
            1.0,
        )
    )

    return RotationEstimate(
        angle_degrees=angle,
        confidence=confidence,
        angle_mad_degrees=float(angle_mad),
        concentration=concentration,
        support_fraction=support_fraction,
        edge_pixel_count=edge_count,
    )


def _normalize_axis_angle(angle_degrees: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
    angle = np.asarray(angle_degrees, dtype=np.float64)
    return ((angle + 45.0) % 90.0) - 45.0


def _weighted_median(values: NDArray[np.float64], weights: NDArray[np.float64]) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cannot calculate a weighted median of an empty array.")
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = np.maximum(weights[order], 0.0)
    total = float(np.sum(sorted_weights))
    if total <= 0:
        return float(np.median(values))
    cumulative = np.cumsum(sorted_weights)
    index = int(np.searchsorted(cumulative, 0.5 * total, side="left"))
    return float(sorted_values[min(index, sorted_values.size - 1)])


def _validate_rotation_pixel_spacing(config: AlignmentConfig) -> None:
    anisotropy = abs(config.pixel_size_x - config.pixel_size_y) / max(
        config.pixel_size_x,
        config.pixel_size_y,
    )
    if anisotropy > config.maximum_pixel_anisotropy_fraction:
        raise ValueError(
            "The lateral pixels are anisotropic. Resample to isotropic XY "
            "spacing before using image-grid rotation, or increase "
            "maximum_pixel_anisotropy_fraction deliberately."
        )


def _rotate_height_map(
    height_map: NDArray[np.float32],
    valid_mask: BoolArray,
    angle_degrees: float,
    config: AlignmentConfig,
) -> tuple[NDArray[np.float32], BoolArray]:
    background_value = float(np.median(height_map[valid_mask]))
    rotated = ndi.rotate(
        height_map,
        angle=angle_degrees,
        axes=(1, 0),
        reshape=config.rotation_reshape,
        order=config.rotation_interpolation_order,
        mode="constant",
        cval=background_value,
        prefilter=config.rotation_interpolation_order > 1,
    ).astype(np.float32)

    rotated_valid = ndi.rotate(
        valid_mask.astype(np.uint8),
        angle=angle_degrees,
        axes=(1, 0),
        reshape=config.rotation_reshape,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    ) > 0
    return rotated, rotated_valid


def _rotation_transform_matrix(
    input_shape: tuple[int, int],
    output_shape: tuple[int, int],
    angle_degrees: float,
) -> NDArray[np.float64]:
    input_height, input_width = input_shape
    output_height, output_width = output_shape

    angle = np.deg2rad(angle_degrees)
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    # Image coordinates use X to the right and Y downward. This matrix
    # matches scipy.ndimage.rotate's visual-positive angle convention.
    rotation = np.array(
        [
            [cosine, sine],
            [-sine, cosine],
        ],
        dtype=np.float64,
    )

    input_center = np.array(
        [(input_width - 1) / 2.0, (input_height - 1) / 2.0],
        dtype=np.float64,
    )
    output_center = np.array(
        [(output_width - 1) / 2.0, (output_height - 1) / 2.0],
        dtype=np.float64,
    )
    translation = output_center - rotation @ input_center

    transform = np.eye(3, dtype=np.float64)
    transform[:2, :2] = rotation
    transform[:2, 2] = translation
    return transform


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze and correct tilt/rotation in a 2-D NPY height map."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--pixel-size-x", type=float, default=1.0)
    parser.add_argument("--pixel-size-y", type=float, default=1.0)
    parser.add_argument("--height-unit-scale", type=float, default=1.0)
    arguments = parser.parse_args()

    height_map = np.load(arguments.input)
    config = AlignmentConfig(
        pixel_size_x=arguments.pixel_size_x,
        pixel_size_y=arguments.pixel_size_y,
        height_unit_scale=arguments.height_unit_scale,
    )
    result = analyze_and_correct_alignment(height_map, config)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(arguments.output, result.corrected_height_map)
    print_alignment_report(result)

    if arguments.report is not None:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        with arguments.report.open("w", encoding="utf-8") as file:
            json.dump(result.report_dict(), file, indent=2)

    if arguments.plot is not None:
        arguments.plot.parent.mkdir(parents=True, exist_ok=True)
        plot_alignment_diagnostics(
            height_map,
            result,
            save_path=arguments.plot,
            show=False,
        )


if __name__ == "__main__":
    _main()
