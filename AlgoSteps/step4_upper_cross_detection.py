from __future__ import annotations

from AlgoSteps.debug_utils import debug_print_context

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from numpy.typing import NDArray
from scipy import ndimage as ndi

from AlgoSteps.step2_lower_cross_detection import LowerCrossDetectionResult
from AlgoSteps.step1_pivot_candidates import (
    BoolArray,
    BoundingBox,
    FloatArray,
    IntArray,
    LowerPlaneDetectionResult,
)
from AlgoSteps.step3_pivot_segmentation import PivotSegmentationResult


@dataclass(frozen=True)
class UpperCrossDetectionConfig:
    """
    Configuration for Stage 4: detect the upper Pivot cross.

    The search is restricted to the already segmented Pivot and to the region
    above the detected lower plane. No inner bounding box is stored.
    """

    # Difference-of-Gaussians parameters.
    small_gaussian_sigma: float = 0.8
    large_gaussian_sigma: float = 8.0

    # Robust threshold:
    # median(response) + multiplier * robust_sigma
    threshold_mad_multiplier: float = 3.0

    # Morphological cleanup.
    opening_size: int = 1
    closing_size: int = 3

    # Exclude the outer Pivot walls and the transition near the lower plane.
    pivot_edge_margin_fraction: float = 0.04
    min_pivot_edge_margin_pixels: int = 3
    lower_plane_gap_fraction: float = 0.08
    min_lower_plane_gap_pixels: int = 3

    # Candidate size relative to the temporary upper search mask.
    min_area_fraction: float = 0.0005
    max_area_fraction: float = 0.08
    min_width_pixels: int = 3
    min_height_pixels: int = 3

    # Fallback expected area when the lower cross is unavailable.
    expected_area_fraction: float = 0.015
    area_tolerance_factor: float = 3.0

    # Alignment with the lower cross / Pivot centre.
    horizontal_tolerance_fraction: float = 0.15
    vertical_tolerance_fraction: float = 0.45

    # Similarity to the already detected lower cross.
    lower_cross_size_tolerance_factor: float = 2.5

    # Candidate-score weights.
    shape_weight: float = 0.35
    horizontal_alignment_weight: float = 0.25
    vertical_position_weight: float = 0.10
    size_weight: float = 0.15
    contrast_weight: float = 0.15

    minimum_score: float = 0.45


@dataclass(frozen=True)
class UpperCrossCandidate:
    """Measurements describing one possible upper Pivot cross."""

    component_label: int

    # Coordinates in the original height map.
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
    horizontal_alignment_score: float
    vertical_position_score: float
    size_score: float
    contrast_score: float
    score: float


@dataclass
class UpperCrossDetectionResult:
    """
    Stage-4 result.

    response_map, search_mask, threshold_mask and label_image are stored in
    coordinates local to the Pivot crop.
    """

    pivot_bounding_box: BoundingBox

    crop_origin_x: int
    crop_origin_y: int
    crop_shape: tuple[int, int]

    upper_search_limit_y_global: int
    threshold: float
    robust_response_sigma: float

    response_map: NDArray[np.float32]
    search_mask: BoolArray
    threshold_mask: BoolArray
    label_image: IntArray

    config: UpperCrossDetectionConfig
    candidates: list[UpperCrossCandidate] = field(default_factory=list)

    @property
    def best_candidate(self) -> UpperCrossCandidate | None:
        if not self.candidates:
            return None
        return self.candidates[0]

    @property
    def is_confident(self) -> bool:
        candidate = self.best_candidate
        return (
            candidate is not None
            and candidate.score >= self.config.minimum_score
        )

    def get_candidate_mask_local(
        self,
        candidate: UpperCrossCandidate,
    ) -> BoolArray:
        return self.label_image == candidate.component_label

    def get_candidate_mask_global(
        self,
        candidate: UpperCrossCandidate,
        image_shape: tuple[int, int],
    ) -> BoolArray:
        global_mask = np.zeros(image_shape, dtype=bool)
        local_mask = self.get_candidate_mask_local(candidate)

        crop_height, crop_width = self.crop_shape

        global_mask[
            self.crop_origin_y:self.crop_origin_y + crop_height,
            self.crop_origin_x:self.crop_origin_x + crop_width,
        ] = local_mask

        return global_mask


def find_upper_cross_candidates(
    height_map: FloatArray,
    pivot_segmentation: PivotSegmentationResult,
    lower_plane_detection: LowerPlaneDetectionResult,
    lower_cross_detection: LowerCrossDetectionResult | None = None,
    config: UpperCrossDetectionConfig | None = None,
) -> UpperCrossDetectionResult:
    """
    Detect and rank upper-cross candidates inside the segmented Pivot.

    Search logic:
    1. Crop the complete Pivot.
    2. Restrict the search to pixels inside the Pivot mask.
    3. Exclude the Pivot boundary and everything at/under the lower plane.
    4. Detect local depressions using Difference of Gaussians.
    5. Rank candidates by plus-sign shape, alignment with the lower cross,
       vertical position, size similarity and local contrast.
    """
    if config is None:
        config = UpperCrossDetectionConfig()

    _validate_inputs(
        height_map=height_map,
        pivot_segmentation=pivot_segmentation,
        lower_plane_detection=lower_plane_detection,
        config=config,
    )

    lower_plane = lower_plane_detection.best_candidate
    if lower_plane is None:
        raise ValueError(
            "Upper-cross detection requires a valid lower-plane candidate."
        )

    pivot_box = pivot_segmentation.bounding_box
    lower_plane_box = lower_plane.bounding_box

    height_map = height_map.astype(np.float32, copy=False)

    pivot_crop = height_map[
        pivot_box.y_min:pivot_box.y_max,
        pivot_box.x_min:pivot_box.x_max,
    ]

    pivot_mask_local = pivot_segmentation.pivot_mask[
        pivot_box.y_min:pivot_box.y_max,
        pivot_box.x_min:pivot_box.x_max,
    ]

    lower_cross = (
        lower_cross_detection.best_candidate
        if lower_cross_detection is not None
        else None
    )

    search_mask, upper_search_limit_y_global = _create_upper_search_mask(
        pivot_mask_local=pivot_mask_local,
        pivot_box=pivot_box,
        lower_plane_box=lower_plane_box,
        config=config,
    )

    if not np.any(search_mask):
        raise ValueError(
            "No valid upper-Pivot area remains after applying the search mask."
        )

    small_scale = ndi.gaussian_filter(
        pivot_crop,
        sigma=config.small_gaussian_sigma,
    )
    large_scale = ndi.gaussian_filter(
        pivot_crop,
        sigma=config.large_gaussian_sigma,
    )

    # According to the supplied examples, the cross is a local depression.
    response_map = (large_scale - small_scale).astype(np.float32)

    response_values = response_map[search_mask]
    response_median = float(np.median(response_values))
    response_mad = float(
        np.median(
            np.abs(response_values - response_median)
        )
    )
    robust_response_sigma = max(
        1.4826 * response_mad,
        1e-8,
    )

    threshold = float(
        response_median
        + config.threshold_mad_multiplier
        * robust_response_sigma
    )

    threshold_mask = (
        (response_map >= threshold)
        & search_mask
    )

    if config.opening_size > 1:
        threshold_mask = ndi.binary_opening(
            threshold_mask,
            structure=np.ones(
                (
                    config.opening_size,
                    config.opening_size,
                ),
                dtype=bool,
            ),
        )

    if config.closing_size > 1:
        threshold_mask = ndi.binary_closing(
            threshold_mask,
            structure=np.ones(
                (
                    config.closing_size,
                    config.closing_size,
                ),
                dtype=bool,
            ),
        )

    label_image, component_count = ndi.label(
        threshold_mask
    )

    search_area = max(
        1,
        int(np.count_nonzero(search_mask)),
    )

    min_area = max(
        1,
        int(
            round(
                search_area
                * config.min_area_fraction
            )
        ),
    )
    max_area = max(
        min_area,
        int(
            round(
                search_area
                * config.max_area_fraction
            )
        ),
    )

    if lower_cross is not None:
        expected_x_global = lower_cross.center_x
        expected_lower_area = lower_cross.area_pixels
        expected_lower_width = (
            lower_cross.bounding_box.width
        )
        expected_lower_height = (
            lower_cross.bounding_box.height
        )
    else:
        expected_x_global = (
            pivot_box.x_min
            + (pivot_box.width - 1) / 2
        )
        expected_lower_area = None
        expected_lower_width = None
        expected_lower_height = None

    expected_x_local = (
        expected_x_global - pivot_box.x_min
    )

    valid_y, _ = np.nonzero(search_mask)
    expected_y_local = float(
        (
            valid_y.min()
            + valid_y.max()
        )
        / 2
    )

    candidates: list[UpperCrossCandidate] = []
    component_slices = ndi.find_objects(
        label_image
    )

    for component_label in range(
        1,
        component_count + 1,
    ):
        component_slice = component_slices[
            component_label - 1
        ]

        if component_slice is None:
            continue

        candidate_mask_local = (
            label_image == component_label
        )
        area_pixels = int(
            np.count_nonzero(
                candidate_mask_local
            )
        )

        if (
            area_pixels < min_area
            or area_pixels > max_area
        ):
            continue

        y_slice, x_slice = component_slice

        local_box = BoundingBox(
            x_min=x_slice.start,
            y_min=y_slice.start,
            x_max=x_slice.stop,
            y_max=y_slice.stop,
        )

        if (
            local_box.width
            < config.min_width_pixels
            or local_box.height
            < config.min_height_pixels
        ):
            continue

        local_y, local_x = np.nonzero(
            candidate_mask_local
        )

        center_x_local = float(
            np.mean(local_x)
        )
        center_y_local = float(
            np.mean(local_y)
        )

        global_box = BoundingBox(
            x_min=(
                pivot_box.x_min
                + local_box.x_min
            ),
            y_min=(
                pivot_box.y_min
                + local_box.y_min
            ),
            x_max=(
                pivot_box.x_min
                + local_box.x_max
            ),
            y_max=(
                pivot_box.y_min
                + local_box.y_max
            ),
        )

        area_fraction = (
            area_pixels / search_area
        )
        width_height_ratio = (
            local_box.width
            / local_box.height
        )

        shape = _measure_cross_shape(
            candidate_mask_local,
            local_box,
        )

        candidate_response = response_map[
            candidate_mask_local
        ]
        mean_depth = float(
            np.mean(candidate_response)
        )
        max_depth = float(
            np.max(candidate_response)
        )
        median_raw_height = float(
            np.median(
                pivot_crop[
                    candidate_mask_local
                ]
            )
        )

        shape_score = _calculate_shape_score(
            width_height_ratio=(
                width_height_ratio
            ),
            fill_ratio=shape["fill_ratio"],
            horizontal_arm_coverage=shape[
                "horizontal_arm_coverage"
            ],
            vertical_arm_coverage=shape[
                "vertical_arm_coverage"
            ],
            corner_occupancy=shape[
                "corner_occupancy"
            ],
        )

        horizontal_alignment_score = (
            _calculate_axis_alignment_score(
                candidate_position=(
                    center_x_local
                ),
                expected_position=(
                    expected_x_local
                ),
                full_size=pivot_box.width,
                tolerance_fraction=(
                    config
                    .horizontal_tolerance_fraction
                ),
            )
        )

        vertical_position_score = (
            _calculate_axis_alignment_score(
                candidate_position=(
                    center_y_local
                ),
                expected_position=(
                    expected_y_local
                ),
                full_size=max(
                    1,
                    upper_search_limit_y_global
                    - pivot_box.y_min,
                ),
                tolerance_fraction=(
                    config
                    .vertical_tolerance_fraction
                ),
            )
        )

        size_score = _calculate_size_score(
            area_pixels=area_pixels,
            width=local_box.width,
            height=local_box.height,
            area_fraction=area_fraction,
            expected_lower_area=(
                expected_lower_area
            ),
            expected_lower_width=(
                expected_lower_width
            ),
            expected_lower_height=(
                expected_lower_height
            ),
            config=config,
        )

        contrast_score = float(
            np.clip(
                (
                    mean_depth - threshold
                )
                / (
                    2.0
                    * robust_response_sigma
                ),
                0.0,
                1.0,
            )
        )

        score = (
            config.shape_weight
            * shape_score
            + config.horizontal_alignment_weight
            * horizontal_alignment_score
            + config.vertical_position_weight
            * vertical_position_score
            + config.size_weight
            * size_score
            + config.contrast_weight
            * contrast_score
        )

        candidates.append(
            UpperCrossCandidate(
                component_label=component_label,
                bounding_box=global_box,
                center_x=(
                    pivot_box.x_min
                    + center_x_local
                ),
                center_y=(
                    pivot_box.y_min
                    + center_y_local
                ),
                area_pixels=area_pixels,
                area_fraction=float(
                    area_fraction
                ),
                width_height_ratio=float(
                    width_height_ratio
                ),
                fill_ratio=shape[
                    "fill_ratio"
                ],
                horizontal_arm_coverage=shape[
                    "horizontal_arm_coverage"
                ],
                vertical_arm_coverage=shape[
                    "vertical_arm_coverage"
                ],
                corner_occupancy=shape[
                    "corner_occupancy"
                ],
                mean_depth=mean_depth,
                max_depth=max_depth,
                median_raw_height=(
                    median_raw_height
                ),
                shape_score=shape_score,
                horizontal_alignment_score=(
                    horizontal_alignment_score
                ),
                vertical_position_score=(
                    vertical_position_score
                ),
                size_score=size_score,
                contrast_score=(
                    contrast_score
                ),
                score=float(score),
            )
        )

    candidates.sort(
        key=lambda candidate: candidate.score,
        reverse=True,
    )

    return UpperCrossDetectionResult(
        pivot_bounding_box=pivot_box,
        crop_origin_x=pivot_box.x_min,
        crop_origin_y=pivot_box.y_min,
        crop_shape=pivot_crop.shape,
        upper_search_limit_y_global=(
            upper_search_limit_y_global
        ),
        threshold=threshold,
        robust_response_sigma=(
            robust_response_sigma
        ),
        response_map=response_map,
        search_mask=search_mask,
        threshold_mask=threshold_mask,
        label_image=label_image,
        config=config,
        candidates=candidates,
    )


def print_upper_cross_candidates(
    result: UpperCrossDetectionResult,
) -> None:
    """Print all upper-cross candidates in score order."""
    if not result.candidates:
        print(
            "No upper-cross candidates were found."
        )
        return

    print("\nAll upper-cross candidates:")
    print("-" * 160)

    for rank, candidate in enumerate(
        result.candidates,
        start=1,
    ):
        box = candidate.bounding_box
        selected = (
            " <-- SELECTED"
            if rank == 1
            else ""
        )

        print(
            f"Rank {rank}: "
            f"label={candidate.component_label}, "
            f"score={candidate.score:.4f}, "
            f"center=("
            f"{candidate.center_x:.2f}, "
            f"{candidate.center_y:.2f}), "
            f"bbox=("
            f"x={box.x_min}:{box.x_max}, "
            f"y={box.y_min}:{box.y_max}), "
            f"size={box.width}x{box.height}, "
            f"area={candidate.area_pixels}, "
            f"area_fraction="
            f"{candidate.area_fraction:.6f}, "
            f"aspect_ratio="
            f"{candidate.width_height_ratio:.3f}, "
            f"fill_ratio="
            f"{candidate.fill_ratio:.3f}, "
            f"horizontal_coverage="
            f"{candidate.horizontal_arm_coverage:.3f}, "
            f"vertical_coverage="
            f"{candidate.vertical_arm_coverage:.3f}, "
            f"corner_occupancy="
            f"{candidate.corner_occupancy:.3f}, "
            f"mean_depth="
            f"{candidate.mean_depth:.4f}, "
            f"max_depth="
            f"{candidate.max_depth:.4f}, "
            f"raw_height="
            f"{candidate.median_raw_height:.4f}, "
            f"shape_score="
            f"{candidate.shape_score:.3f}, "
            f"x_alignment="
            f"{candidate.horizontal_alignment_score:.3f}, "
            f"y_position="
            f"{candidate.vertical_position_score:.3f}, "
            f"size_score="
            f"{candidate.size_score:.3f}, "
            f"contrast_score="
            f"{candidate.contrast_score:.3f}"
            f"{selected}"
        )

    print("-" * 160)


def plot_upper_cross_detection(
    height_map: FloatArray,
    result: UpperCrossDetectionResult,
    lower_plane_detection: LowerPlaneDetectionResult,
    lower_cross_detection: LowerCrossDetectionResult | None = None,
) -> None:
    """Display Stage-4 masks, response and selected candidate."""
    pivot_box = result.pivot_bounding_box

    crop = height_map[
        pivot_box.y_min:pivot_box.y_max,
        pivot_box.x_min:pivot_box.x_max,
    ]

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6),
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
    axes[0].set_title(
        "Segmented Pivot crop"
    )

    response_image = axes[1].imshow(
        np.where(
            result.search_mask,
            result.response_map,
            np.nan,
        ),
        cmap="magma",
        aspect="auto",
    )
    figure.colorbar(
        response_image,
        ax=axes[1],
        label="Local depression response",
    )
    axes[1].set_title(
        "Upper-cross DoG response\n"
        f"threshold={result.threshold:.4f}"
    )

    axes[2].imshow(
        result.threshold_mask,
        cmap="gray",
        aspect="auto",
    )
    axes[2].set_title(
        "Thresholded candidates"
    )

    lower_plane = (
        lower_plane_detection.best_candidate
    )
    if lower_plane is not None:
        plane_box = lower_plane.bounding_box
        plane_box_local = BoundingBox(
            x_min=(
                plane_box.x_min
                - pivot_box.x_min
            ),
            y_min=(
                plane_box.y_min
                - pivot_box.y_min
            ),
            x_max=(
                plane_box.x_max
                - pivot_box.x_min
            ),
            y_max=(
                plane_box.y_max
                - pivot_box.y_min
            ),
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
                edgecolor="white",
                linestyle="--",
                linewidth=1.5,
                label="Lower plane",
            )
        )

    for rank, candidate in enumerate(
        result.candidates,
        start=1,
    ):
        best = rank == 1
        box = candidate.bounding_box

        local_box = BoundingBox(
            x_min=box.x_min - pivot_box.x_min,
            y_min=box.y_min - pivot_box.y_min,
            x_max=box.x_max - pivot_box.x_min,
            y_max=box.y_max - pivot_box.y_min,
        )

        for axis in axes:
            axis.add_patch(
                Rectangle(
                    (
                        local_box.x_min,
                        local_box.y_min,
                    ),
                    local_box.width,
                    local_box.height,
                    fill=False,
                    edgecolor=(
                        "red"
                        if best
                        else "cyan"
                    ),
                    linewidth=(
                        3
                        if best
                        else 1.5
                    ),
                )
            )

            axis.text(
                local_box.x_min,
                max(
                    0,
                    local_box.y_min - 2,
                ),
                (
                    f"#{rank}: "
                    f"{candidate.score:.3f}"
                ),
                color=(
                    "red"
                    if best
                    else "cyan"
                ),
                fontsize=9,
                bbox={
                    "facecolor": "black",
                    "alpha": 0.65,
                    "pad": 2,
                },
            )

    best_candidate = result.best_candidate

    if best_candidate is not None:
        center_x_local = (
            best_candidate.center_x
            - pivot_box.x_min
        )
        center_y_local = (
            best_candidate.center_y
            - pivot_box.y_min
        )

        for axis in axes:
            axis.scatter(
                center_x_local,
                center_y_local,
                marker="x",
                s=110,
                linewidths=3,
                color="red",
            )

    lower_cross = (
        lower_cross_detection.best_candidate
        if lower_cross_detection is not None
        else None
    )

    if lower_cross is not None:
        axes[0].scatter(
            lower_cross.center_x
            - pivot_box.x_min,
            lower_cross.center_y
            - pivot_box.y_min,
            marker="x",
            s=90,
            linewidths=2,
            color="white",
            label="Lower cross",
        )

    search_limit_local = (
        result.upper_search_limit_y_global
        - pivot_box.y_min
    )

    for axis in axes:
        axis.axhline(
            search_limit_local,
            linestyle="--",
            linewidth=1.5,
        )
        axis.set_xlabel(
            "Local X [pixels]"
        )
        axis.set_ylabel(
            "Local Y [pixels]"
        )

    axes[0].legend()

    figure.suptitle(
        "Stage 4 — Upper Pivot Cross Detection\n"
        f"confident={result.is_confident}"
    )
    figure.tight_layout()
    plt.show()


def get_upper_cross_detection(
    height_map: FloatArray,
    pivot_segmentation: PivotSegmentationResult,
    lower_plane_detection: LowerPlaneDetectionResult,
    lower_cross_detection: LowerCrossDetectionResult,
    print_debug: bool = False,
    show_debug: bool = False,
) -> UpperCrossDetectionResult:
    """Detect the upper Pivot cross and optionally display its debug plot."""
    with debug_print_context(print_debug):
        detection_result = find_upper_cross_candidates(
            height_map=height_map,
            pivot_segmentation=pivot_segmentation,
            lower_plane_detection=lower_plane_detection,
            lower_cross_detection=lower_cross_detection,
        )

    if print_debug:
        print_upper_cross_candidates(detection_result)

    if show_debug:
        plot_upper_cross_detection(
            height_map=height_map,
            result=detection_result,
            lower_plane_detection=lower_plane_detection,
            lower_cross_detection=lower_cross_detection,
        )

    return detection_result


def _create_upper_search_mask(
    pivot_mask_local: BoolArray,
    pivot_box: BoundingBox,
    lower_plane_box: BoundingBox,
    config: UpperCrossDetectionConfig,
) -> tuple[BoolArray, int]:
    """
    Create a temporary search mask inside the upper Pivot body.

    No inner bounding box is created or stored.
    """
    shortest_pivot_dimension = min(
        pivot_box.width,
        pivot_box.height,
    )

    edge_margin = max(
        config.min_pivot_edge_margin_pixels,
        int(
            round(
                shortest_pivot_dimension
                * config.pivot_edge_margin_fraction
            )
        ),
    )

    lower_plane_gap = max(
        config.min_lower_plane_gap_pixels,
        int(
            round(
                lower_plane_box.height
                * config.lower_plane_gap_fraction
            )
        ),
    )

    upper_search_limit_y_global = max(
        pivot_box.y_min + 1,
        lower_plane_box.y_min
        - lower_plane_gap,
    )

    upper_search_limit_y_local = int(
        np.clip(
            upper_search_limit_y_global
            - pivot_box.y_min,
            1,
            pivot_box.height,
        )
    )

    distance_to_pivot_boundary = (
        ndi.distance_transform_edt(
            pivot_mask_local
        )
    )

    search_mask = (
        pivot_mask_local
        & (
            distance_to_pivot_boundary
            >= edge_margin
        )
    )

    search_mask[
        upper_search_limit_y_local:,
        :
    ] = False

    return (
        search_mask,
        upper_search_limit_y_global,
    )


def _measure_cross_shape(
    candidate_mask: BoolArray,
    bounding_box: BoundingBox,
) -> dict[str, float]:
    """Measure plus-sign-like geometry without a fixed template."""
    crop = candidate_mask[
        bounding_box.y_min:bounding_box.y_max,
        bounding_box.x_min:bounding_box.x_max,
    ]

    height, width = crop.shape
    fill_ratio = float(
        np.mean(crop)
    )

    y_coordinates, x_coordinates = (
        np.nonzero(crop)
    )

    center_y = float(
        np.mean(y_coordinates)
    )
    center_x = float(
        np.mean(x_coordinates)
    )

    horizontal_half_band = max(
        1,
        int(round(height * 0.125)),
    )
    vertical_half_band = max(
        1,
        int(round(width * 0.125)),
    )

    horizontal_start = max(
        0,
        int(round(center_y))
        - horizontal_half_band,
    )
    horizontal_stop = min(
        height,
        int(round(center_y))
        + horizontal_half_band
        + 1,
    )

    vertical_start = max(
        0,
        int(round(center_x))
        - vertical_half_band,
    )
    vertical_stop = min(
        width,
        int(round(center_x))
        + vertical_half_band
        + 1,
    )

    horizontal_band = crop[
        horizontal_start:horizontal_stop,
        :,
    ]
    vertical_band = crop[
        :,
        vertical_start:vertical_stop,
    ]

    horizontal_arm_coverage = float(
        np.mean(
            np.any(
                horizontal_band,
                axis=0,
            )
        )
    )
    vertical_arm_coverage = float(
        np.mean(
            np.any(
                vertical_band,
                axis=1,
            )
        )
    )

    corner_height = max(
        1,
        height // 3,
    )
    corner_width = max(
        1,
        width // 3,
    )

    corner_pixels = np.concatenate(
        [
            crop[
                :corner_height,
                :corner_width,
            ].ravel(),
            crop[
                :corner_height,
                -corner_width:,
            ].ravel(),
            crop[
                -corner_height:,
                :corner_width,
            ].ravel(),
            crop[
                -corner_height:,
                -corner_width:,
            ].ravel(),
        ]
    )

    corner_occupancy = float(
        np.mean(corner_pixels)
    )

    return {
        "fill_ratio": fill_ratio,
        "horizontal_arm_coverage": (
            horizontal_arm_coverage
        ),
        "vertical_arm_coverage": (
            vertical_arm_coverage
        ),
        "corner_occupancy": (
            corner_occupancy
        ),
    }


def _calculate_shape_score(
    width_height_ratio: float,
    fill_ratio: float,
    horizontal_arm_coverage: float,
    vertical_arm_coverage: float,
    corner_occupancy: float,
) -> float:
    aspect_score = float(
        np.exp(
            -abs(
                np.log(
                    max(
                        width_height_ratio,
                        1e-8,
                    )
                )
            )
            / 0.7
        )
    )

    fill_score = float(
        np.exp(
            -0.5
            * (
                (
                    fill_ratio - 0.45
                )
                / 0.25
            )
            ** 2
        )
    )

    score = (
        0.20 * aspect_score
        + 0.25
        * horizontal_arm_coverage
        + 0.25
        * vertical_arm_coverage
        + 0.20
        * (1.0 - corner_occupancy)
        + 0.10 * fill_score
    )

    return float(
        np.clip(
            score,
            0.0,
            1.0,
        )
    )


def _calculate_axis_alignment_score(
    candidate_position: float,
    expected_position: float,
    full_size: int,
    tolerance_fraction: float,
) -> float:
    half_size = max(
        full_size / 2.0,
        1.0,
    )

    normalized_distance = (
        abs(
            candidate_position
            - expected_position
        )
        / half_size
    )

    tolerance = max(
        tolerance_fraction,
        1e-8,
    )

    return float(
        np.exp(
            -0.5
            * (
                normalized_distance
                / tolerance
            )
            ** 2
        )
    )


def _calculate_size_score(
    area_pixels: int,
    width: int,
    height: int,
    area_fraction: float,
    expected_lower_area: int | None,
    expected_lower_width: int | None,
    expected_lower_height: int | None,
    config: UpperCrossDetectionConfig,
) -> float:
    """
    Prefer a candidate similar in size to the lower cross.

    If the lower cross is unavailable, use the configured expected area
    fraction as a weaker fallback.
    """
    if (
        expected_lower_area is None
        or expected_lower_width is None
        or expected_lower_height is None
    ):
        return _log_ratio_score(
            actual=area_fraction,
            expected=(
                config.expected_area_fraction
            ),
            tolerance_factor=(
                config.area_tolerance_factor
            ),
        )

    area_score = _log_ratio_score(
        actual=float(area_pixels),
        expected=float(expected_lower_area),
        tolerance_factor=(
            config
            .lower_cross_size_tolerance_factor
        ),
    )
    width_score = _log_ratio_score(
        actual=float(width),
        expected=float(expected_lower_width),
        tolerance_factor=(
            config
            .lower_cross_size_tolerance_factor
        ),
    )
    height_score = _log_ratio_score(
        actual=float(height),
        expected=float(expected_lower_height),
        tolerance_factor=(
            config
            .lower_cross_size_tolerance_factor
        ),
    )

    return float(
        (
            area_score
            + width_score
            + height_score
        )
        / 3.0
    )


def _log_ratio_score(
    actual: float,
    expected: float,
    tolerance_factor: float,
) -> float:
    if actual <= 0 or expected <= 0:
        return 0.0

    logarithmic_distance = abs(
        np.log(actual / expected)
    )
    logarithmic_tolerance = max(
        np.log(tolerance_factor),
        1e-8,
    )

    return float(
        np.exp(
            -0.5
            * (
                logarithmic_distance
                / logarithmic_tolerance
            )
            ** 2
        )
    )


def _validate_inputs(
    height_map: np.ndarray,
    pivot_segmentation: PivotSegmentationResult,
    lower_plane_detection: LowerPlaneDetectionResult,
    config: UpperCrossDetectionConfig,
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
            f"Expected numeric data, received "
            f"{height_map.dtype}."
        )

    if not np.isfinite(height_map).all():
        raise ValueError(
            "Height map contains NaN or infinite values."
        )

    if (
        pivot_segmentation.pivot_mask.shape
        != height_map.shape
    ):
        raise ValueError(
            "Pivot mask and height map must have the same shape."
        )

    if (
        lower_plane_detection.label_image.shape
        != height_map.shape
    ):
        raise ValueError(
            "Lower-plane detection and height map must have the same shape."
        )

    if config.small_gaussian_sigma < 0:
        raise ValueError(
            "small_gaussian_sigma cannot be negative."
        )

    if (
        config.large_gaussian_sigma
        <= config.small_gaussian_sigma
    ):
        raise ValueError(
            "large_gaussian_sigma must be greater than "
            "small_gaussian_sigma."
        )

    if config.threshold_mad_multiplier <= 0:
        raise ValueError(
            "threshold_mad_multiplier must be positive."
        )

    if not (
        0
        <= config.pivot_edge_margin_fraction
        < 0.5
    ):
        raise ValueError(
            "pivot_edge_margin_fraction must be in [0, 0.5)."
        )

    if not (
        0
        <= config.lower_plane_gap_fraction
        < 1.0
    ):
        raise ValueError(
            "lower_plane_gap_fraction must be in [0, 1)."
        )

    if (
        config.min_area_fraction < 0
        or config.max_area_fraction <= 0
        or config.min_area_fraction
        >= config.max_area_fraction
    ):
        raise ValueError(
            "Invalid candidate-area fractions."
        )

    weights = (
        config.shape_weight
        + config.horizontal_alignment_weight
        + config.vertical_position_weight
        + config.size_weight
        + config.contrast_weight
    )

    if not np.isclose(
        weights,
        1.0,
    ):
        raise ValueError(
            "Upper-cross score weights must sum to 1.0."
        )
