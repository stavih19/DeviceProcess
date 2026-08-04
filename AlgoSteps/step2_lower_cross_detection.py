# from __future__ import annotations

# from dataclasses import dataclass, field

# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.patches import Rectangle
# from numpy.typing import NDArray
# from scipy import ndimage as ndi

# from pivot_candidates import (
#     BoolArray,
#     BoundingBox,
#     FloatArray,
#     IntArray,
#     LowerPlaneCandidate,
#     LowerPlaneDetectionResult,
# )


# @dataclass(frozen=True)
# class LowerCrossDetectionConfig:
#     """
#     Configuration for detecting the lower Pivot cross.

#     No inner bounding box is required. A temporary conservative search mask is
#     created directly from the selected lower-plane component by excluding a
#     configurable band near its boundary.
#     """

#     # Difference-of-Gaussians parameters.
#     small_gaussian_sigma: float = 0.8
#     large_gaussian_sigma: float = 8.0

#     # Robust threshold:
#     # median(response) + threshold_mad_multiplier * robust_sigma
#     threshold_mad_multiplier: float = 3.0

#     # Morphological cleanup of the thresholded cross response.
#     opening_size: int = 1
#     closing_size: int = 3

#     # Exclude partial/sloped pixels near the lower-plane boundary.
#     edge_margin_fraction: float = 0.08
#     min_edge_margin_pixels: int = 3

#     # Cross connected-component limits relative to the temporary search area.
#     min_area_fraction: float = 0.0005
#     max_area_fraction: float = 0.08

#     min_width_pixels: int = 3
#     min_height_pixels: int = 3

#     # Used only for ranking, not as a strict assumption.
#     expected_area_fraction: float = 0.02
#     area_tolerance_factor: float = 3.0

#     # The lower cross is expected near the lower-plane centre.
#     center_tolerance_fraction: float = 0.30

#     # Candidate-score weights.
#     shape_weight: float = 0.40
#     center_weight: float = 0.30
#     area_weight: float = 0.15
#     contrast_weight: float = 0.15


# @dataclass(frozen=True)
# class LowerCrossCandidate:
#     """Measurements describing one possible lower Pivot cross."""

#     component_label: int

#     # Coordinates are relative to the original height map.
#     bounding_box: BoundingBox
#     center_x: float
#     center_y: float

#     area_pixels: int
#     area_fraction: float

#     width_height_ratio: float
#     fill_ratio: float

#     horizontal_arm_coverage: float
#     vertical_arm_coverage: float
#     corner_occupancy: float

#     mean_depth: float
#     max_depth: float

#     shape_score: float
#     center_score: float
#     area_score: float
#     contrast_score: float
#     score: float


# @dataclass
# class LowerCrossDetectionResult:
#     """
#     Result of lower-cross detection for one selected lower-plane candidate.

#     response_map, search_mask, threshold_mask and label_image use local
#     coordinates relative to crop_origin_x and crop_origin_y.
#     """

#     lower_plane_candidate: LowerPlaneCandidate

#     crop_origin_x: int
#     crop_origin_y: int
#     crop_shape: tuple[int, int]

#     threshold: float
#     robust_response_sigma: float

#     response_map: NDArray[np.float32]
#     search_mask: BoolArray
#     threshold_mask: BoolArray
#     label_image: IntArray

#     config: LowerCrossDetectionConfig
#     candidates: list[LowerCrossCandidate] = field(default_factory=list)

#     @property
#     def best_candidate(self) -> LowerCrossCandidate | None:
#         if not self.candidates:
#             return None
#         return self.candidates[0]

#     def get_candidate_mask_local(
#         self,
#         candidate: LowerCrossCandidate,
#     ) -> BoolArray:
#         return self.label_image == candidate.component_label

#     def get_candidate_mask_global(
#         self,
#         candidate: LowerCrossCandidate,
#         image_shape: tuple[int, int],
#     ) -> BoolArray:
#         global_mask = np.zeros(image_shape, dtype=bool)
#         local_mask = self.get_candidate_mask_local(candidate)

#         crop_height, crop_width = self.crop_shape

#         global_mask[
#             self.crop_origin_y:self.crop_origin_y + crop_height,
#             self.crop_origin_x:self.crop_origin_x + crop_width,
#         ] = local_mask

#         return global_mask


# def find_lower_cross_candidates(
#     height_map: FloatArray,
#     lower_plane_detection: LowerPlaneDetectionResult,
#     lower_plane_candidate: LowerPlaneCandidate | None = None,
#     config: LowerCrossDetectionConfig | None = None,
# ) -> LowerCrossDetectionResult:
#     """
#     Detect and rank lower-cross candidates inside a lower-plane candidate.

#     The lower cross is treated as a local depression. A difference-of-Gaussians
#     response is calculated:

#         response = large_scale_background - small_scale_height_map

#     No inner bounding box is used. Partial edge pixels are excluded by creating
#     a temporary search mask from the component's distance to its boundary.
#     """
#     if config is None:
#         config = LowerCrossDetectionConfig()

#     _validate_inputs(
#         height_map=height_map,
#         lower_plane_detection=lower_plane_detection,
#         config=config,
#     )

#     if lower_plane_candidate is None:
#         lower_plane_candidate = lower_plane_detection.best_candidate

#     if lower_plane_candidate is None:
#         raise ValueError(
#             "Lower-cross detection requires a lower-plane candidate."
#         )

#     height_map = height_map.astype(np.float32, copy=False)

#     plane_box = lower_plane_candidate.bounding_box

#     plane_crop = height_map[
#         plane_box.y_min:plane_box.y_max,
#         plane_box.x_min:plane_box.x_max,
#     ]

#     component_mask_global = lower_plane_detection.get_candidate_mask(
#         lower_plane_candidate
#     )
#     component_mask_local = component_mask_global[
#         plane_box.y_min:plane_box.y_max,
#         plane_box.x_min:plane_box.x_max,
#     ]

#     # Temporary conservative mask only. It is not stored in the lower-plane
#     # candidate and does not create an inner bounding box.
#     search_mask = _create_cross_search_mask(
#         component_mask=component_mask_local,
#         plane_box=plane_box,
#         config=config,
#     )

#     if not np.any(search_mask):
#         raise ValueError(
#             "The lower-plane component does not contain a valid interior "
#             "region for cross detection."
#         )

#     small_scale = ndi.gaussian_filter(
#         plane_crop,
#         sigma=config.small_gaussian_sigma,
#     )
#     large_scale = ndi.gaussian_filter(
#         plane_crop,
#         sigma=config.large_gaussian_sigma,
#     )

#     # A depression is lower than the local large-scale estimate.
#     response_map = (large_scale - small_scale).astype(np.float32)

#     response_values = response_map[search_mask]
#     response_median = float(np.median(response_values))
#     response_mad = float(
#         np.median(np.abs(response_values - response_median))
#     )
#     robust_response_sigma = max(1.4826 * response_mad, 1e-8)

#     threshold = float(
#         response_median
#         + config.threshold_mad_multiplier * robust_response_sigma
#     )

#     threshold_mask = (response_map >= threshold) & search_mask

#     if config.opening_size > 1:
#         threshold_mask = ndi.binary_opening(
#             threshold_mask,
#             structure=np.ones(
#                 (config.opening_size, config.opening_size),
#                 dtype=bool,
#             ),
#         )

#     if config.closing_size > 1:
#         threshold_mask = ndi.binary_closing(
#             threshold_mask,
#             structure=np.ones(
#                 (config.closing_size, config.closing_size),
#                 dtype=bool,
#             ),
#         )

#     label_image, component_count = ndi.label(threshold_mask)

#     search_area = max(1, int(np.count_nonzero(search_mask)))

#     min_area = max(
#         1,
#         int(round(search_area * config.min_area_fraction)),
#     )
#     max_area = max(
#         min_area,
#         int(round(search_area * config.max_area_fraction)),
#     )

#     expected_center_x_local = (
#         lower_plane_candidate.centroid_x - plane_box.x_min
#     )
#     expected_center_y_local = (
#         lower_plane_candidate.centroid_y - plane_box.y_min
#     )

#     candidates: list[LowerCrossCandidate] = []
#     component_slices = ndi.find_objects(label_image)

#     for component_label in range(1, component_count + 1):
#         component_slice = component_slices[component_label - 1]

#         if component_slice is None:
#             continue

#         local_component_mask = label_image == component_label
#         area_pixels = int(np.count_nonzero(local_component_mask))

#         if area_pixels < min_area or area_pixels > max_area:
#             continue

#         y_slice, x_slice = component_slice

#         local_box = BoundingBox(
#             x_min=x_slice.start,
#             y_min=y_slice.start,
#             x_max=x_slice.stop,
#             y_max=y_slice.stop,
#         )

#         if (
#             local_box.width < config.min_width_pixels
#             or local_box.height < config.min_height_pixels
#         ):
#             continue

#         local_y, local_x = np.nonzero(local_component_mask)

#         center_x_local = float(np.mean(local_x))
#         center_y_local = float(np.mean(local_y))

#         global_box = BoundingBox(
#             x_min=plane_box.x_min + local_box.x_min,
#             y_min=plane_box.y_min + local_box.y_min,
#             x_max=plane_box.x_min + local_box.x_max,
#             y_max=plane_box.y_min + local_box.y_max,
#         )

#         area_fraction = area_pixels / search_area
#         width_height_ratio = local_box.width / local_box.height

#         shape_measurements = _measure_cross_shape(
#             local_component_mask=local_component_mask,
#             local_box=local_box,
#         )

#         component_response = response_map[local_component_mask]
#         mean_depth = float(np.mean(component_response))
#         max_depth = float(np.max(component_response))

#         shape_score = _calculate_shape_score(
#             width_height_ratio=width_height_ratio,
#             fill_ratio=shape_measurements["fill_ratio"],
#             horizontal_arm_coverage=shape_measurements[
#                 "horizontal_arm_coverage"
#             ],
#             vertical_arm_coverage=shape_measurements[
#                 "vertical_arm_coverage"
#             ],
#             corner_occupancy=shape_measurements["corner_occupancy"],
#         )

#         center_score = _calculate_center_score(
#             center_x_local=center_x_local,
#             center_y_local=center_y_local,
#             expected_center_x_local=expected_center_x_local,
#             expected_center_y_local=expected_center_y_local,
#             plane_width=plane_box.width,
#             plane_height=plane_box.height,
#             tolerance_fraction=config.center_tolerance_fraction,
#         )

#         area_score = _calculate_area_score(
#             area_fraction=area_fraction,
#             expected_area_fraction=config.expected_area_fraction,
#             tolerance_factor=config.area_tolerance_factor,
#         )

#         contrast_score = float(
#             np.clip(
#                 (mean_depth - threshold)
#                 / (2.0 * robust_response_sigma),
#                 0.0,
#                 1.0,
#             )
#         )

#         score = (
#             config.shape_weight * shape_score
#             + config.center_weight * center_score
#             + config.area_weight * area_score
#             + config.contrast_weight * contrast_score
#         )

#         candidates.append(
#             LowerCrossCandidate(
#                 component_label=component_label,
#                 bounding_box=global_box,
#                 center_x=plane_box.x_min + center_x_local,
#                 center_y=plane_box.y_min + center_y_local,
#                 area_pixels=area_pixels,
#                 area_fraction=float(area_fraction),
#                 width_height_ratio=float(width_height_ratio),
#                 fill_ratio=shape_measurements["fill_ratio"],
#                 horizontal_arm_coverage=shape_measurements[
#                     "horizontal_arm_coverage"
#                 ],
#                 vertical_arm_coverage=shape_measurements[
#                     "vertical_arm_coverage"
#                 ],
#                 corner_occupancy=shape_measurements["corner_occupancy"],
#                 mean_depth=mean_depth,
#                 max_depth=max_depth,
#                 shape_score=shape_score,
#                 center_score=center_score,
#                 area_score=area_score,
#                 contrast_score=contrast_score,
#                 score=float(score),
#             )
#         )

#     candidates.sort(
#         key=lambda candidate: candidate.score,
#         reverse=True,
#     )

#     return LowerCrossDetectionResult(
#         lower_plane_candidate=lower_plane_candidate,
#         crop_origin_x=plane_box.x_min,
#         crop_origin_y=plane_box.y_min,
#         crop_shape=plane_crop.shape,
#         threshold=threshold,
#         robust_response_sigma=robust_response_sigma,
#         response_map=response_map,
#         search_mask=search_mask,
#         threshold_mask=threshold_mask,
#         label_image=label_image,
#         config=config,
#         candidates=candidates,
#     )


# def print_lower_cross_candidates(
#     detection_result: LowerCrossDetectionResult,
# ) -> None:
#     """Print all candidates in descending score order."""
#     if not detection_result.candidates:
#         print("No lower-cross candidates were found.")
#         return

#     print("\nAll lower-cross candidates:")
#     print("-" * 150)

#     for rank, candidate in enumerate(
#         detection_result.candidates,
#         start=1,
#     ):
#         box = candidate.bounding_box
#         selected = " <-- SELECTED" if rank == 1 else ""

#         print(
#             f"Rank {rank}: "
#             f"label={candidate.component_label}, "
#             f"score={candidate.score:.4f}, "
#             f"center=({candidate.center_x:.2f}, "
#             f"{candidate.center_y:.2f}), "
#             f"bbox=(x={box.x_min}:{box.x_max}, "
#             f"y={box.y_min}:{box.y_max}), "
#             f"size={box.width}x{box.height}, "
#             f"area={candidate.area_pixels}, "
#             f"area_fraction={candidate.area_fraction:.5f}, "
#             f"aspect_ratio={candidate.width_height_ratio:.3f}, "
#             f"fill_ratio={candidate.fill_ratio:.3f}, "
#             f"horizontal_coverage="
#             f"{candidate.horizontal_arm_coverage:.3f}, "
#             f"vertical_coverage="
#             f"{candidate.vertical_arm_coverage:.3f}, "
#             f"corner_occupancy={candidate.corner_occupancy:.3f}, "
#             f"mean_depth={candidate.mean_depth:.4f}, "
#             f"max_depth={candidate.max_depth:.4f}, "
#             f"shape_score={candidate.shape_score:.3f}, "
#             f"center_score={candidate.center_score:.3f}, "
#             f"area_score={candidate.area_score:.3f}, "
#             f"contrast_score={candidate.contrast_score:.3f}"
#             f"{selected}"
#         )

#     print("-" * 150)


# def plot_lower_cross_detection(
#     height_map: FloatArray,
#     detection_result: LowerCrossDetectionResult,
# ) -> None:
#     """
#     Display the lower-plane crop, response map and thresholded candidates.
#     """
#     plane_box = detection_result.lower_plane_candidate.bounding_box

#     plane_crop = height_map[
#         plane_box.y_min:plane_box.y_max,
#         plane_box.x_min:plane_box.x_max,
#     ]

#     figure, axes = plt.subplots(
#         1,
#         3,
#         figsize=(18, 6),
#     )

#     height_image = axes[0].imshow(
#         plane_crop,
#         cmap="viridis",
#         aspect="auto",
#     )
#     figure.colorbar(
#         height_image,
#         ax=axes[0],
#         label="Height",
#     )
#     axes[0].set_title("Lower-plane crop")

#     response_image = axes[1].imshow(
#         np.where(
#             detection_result.search_mask,
#             detection_result.response_map,
#             np.nan,
#         ),
#         cmap="magma",
#         aspect="auto",
#     )
#     figure.colorbar(
#         response_image,
#         ax=axes[1],
#         label="Local depression response",
#     )
#     axes[1].set_title(
#         "Difference-of-Gaussians response\n"
#         f"threshold={detection_result.threshold:.4f}"
#     )

#     axes[2].imshow(
#         detection_result.threshold_mask,
#         cmap="gray",
#         aspect="auto",
#     )
#     axes[2].set_title("Thresholded cross candidates")

#     for rank, candidate in enumerate(
#         detection_result.candidates,
#         start=1,
#     ):
#         is_best = rank == 1
#         global_box = candidate.bounding_box

#         local_box = BoundingBox(
#             x_min=global_box.x_min - plane_box.x_min,
#             y_min=global_box.y_min - plane_box.y_min,
#             x_max=global_box.x_max - plane_box.x_min,
#             y_max=global_box.y_max - plane_box.y_min,
#         )

#         for axis in axes:
#             rectangle = Rectangle(
#                 (local_box.x_min, local_box.y_min),
#                 local_box.width,
#                 local_box.height,
#                 fill=False,
#                 edgecolor="red" if is_best else "cyan",
#                 linewidth=3 if is_best else 1.5,
#             )
#             axis.add_patch(rectangle)

#             axis.text(
#                 local_box.x_min,
#                 max(0, local_box.y_min - 2),
#                 f"#{rank}: {candidate.score:.3f}",
#                 color="red" if is_best else "cyan",
#                 fontsize=9,
#                 bbox={
#                     "facecolor": "black",
#                     "alpha": 0.65,
#                     "pad": 2,
#                 },
#             )

#     best_candidate = detection_result.best_candidate

#     if best_candidate is not None:
#         center_x_local = best_candidate.center_x - plane_box.x_min
#         center_y_local = best_candidate.center_y - plane_box.y_min

#         for axis in axes:
#             axis.scatter(
#                 center_x_local,
#                 center_y_local,
#                 marker="x",
#                 s=100,
#                 linewidths=3,
#                 color="red",
#             )

#     for axis in axes:
#         axis.set_xlabel("Local X [pixels]")
#         axis.set_ylabel("Local Y [pixels]")

#     figure.suptitle(
#         "Lower Pivot Cross Detection\n"
#         "Selected candidate is marked in red"
#     )
#     figure.tight_layout()
#     plt.show()


# def _create_cross_search_mask(
#     component_mask: BoolArray,
#     plane_box: BoundingBox,
#     config: LowerCrossDetectionConfig,
# ) -> BoolArray:
#     """
#     Exclude a conservative boundary band without creating an inner box.
#     """
#     shortest_dimension = min(
#         plane_box.width,
#         plane_box.height,
#     )

#     edge_margin = max(
#         config.min_edge_margin_pixels,
#         int(
#             round(
#                 shortest_dimension
#                 * config.edge_margin_fraction
#             )
#         ),
#     )

#     distance_to_boundary = ndi.distance_transform_edt(
#         component_mask
#     )

#     return (
#         component_mask
#         & (distance_to_boundary >= edge_margin)
#     )


# def _measure_cross_shape(
#     local_component_mask: BoolArray,
#     local_box: BoundingBox,
# ) -> dict[str, float]:
#     """Measure whether a component resembles a plus sign."""
#     crop = local_component_mask[
#         local_box.y_min:local_box.y_max,
#         local_box.x_min:local_box.x_max,
#     ]

#     height, width = crop.shape
#     fill_ratio = float(np.mean(crop))

#     local_y, local_x = np.nonzero(crop)
#     center_y = float(np.mean(local_y))
#     center_x = float(np.mean(local_x))

#     horizontal_half_band = max(
#         1,
#         int(round(height * 0.125)),
#     )
#     vertical_half_band = max(
#         1,
#         int(round(width * 0.125)),
#     )

#     horizontal_start = max(
#         0,
#         int(round(center_y)) - horizontal_half_band,
#     )
#     horizontal_stop = min(
#         height,
#         int(round(center_y)) + horizontal_half_band + 1,
#     )

#     vertical_start = max(
#         0,
#         int(round(center_x)) - vertical_half_band,
#     )
#     vertical_stop = min(
#         width,
#         int(round(center_x)) + vertical_half_band + 1,
#     )

#     horizontal_band = crop[
#         horizontal_start:horizontal_stop,
#         :,
#     ]
#     vertical_band = crop[
#         :,
#         vertical_start:vertical_stop,
#     ]

#     horizontal_arm_coverage = float(
#         np.mean(np.any(horizontal_band, axis=0))
#     )
#     vertical_arm_coverage = float(
#         np.mean(np.any(vertical_band, axis=1))
#     )

#     corner_height = max(1, height // 3)
#     corner_width = max(1, width // 3)

#     corner_pixels = np.concatenate(
#         [
#             crop[:corner_height, :corner_width].ravel(),
#             crop[:corner_height, -corner_width:].ravel(),
#             crop[-corner_height:, :corner_width].ravel(),
#             crop[-corner_height:, -corner_width:].ravel(),
#         ]
#     )
#     corner_occupancy = float(np.mean(corner_pixels))

#     return {
#         "fill_ratio": fill_ratio,
#         "horizontal_arm_coverage": horizontal_arm_coverage,
#         "vertical_arm_coverage": vertical_arm_coverage,
#         "corner_occupancy": corner_occupancy,
#     }


# def _calculate_shape_score(
#     width_height_ratio: float,
#     fill_ratio: float,
#     horizontal_arm_coverage: float,
#     vertical_arm_coverage: float,
#     corner_occupancy: float,
# ) -> float:
#     """Calculate a template-free plus-sign shape score."""
#     aspect_score = float(
#         np.exp(
#             -abs(
#                 np.log(
#                     max(width_height_ratio, 1e-8)
#                 )
#             )
#             / 0.7
#         )
#     )

#     fill_score = float(
#         np.exp(
#             -0.5
#             * (
#                 (fill_ratio - 0.45)
#                 / 0.25
#             )
#             ** 2
#         )
#     )

#     score = (
#         0.20 * aspect_score
#         + 0.25 * horizontal_arm_coverage
#         + 0.25 * vertical_arm_coverage
#         + 0.20 * (1.0 - corner_occupancy)
#         + 0.10 * fill_score
#     )

#     return float(np.clip(score, 0.0, 1.0))


# def _calculate_center_score(
#     center_x_local: float,
#     center_y_local: float,
#     expected_center_x_local: float,
#     expected_center_y_local: float,
#     plane_width: int,
#     plane_height: int,
#     tolerance_fraction: float,
# ) -> float:
#     """Score the candidate's normalized 2D distance from the plane centre."""
#     normalized_dx = (
#         center_x_local - expected_center_x_local
#     ) / max(plane_width / 2.0, 1.0)

#     normalized_dy = (
#         center_y_local - expected_center_y_local
#     ) / max(plane_height / 2.0, 1.0)

#     normalized_distance = float(
#         np.hypot(normalized_dx, normalized_dy)
#     )

#     tolerance = max(tolerance_fraction, 1e-8)

#     return float(
#         np.exp(
#             -0.5
#             * (
#                 normalized_distance
#                 / tolerance
#             )
#             ** 2
#         )
#     )


# def _calculate_area_score(
#     area_fraction: float,
#     expected_area_fraction: float,
#     tolerance_factor: float,
# ) -> float:
#     if area_fraction <= 0:
#         return 0.0

#     logarithmic_distance = abs(
#         np.log(
#             area_fraction
#             / expected_area_fraction
#         )
#     )

#     logarithmic_tolerance = max(
#         np.log(tolerance_factor),
#         1e-8,
#     )

#     return float(
#         np.exp(
#             -0.5
#             * (
#                 logarithmic_distance
#                 / logarithmic_tolerance
#             )
#             ** 2
#         )
#     )


# def _validate_inputs(
#     height_map: np.ndarray,
#     lower_plane_detection: LowerPlaneDetectionResult,
#     config: LowerCrossDetectionConfig,
# ) -> None:
#     if height_map.ndim != 2:
#         raise ValueError(
#             "Expected a two-dimensional height map, "
#             f"received shape {height_map.shape}."
#         )

#     if lower_plane_detection.label_image.shape != height_map.shape:
#         raise ValueError(
#             "The lower-plane detection result and height map "
#             "must have the same shape."
#         )

#     if config.small_gaussian_sigma < 0:
#         raise ValueError(
#             "small_gaussian_sigma cannot be negative."
#         )

#     if (
#         config.large_gaussian_sigma
#         <= config.small_gaussian_sigma
#     ):
#         raise ValueError(
#             "large_gaussian_sigma must be greater than "
#             "small_gaussian_sigma."
#         )

#     if config.threshold_mad_multiplier <= 0:
#         raise ValueError(
#             "threshold_mad_multiplier must be positive."
#         )

#     if not 0 <= config.edge_margin_fraction < 0.5:
#         raise ValueError(
#             "edge_margin_fraction must be in [0, 0.5)."
#         )

#     if config.min_edge_margin_pixels < 0:
#         raise ValueError(
#             "min_edge_margin_pixels cannot be negative."
#         )

#     if (
#         config.min_area_fraction < 0
#         or config.max_area_fraction <= 0
#         or config.min_area_fraction
#         >= config.max_area_fraction
#     ):
#         raise ValueError(
#             "Invalid cross component area fractions."
#         )

#     if config.expected_area_fraction <= 0:
#         raise ValueError(
#             "expected_area_fraction must be positive."
#         )

#     if config.area_tolerance_factor <= 1:
#         raise ValueError(
#             "area_tolerance_factor must be greater than 1."
#         )

#     weights = (
#         config.shape_weight
#         + config.center_weight
#         + config.area_weight
#         + config.contrast_weight
#     )

#     if not np.isclose(weights, 1.0):
#         raise ValueError(
#             "Cross-candidate score weights must sum to 1.0."
#         )

from __future__ import annotations

from AlgoSteps.debug_utils import debug_print_context

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from numpy.typing import NDArray
from scipy import ndimage as ndi

from AlgoSteps.step1_pivot_candidates import (
    BoolArray,
    BoundingBox,
    FloatArray,
    IntArray,
    LowerPlaneCandidate,
    LowerPlaneDetectionResult,
)


@dataclass(frozen=True)
class LowerCrossDetectionConfig:
    """
    Configurable parameters for detecting the lower Pivot cross.

    The cross is assumed to be a local depression relative to the surrounding
    lower plane. The defaults are initial values and should be validated on all
    supplied datasets.
    """

    # Small smoothing preserves the cross while reducing pixel noise.
    small_gaussian_sigma: float = 0.8

    # Large smoothing estimates the slowly varying local plane/background.
    large_gaussian_sigma: float = 8.0

    # Threshold = response median + multiplier * robust standard deviation.
    threshold_mad_multiplier: float = 3.0

    # Candidate-mask cleanup.
    opening_size: int = 1
    closing_size: int = 3

    # Exclude pixels too close to the detected plane boundary.
    edge_margin_fraction: float = 0.08
    min_edge_margin_pixels: int = 3

    # Candidate component size relative to the conservative lower-plane mask.
    min_area_fraction: float = 0.0005
    max_area_fraction: float = 0.08

    min_width_pixels: int = 3
    min_height_pixels: int = 3

    # Approximate expected cross area fraction. Used only for ranking.
    expected_area_fraction: float = 0.02
    area_tolerance_factor: float = 3.0

    # Expected horizontal alignment with the lower-plane centre.
    horizontal_center_tolerance_fraction: float = 0.25

    # Relative weights used for ranking cross candidates.
    shape_weight: float = 0.40
    horizontal_center_weight: float = 0.30
    area_weight: float = 0.15
    contrast_weight: float = 0.15


@dataclass(frozen=True)
class LowerCrossCandidate:
    """Measurements describing one possible lower Pivot cross."""

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

    shape_score: float
    horizontal_center_score: float
    area_score: float
    contrast_score: float
    score: float


@dataclass
class LowerCrossDetectionResult:
    """
    Result of lower-cross detection for one lower-plane candidate.

    response_map, search_mask, threshold_mask and label_image use local
    coordinates relative to crop_origin_x/crop_origin_y.
    """

    lower_plane_candidate: LowerPlaneCandidate

    crop_origin_x: int
    crop_origin_y: int
    crop_shape: tuple[int, int]

    threshold: float
    robust_response_sigma: float

    response_map: NDArray[np.float32]
    search_mask: BoolArray
    threshold_mask: BoolArray
    label_image: IntArray

    config: LowerCrossDetectionConfig
    candidates: list[LowerCrossCandidate] = field(default_factory=list)

    @property
    def best_candidate(self) -> LowerCrossCandidate | None:
        if not self.candidates:
            return None
        return self.candidates[0]

    def get_candidate_mask_local(
        self,
        candidate: LowerCrossCandidate,
    ) -> BoolArray:
        return self.label_image == candidate.component_label

    def get_candidate_mask_global(
        self,
        candidate: LowerCrossCandidate,
        image_shape: tuple[int, int],
    ) -> BoolArray:
        global_mask = np.zeros(image_shape, dtype=bool)
        local_mask = self.get_candidate_mask_local(candidate)

        y_start = self.crop_origin_y
        x_start = self.crop_origin_x
        crop_height, crop_width = self.crop_shape

        global_mask[
            y_start:y_start + crop_height,
            x_start:x_start + crop_width,
        ] = local_mask

        return global_mask


def find_lower_cross_candidates(
    height_map: FloatArray,
    lower_plane_detection: LowerPlaneDetectionResult,
    lower_plane_candidate: LowerPlaneCandidate | None = None,
    config: LowerCrossDetectionConfig | None = None,
) -> LowerCrossDetectionResult:
    """
    Detect and rank lower-cross candidates inside a detected lower plane.

    The algorithm uses a difference-of-Gaussians response:

        response = large_scale_background - small_scale_height_map

    A local depression, such as the cross, therefore receives a positive
    response. The search is restricted to a conservative interior region of
    the selected lower-plane component.
    """
    if config is None:
        config = LowerCrossDetectionConfig()

    _validate_inputs(
        height_map=height_map,
        lower_plane_detection=lower_plane_detection,
        config=config,
    )

    if lower_plane_candidate is None:
        lower_plane_candidate = lower_plane_detection.best_candidate

    if lower_plane_candidate is None:
        raise ValueError(
            "Lower-cross detection requires at least one lower-plane candidate."
        )

    height_map = height_map.astype(np.float32, copy=False)

    outer_box = lower_plane_candidate.bounding_box
    crop = height_map[
        outer_box.y_min:outer_box.y_max,
        outer_box.x_min:outer_box.x_max,
    ]

    component_mask_global = lower_plane_detection.get_candidate_mask(
        lower_plane_candidate
    )
    inner_mask_global = lower_plane_detection.get_candidate_mask(
        lower_plane_candidate
    )

    component_mask = component_mask_global[
        outer_box.y_min:outer_box.y_max,
        outer_box.x_min:outer_box.x_max,
    ]
    inner_mask = inner_mask_global[
        outer_box.y_min:outer_box.y_max,
        outer_box.x_min:outer_box.x_max,
    ]

    search_mask = _create_cross_search_mask(
        component_mask=component_mask,
        inner_mask=inner_mask,
        outer_box=outer_box,
        config=config,
    )

    if not np.any(search_mask):
        raise ValueError(
            "The lower-plane candidate does not contain a valid interior "
            "region for cross detection."
        )

    # Estimate slow local variation and compare it with a lightly smoothed map.
    small_scale = ndi.gaussian_filter(
        crop,
        sigma=config.small_gaussian_sigma,
    )
    large_scale = ndi.gaussian_filter(
        crop,
        sigma=config.large_gaussian_sigma,
    )

    # The lower cross is a local depression, so its response should be positive.
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

    measurement_area = max(
        1,
        lower_plane_candidate.area_pixels,
    )
    min_area = max(
        1,
        int(round(measurement_area * config.min_area_fraction)),
    )
    max_area = max(
        min_area,
        int(round(measurement_area * config.max_area_fraction)),
    )

    inner_box = lower_plane_candidate.bounding_box
    expected_center_x_local = (
        (inner_box.x_min + inner_box.x_max - 1) / 2
        - outer_box.x_min
    )

    candidates: list[LowerCrossCandidate] = []
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
            x_min=x_slice.start,
            y_min=y_slice.start,
            x_max=x_slice.stop,
            y_max=y_slice.stop,
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
            x_min=outer_box.x_min + local_box.x_min,
            y_min=outer_box.y_min + local_box.y_min,
            x_max=outer_box.x_min + local_box.x_max,
            y_max=outer_box.y_min + local_box.y_max,
        )

        area_fraction = area_pixels / measurement_area
        width_height_ratio = local_box.width / local_box.height

        shape_measurements = _measure_cross_shape(
            local_component_mask=local_component_mask,
            local_box=local_box,
        )

        component_response = response_map[local_component_mask]
        mean_depth = float(np.mean(component_response))
        max_depth = float(np.max(component_response))

        shape_score = _calculate_shape_score(
            width_height_ratio=width_height_ratio,
            fill_ratio=shape_measurements["fill_ratio"],
            horizontal_arm_coverage=shape_measurements[
                "horizontal_arm_coverage"
            ],
            vertical_arm_coverage=shape_measurements[
                "vertical_arm_coverage"
            ],
            corner_occupancy=shape_measurements["corner_occupancy"],
        )

        horizontal_center_score = _calculate_horizontal_center_score(
            center_x_local=center_x_local,
            expected_center_x_local=expected_center_x_local,
            plane_width=inner_box.width,
            tolerance_fraction=(
                config.horizontal_center_tolerance_fraction
            ),
        )

        area_score = _calculate_area_score(
            area_fraction=area_fraction,
            expected_area_fraction=config.expected_area_fraction,
            tolerance_factor=config.area_tolerance_factor,
        )

        contrast_score = float(
            np.clip(
                (
                    mean_depth - threshold
                )
                / (2.0 * robust_response_sigma),
                0.0,
                1.0,
            )
        )

        score = (
            config.shape_weight * shape_score
            + config.horizontal_center_weight * horizontal_center_score
            + config.area_weight * area_score
            + config.contrast_weight * contrast_score
        )

        candidates.append(
            LowerCrossCandidate(
                component_label=component_label,
                bounding_box=global_box,
                center_x=outer_box.x_min + center_x_local,
                center_y=outer_box.y_min + center_y_local,
                area_pixels=area_pixels,
                area_fraction=float(area_fraction),
                width_height_ratio=float(width_height_ratio),
                fill_ratio=shape_measurements["fill_ratio"],
                horizontal_arm_coverage=shape_measurements[
                    "horizontal_arm_coverage"
                ],
                vertical_arm_coverage=shape_measurements[
                    "vertical_arm_coverage"
                ],
                corner_occupancy=shape_measurements["corner_occupancy"],
                mean_depth=mean_depth,
                max_depth=max_depth,
                shape_score=shape_score,
                horizontal_center_score=horizontal_center_score,
                area_score=area_score,
                contrast_score=contrast_score,
                score=float(score),
            )
        )

    candidates.sort(
        key=lambda candidate: candidate.score,
        reverse=True,
    )

    return LowerCrossDetectionResult(
        lower_plane_candidate=lower_plane_candidate,
        crop_origin_x=outer_box.x_min,
        crop_origin_y=outer_box.y_min,
        crop_shape=crop.shape,
        threshold=threshold,
        robust_response_sigma=robust_response_sigma,
        response_map=response_map,
        search_mask=search_mask,
        threshold_mask=threshold_mask,
        label_image=label_image,
        config=config,
        candidates=candidates,
    )


def print_lower_cross_candidates(
    detection_result: LowerCrossDetectionResult,
) -> None:
    """Print all lower-cross candidates in descending score order."""
    if not detection_result.candidates:
        print("No lower-cross candidates were found.")
        return

    print("\nAll lower-cross candidates:")
    print("-" * 150)

    for rank, candidate in enumerate(
        detection_result.candidates,
        start=1,
    ):
        box = candidate.bounding_box
        selected = " <-- SELECTED" if rank == 1 else ""

        print(
            f"Rank {rank}: "
            f"label={candidate.component_label}, "
            f"score={candidate.score:.4f}, "
            f"center=({candidate.center_x:.2f}, "
            f"{candidate.center_y:.2f}), "
            f"bbox=(x={box.x_min}:{box.x_max}, "
            f"y={box.y_min}:{box.y_max}), "
            f"size={box.width}x{box.height}, "
            f"area={candidate.area_pixels}, "
            f"area_fraction={candidate.area_fraction:.5f}, "
            f"aspect_ratio={candidate.width_height_ratio:.3f}, "
            f"fill_ratio={candidate.fill_ratio:.3f}, "
            f"horizontal_coverage="
            f"{candidate.horizontal_arm_coverage:.3f}, "
            f"vertical_coverage="
            f"{candidate.vertical_arm_coverage:.3f}, "
            f"corner_occupancy={candidate.corner_occupancy:.3f}, "
            f"mean_depth={candidate.mean_depth:.4f}, "
            f"max_depth={candidate.max_depth:.4f}, "
            f"shape_score={candidate.shape_score:.3f}, "
            f"center_score={candidate.horizontal_center_score:.3f}, "
            f"area_score={candidate.area_score:.3f}, "
            f"contrast_score={candidate.contrast_score:.3f}"
            f"{selected}"
        )

    print("-" * 150)


def plot_lower_cross_detection(
    height_map: FloatArray,
    detection_result: LowerCrossDetectionResult,
) -> None:
    """
    Display the lower-plane crop, local-depression response and threshold mask.
    """
    outer_box = detection_result.lower_plane_candidate.bounding_box
    crop = height_map[
        outer_box.y_min:outer_box.y_max,
        outer_box.x_min:outer_box.x_max,
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
    axes[0].set_title("Lower-plane crop")

    response_image = axes[1].imshow(
        np.where(
            detection_result.search_mask,
            detection_result.response_map,
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
        "Difference-of-Gaussians response\n"
        f"threshold={detection_result.threshold:.4f}"
    )

    axes[2].imshow(
        detection_result.threshold_mask,
        cmap="gray",
        aspect="auto",
    )
    axes[2].set_title("Thresholded cross candidates")

    for rank, candidate in enumerate(
        detection_result.candidates,
        start=1,
    ):
        is_best = rank == 1
        global_box = candidate.bounding_box

        local_box = BoundingBox(
            x_min=global_box.x_min - outer_box.x_min,
            y_min=global_box.y_min - outer_box.y_min,
            x_max=global_box.x_max - outer_box.x_min,
            y_max=global_box.y_max - outer_box.y_min,
        )

        for axis in axes:
            rectangle = Rectangle(
                (local_box.x_min, local_box.y_min),
                local_box.width,
                local_box.height,
                fill=False,
                edgecolor="red" if is_best else "cyan",
                linewidth=3 if is_best else 1.5,
            )
            axis.add_patch(rectangle)

            axis.text(
                local_box.x_min,
                max(0, local_box.y_min - 2),
                f"#{rank}: {candidate.score:.3f}",
                color="red" if is_best else "cyan",
                fontsize=9,
                bbox={
                    "facecolor": "black",
                    "alpha": 0.65,
                    "pad": 2,
                },
            )

    best_candidate = detection_result.best_candidate
    if best_candidate is not None:
        center_x_local = best_candidate.center_x - outer_box.x_min
        center_y_local = best_candidate.center_y - outer_box.y_min

        for axis in axes:
            axis.scatter(
                center_x_local,
                center_y_local,
                marker="x",
                s=100,
                linewidths=3,
                color="red",
            )

    for axis in axes:
        axis.set_xlabel("Local X [pixels]")
        axis.set_ylabel("Local Y [pixels]")

    figure.suptitle(
        "Lower Pivot Cross Detection\n"
        "Selected candidate is marked in red"
    )
    figure.tight_layout()
    plt.show()


def get_lower_cross_detection(
    height_map: FloatArray,
    lower_plane_detection: LowerPlaneDetectionResult,
    print_debug: bool = False,
    show_debug: bool = False,
) -> LowerCrossDetectionResult:
    """Detect the lower Pivot cross and optionally display its debug plot."""
    best_plane = lower_plane_detection.best_candidate
    if best_plane is None:
        raise ValueError(
            "Lower-cross detection requires a valid lower-plane candidate."
        )

    with debug_print_context(print_debug):
        detection_result = find_lower_cross_candidates(
            height_map=height_map,
            lower_plane_detection=lower_plane_detection,
            lower_plane_candidate=best_plane,
        )

    if detection_result.best_candidate is None:
        raise ValueError("No lower Pivot cross candidate was found.")

    if print_debug:
        print_lower_cross_candidates(detection_result)

    if show_debug:
        plot_lower_cross_detection(
            height_map=height_map,
            detection_result=detection_result,
        )

    return detection_result


def _create_cross_search_mask(
    component_mask: BoolArray,
    inner_mask: BoolArray,
    outer_box: BoundingBox,
    config: LowerCrossDetectionConfig,
) -> BoolArray:
    """
    Build a conservative search mask that excludes partial component edges.
    """
    shortest_dimension = min(
        outer_box.width,
        outer_box.height,
    )
    edge_margin = max(
        config.min_edge_margin_pixels,
        int(
            round(
                shortest_dimension
                * config.edge_margin_fraction
            )
        ),
    )

    distance_to_boundary = ndi.distance_transform_edt(
        component_mask
    )

    return (
        inner_mask
        & (distance_to_boundary >= edge_margin)
    )


def _measure_cross_shape(
    local_component_mask: BoolArray,
    local_box: BoundingBox,
) -> dict[str, float]:
    """
    Measure whether a component resembles a plus sign without using a template.
    """
    crop = local_component_mask[
        local_box.y_min:local_box.y_max,
        local_box.x_min:local_box.x_max,
    ]

    height, width = crop.shape
    fill_ratio = float(np.mean(crop))

    local_y, local_x = np.nonzero(crop)
    center_y = float(np.mean(local_y))
    center_x = float(np.mean(local_x))

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

    horizontal_band = crop[
        horizontal_start:horizontal_stop,
        :,
    ]
    vertical_band = crop[
        :,
        vertical_start:vertical_stop,
    ]

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


def _calculate_shape_score(
    width_height_ratio: float,
    fill_ratio: float,
    horizontal_arm_coverage: float,
    vertical_arm_coverage: float,
    corner_occupancy: float,
) -> float:
    """
    Calculate a template-free plus-sign shape score in the [0, 1] range.
    """
    aspect_score = float(
        np.exp(
            -abs(
                np.log(
                    max(width_height_ratio, 1e-8)
                )
            )
            / 0.7
        )
    )

    # A plus sign usually occupies about half of its bounding box.
    fill_score = float(
        np.exp(
            -0.5
            * (
                (fill_ratio - 0.45)
                / 0.25
            )
            ** 2
        )
    )

    score = (
        0.20 * aspect_score
        + 0.25 * horizontal_arm_coverage
        + 0.25 * vertical_arm_coverage
        + 0.20 * (1.0 - corner_occupancy)
        + 0.10 * fill_score
    )

    return float(np.clip(score, 0.0, 1.0))


def _calculate_horizontal_center_score(
    center_x_local: float,
    expected_center_x_local: float,
    plane_width: int,
    tolerance_fraction: float,
) -> float:
    half_width = max(plane_width / 2.0, 1.0)
    normalized_distance = (
        abs(
            center_x_local
            - expected_center_x_local
        )
        / half_width
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


def _calculate_area_score(
    area_fraction: float,
    expected_area_fraction: float,
    tolerance_factor: float,
) -> float:
    if area_fraction <= 0:
        return 0.0

    logarithmic_distance = abs(
        np.log(
            area_fraction
            / expected_area_fraction
        )
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
    lower_plane_detection: LowerPlaneDetectionResult,
    config: LowerCrossDetectionConfig,
) -> None:
    if height_map.ndim != 2:
        raise ValueError(
            "Expected a two-dimensional height map, "
            f"received shape {height_map.shape}."
        )

    if lower_plane_detection.label_image.shape != height_map.shape:
        raise ValueError(
            "The lower-plane detection result and height map "
            "must have the same shape."
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

    if not 0 <= config.edge_margin_fraction < 0.5:
        raise ValueError(
            "edge_margin_fraction must be in [0, 0.5)."
        )

    if config.min_edge_margin_pixels < 0:
        raise ValueError(
            "min_edge_margin_pixels cannot be negative."
        )

    if (
        config.min_area_fraction < 0
        or config.max_area_fraction <= 0
        or config.min_area_fraction
        >= config.max_area_fraction
    ):
        raise ValueError(
            "Invalid cross component area fractions."
        )

    if config.expected_area_fraction <= 0:
        raise ValueError(
            "expected_area_fraction must be positive."
        )

    if config.area_tolerance_factor <= 1:
        raise ValueError(
            "area_tolerance_factor must be greater than 1."
        )

    weights = (
        config.shape_weight
        + config.horizontal_center_weight
        + config.area_weight
        + config.contrast_weight
    )
    if not np.isclose(weights, 1.0):
        raise ValueError(
            "Cross candidate score weights must sum to 1.0."
        )
