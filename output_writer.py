from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from models import AnalysisResult, OutputFileNames, RawAnalysisResult


RESULT_JSON_FILE = "result.json"
LABEL_MAP_FILE = "labels.npy"
LABEL_IMAGE_FILE = "labels.png"


def validate_label_map(
    label_map: np.ndarray,
    expected_shape: tuple[int, int],
) -> None:
    if label_map.shape != expected_shape:
        raise ValueError(
            f"Label-map shape {label_map.shape} does not match input shape "
            f"{expected_shape}."
        )

    allowed_labels = {0, 1, 2}
    actual_labels = set(np.unique(label_map).tolist())

    if not actual_labels.issubset(allowed_labels):
        raise ValueError(
            f"Unsupported labels {sorted(actual_labels)}. "
            "Allowed values are 0, 1 and 2."
        )


def save_label_preview(label_map: np.ndarray, output_path: Path) -> None:
    """
    Save an RGB preview:
    black = background
    red   = Pivot
    green = Xpander

    Exact numeric labels are stored separately in labels.npy.
    """
    preview = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    preview[label_map == 1] = (255, 0, 0)
    preview[label_map == 2] = (0, 255, 0)

    Image.fromarray(preview, mode="RGB").save(output_path)


def save_result_files(
    input_file_name: str,
    input_shape: tuple[int, int],
    raw_result: RawAnalysisResult,
    output_root: Path,
) -> AnalysisResult:
    """
    Save all outputs belonging to one input in a dedicated directory.

    Example:
        outputs/
            0/
                result.json
                labels.npy
                labels.png
    """
    if raw_result.label_map is None:
        raise ValueError("The algorithm did not return a label map.")

    validate_label_map(raw_result.label_map, input_shape)

    input_stem = Path(input_file_name).stem
    file_output_dir = output_root / input_stem
    file_output_dir.mkdir(parents=True, exist_ok=True)

    np.save(
        file_output_dir / LABEL_MAP_FILE,
        raw_result.label_map,
        allow_pickle=False,
    )
    save_label_preview(
        raw_result.label_map,
        file_output_dir / LABEL_IMAGE_FILE,
    )

    output_files = OutputFileNames(
        result_json=RESULT_JSON_FILE,
        label_map_npy=LABEL_MAP_FILE,
        label_image_png=LABEL_IMAGE_FILE,
    )

    result = AnalysisResult(
        input_file_name=input_file_name,
        status=raw_result.status,
        pivot_bounding_box_px=raw_result.pivot_bounding_box_px,
        xpander_bounding_box_px=raw_result.xpander_bounding_box_px,
        pivot_height_difference_um=raw_result.pivot_height_difference_um,
        pivot_cross_centers_px=raw_result.pivot_cross_centers_px,
        xpander_radius_x_um=raw_result.xpander_radius_x_um,
        xpander_radius_y_um=raw_result.xpander_radius_y_um,
        radius_fit_score_x=raw_result.radius_fit_score_x,
        radius_fit_score_y=raw_result.radius_fit_score_y,
        radius_fit_score_overall=raw_result.radius_fit_score_overall,
        tilt_x_deg=raw_result.tilt_x_deg,
        tilt_y_deg=raw_result.tilt_y_deg,
        rotation_deg=raw_result.rotation_deg,
        faults=raw_result.faults,
        output_files=output_files,
    )

    with (file_output_dir / RESULT_JSON_FILE).open("w", encoding="utf-8") as file:
        json.dump(asdict(result), file, indent=2, ensure_ascii=False)

    return result


def save_batch_summary(
    results: list[AnalysisResult],
    output_root: Path,
) -> None:
    """Save aggregate JSON and CSV files for all processed inputs."""

    # ------------------------------------------------------------------
    # Aggregate JSON
    # ------------------------------------------------------------------
    with (output_root / "results.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            [asdict(result) for result in results],
            file,
            indent=2,
            ensure_ascii=False,
        )

    # ------------------------------------------------------------------
    # Aggregate CSV
    # ------------------------------------------------------------------
    with (output_root / "results.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        field_names = [
            "input_file_name",
            "status",

            # Pivot bounding box:
            # TL -> TR -> BR -> BL
            "pivot_tl_x_px",
            "pivot_tl_y_px",
            "pivot_tr_x_px",
            "pivot_tr_y_px",
            "pivot_br_x_px",
            "pivot_br_y_px",
            "pivot_bl_x_px",
            "pivot_bl_y_px",

            # Xpander bounding box:
            # TL -> TR -> BR -> BL
            "xpander_tl_x_px",
            "xpander_tl_y_px",
            "xpander_tr_x_px",
            "xpander_tr_y_px",
            "xpander_br_x_px",
            "xpander_br_y_px",
            "xpander_bl_x_px",
            "xpander_bl_y_px",

            # Pivot measurements
            "pivot_height_difference_um",

            # Cross centres
            "lower_cross_x_px",
            "lower_cross_y_px",
            "upper_cross_x_px",
            "upper_cross_y_px",

            # Xpander curvature
            "xpander_radius_x_um",
            "xpander_radius_y_um",

            # Fit quality
            "radius_fit_score_x",
            "radius_fit_score_y",
            "radius_fit_score_overall",

            # Alignment correction
            "tilt_x_deg",
            "tilt_y_deg",
            "rotation_deg",

            # Fault information
            "faulty_elements",
            "fault_reasons",

            # Generated files
            "result_json",
            "label_map_npy",
            "label_image_png",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=field_names,
        )
        writer.writeheader()

        for result in results:

            # ----------------------------------------------------------
            # Cross centres
            #
            # algorithm.py stores them in this order:
            #   [lower_cross, upper_cross]
            # ----------------------------------------------------------
            crosses = result.pivot_cross_centers_px

            lower_cross = (
                crosses[0]
                if len(crosses) > 0
                else None
            )

            upper_cross = (
                crosses[1]
                if len(crosses) > 1
                else None
            )

            # ----------------------------------------------------------
            # Pivot bounding box
            #
            # Expected order:
            #   0 = top-left
            #   1 = top-right
            #   2 = bottom-right
            #   3 = bottom-left
            # ----------------------------------------------------------
            pivot_box = result.pivot_bounding_box_px

            pivot_tl = (
                pivot_box[0]
                if len(pivot_box) > 0
                else None
            )
            pivot_tr = (
                pivot_box[1]
                if len(pivot_box) > 1
                else None
            )
            pivot_br = (
                pivot_box[2]
                if len(pivot_box) > 2
                else None
            )
            pivot_bl = (
                pivot_box[3]
                if len(pivot_box) > 3
                else None
            )

            # ----------------------------------------------------------
            # Xpander bounding box
            #
            # Expected order:
            #   0 = top-left
            #   1 = top-right
            #   2 = bottom-right
            #   3 = bottom-left
            # ----------------------------------------------------------
            xpander_box = result.xpander_bounding_box_px

            xpander_tl = (
                xpander_box[0]
                if len(xpander_box) > 0
                else None
            )
            xpander_tr = (
                xpander_box[1]
                if len(xpander_box) > 1
                else None
            )
            xpander_br = (
                xpander_box[2]
                if len(xpander_box) > 2
                else None
            )
            xpander_bl = (
                xpander_box[3]
                if len(xpander_box) > 3
                else None
            )

            writer.writerow(
                {
                    "input_file_name":
                        result.input_file_name,

                    "status":
                        result.status,

                    # Pivot BB
                    "pivot_tl_x_px":
                        pivot_tl.x if pivot_tl else None,
                    "pivot_tl_y_px":
                        pivot_tl.y if pivot_tl else None,

                    "pivot_tr_x_px":
                        pivot_tr.x if pivot_tr else None,
                    "pivot_tr_y_px":
                        pivot_tr.y if pivot_tr else None,

                    "pivot_br_x_px":
                        pivot_br.x if pivot_br else None,
                    "pivot_br_y_px":
                        pivot_br.y if pivot_br else None,

                    "pivot_bl_x_px":
                        pivot_bl.x if pivot_bl else None,
                    "pivot_bl_y_px":
                        pivot_bl.y if pivot_bl else None,

                    # Xpander BB
                    "xpander_tl_x_px":
                        xpander_tl.x if xpander_tl else None,
                    "xpander_tl_y_px":
                        xpander_tl.y if xpander_tl else None,

                    "xpander_tr_x_px":
                        xpander_tr.x if xpander_tr else None,
                    "xpander_tr_y_px":
                        xpander_tr.y if xpander_tr else None,

                    "xpander_br_x_px":
                        xpander_br.x if xpander_br else None,
                    "xpander_br_y_px":
                        xpander_br.y if xpander_br else None,

                    "xpander_bl_x_px":
                        xpander_bl.x if xpander_bl else None,
                    "xpander_bl_y_px":
                        xpander_bl.y if xpander_bl else None,

                    # Pivot measurement
                    "pivot_height_difference_um":
                        result.pivot_height_difference_um,

                    # Crosses
                    "lower_cross_x_px":
                        lower_cross.x if lower_cross else None,
                    "lower_cross_y_px":
                        lower_cross.y if lower_cross else None,

                    "upper_cross_x_px":
                        upper_cross.x if upper_cross else None,
                    "upper_cross_y_px":
                        upper_cross.y if upper_cross else None,

                    # Curvature
                    "xpander_radius_x_um":
                        result.xpander_radius_x_um,
                    "xpander_radius_y_um":
                        result.xpander_radius_y_um,

                    # Fit scores
                    "radius_fit_score_x":
                        result.radius_fit_score_x,
                    "radius_fit_score_y":
                        result.radius_fit_score_y,
                    "radius_fit_score_overall":
                        result.radius_fit_score_overall,

                    # Alignment
                    "tilt_x_deg":
                        result.tilt_x_deg,
                    "tilt_y_deg":
                        result.tilt_y_deg,
                    "rotation_deg":
                        result.rotation_deg,

                    # Faults
                    "faulty_elements": "; ".join(
                        fault.element
                        for fault in result.faults
                    ),

                    "fault_reasons": "; ".join(
                        fault.reason
                        for fault in result.faults
                    ),
                }
            )