from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy import ndimage as ndi


Status = Literal["PASS", "WARN", "FAIL"]


@dataclass(frozen=True)
class RangeRule:
    warn_min: float
    warn_max: float
    fail_min: float
    fail_max: float


@dataclass(frozen=True)
class MaxRule:
    warn_max: float
    fail_max: float


@dataclass(frozen=True)
class MinRule:
    warn_min: float
    fail_min: float


@dataclass(frozen=True)
class PlausibilityConfig:
    # Absolute dimensions observed in the approved baseline set so far.
    pivot_width: RangeRule = RangeRule(255.0, 320.0, 220.0, 360.0)
    pivot_height: RangeRule = RangeRule(178.0, 198.0, 150.0, 230.0)
    xpander_width: RangeRule = RangeRule(475.0, 515.0, 420.0, 580.0)
    xpander_height: RangeRule = RangeRule(480.0, 525.0, 420.0, 600.0)

    # Scale-independent geometry.
    pivot_aspect_ratio: RangeRule = RangeRule(1.40, 1.75, 1.25, 1.90)
    xpander_aspect_ratio: RangeRule = RangeRule(0.93, 1.07, 0.85, 1.15)
    xpander_to_pivot_width: RangeRule = RangeRule(1.55, 1.90, 1.40, 2.05)
    xpander_to_pivot_height: RangeRule = RangeRule(2.55, 2.85, 2.35, 3.05)

    # Cross geometry. result.json stores lower cross first and upper cross second.
    cross_delta_x_px: MaxRule = MaxRule(warn_max=3.0, fail_max=8.0)
    cross_vertical_separation_px: RangeRule = RangeRule(108.0, 125.0, 95.0, 140.0)
    cross_vertical_separation_over_pivot_height: RangeRule = RangeRule(
        0.58, 0.68, 0.50, 0.76
    )
    cross_to_pivot_center_x_px: MaxRule = MaxRule(warn_max=4.0, fail_max=10.0)

    # Horizontal relationship between the two components.
    xpander_pivot_center_x_fraction_of_pivot_width: MaxRule = MaxRule(
        warn_max=0.06, fail_max=0.12
    )
    xpander_left_right_extension_asymmetry: MaxRule = MaxRule(
        warn_max=0.25, fail_max=0.45
    )

    # Vertical gap: Pivot top - Xpander bottom.
    xpander_to_pivot_vertical_gap_px: RangeRule = RangeRule(
        85.0, 130.0, 60.0, 165.0
    )

    # Numeric outputs.
    pivot_height_difference_um: RangeRule = RangeRule(
        17.8, 18.5, 16.5, 20.0
    )
    xpander_radius_um: RangeRule = RangeRule(
        790.0, 875.0, 700.0, 1000.0
    )
    radius_ratio: RangeRule = RangeRule(0.95, 1.05, 0.90, 1.10)
    radius_fit_score: MinRule = MinRule(warn_min=0.93, fail_min=0.85)

    # Label-map structural sanity.
    largest_component_fraction: MinRule = MinRule(warn_min=0.98, fail_min=0.90)


@dataclass(frozen=True)
class BBox:
    x_min: int
    y_min: int
    x_max: int  # exclusive
    y_max: int  # exclusive

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    @property
    def center_x(self) -> float:
        return (self.x_min + self.x_max - 1) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y_min + self.y_max - 1) / 2.0


@dataclass
class CheckResult:
    name: str
    status: Status
    value: Any
    expected: str
    details: str = ""


@dataclass
class FileReport:
    name: str
    status: Status
    checks: list[CheckResult]
    result_dir: Path


STATUS_RANK: dict[Status, int] = {"PASS": 0, "WARN": 1, "FAIL": 2}


def worst_status(*statuses: Status) -> Status:
    return max(statuses, key=lambda status: STATUS_RANK[status])


def format_number(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return str(value)
        return f"{float(value):.6f}".rstrip("0").rstrip(".")
    return str(value)


def check_range(name: str, value: float, rule: RangeRule) -> CheckResult:
    if not math.isfinite(value):
        return CheckResult(name, "FAIL", value, "finite numeric value")

    if value < rule.fail_min or value > rule.fail_max:
        status: Status = "FAIL"
    elif value < rule.warn_min or value > rule.warn_max:
        status = "WARN"
    else:
        status = "PASS"

    expected = (
        f"PASS {rule.warn_min:g}..{rule.warn_max:g}; "
        f"FAIL outside {rule.fail_min:g}..{rule.fail_max:g}"
    )
    return CheckResult(name, status, value, expected)


def check_max(name: str, value: float, rule: MaxRule) -> CheckResult:
    if not math.isfinite(value):
        return CheckResult(name, "FAIL", value, "finite numeric value")

    if value > rule.fail_max:
        status: Status = "FAIL"
    elif value > rule.warn_max:
        status = "WARN"
    else:
        status = "PASS"

    expected = f"PASS <= {rule.warn_max:g}; FAIL > {rule.fail_max:g}"
    return CheckResult(name, status, value, expected)


def check_min(name: str, value: float, rule: MinRule) -> CheckResult:
    if not math.isfinite(value):
        return CheckResult(name, "FAIL", value, "finite numeric value")

    if value < rule.fail_min:
        status: Status = "FAIL"
    elif value < rule.warn_min:
        status = "WARN"
    else:
        status = "PASS"

    expected = f"PASS >= {rule.warn_min:g}; FAIL < {rule.fail_min:g}"
    return CheckResult(name, status, value, expected)


def check_bool(name: str, condition: bool, expected: str, details: str = "") -> CheckResult:
    return CheckResult(
        name=name,
        status="PASS" if condition else "FAIL",
        value=condition,
        expected=expected,
        details=details,
    )


def bbox_from_label(labels: np.ndarray, label: int) -> BBox | None:
    ys, xs = np.nonzero(labels == label)
    if xs.size == 0:
        return None
    return BBox(
        x_min=int(xs.min()),
        y_min=int(ys.min()),
        x_max=int(xs.max()) + 1,
        y_max=int(ys.max()) + 1,
    )


def largest_connected_component_fraction(mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    total = int(mask.sum())
    if total == 0:
        return 0.0

    structure = np.ones((3, 3), dtype=np.uint8)
    labeled, component_count = ndi.label(mask, structure=structure)
    if component_count == 0:
        return 0.0

    counts = np.bincount(labeled.ravel())
    largest = int(counts[1:].max()) if counts.size > 1 else 0
    return largest / total


def get_float(data: dict[str, Any], key: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return None
    return value_float if math.isfinite(value_float) else None


def parse_cross_centers(data: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float]] | None:
    centers = data.get("pivot_cross_centers_px")
    if not isinstance(centers, list) or len(centers) != 2:
        return None

    parsed: list[tuple[float, float]] = []
    for center in centers:
        if not isinstance(center, dict):
            return None
        try:
            x = float(center["x"])
            y = float(center["y"])
        except (KeyError, TypeError, ValueError):
            return None
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        parsed.append((x, y))

    # algorithm._collect_cross_centers stores lower first, then upper.
    return parsed[0], parsed[1]


def discover_result_dirs(path: Path) -> list[Path]:
    path = path.resolve()

    if path.is_file():
        if path.name == "result.json":
            return [path.parent]
        raise ValueError("Input file must be result.json, or pass an output directory.")

    if not path.is_dir():
        raise ValueError(f"Path does not exist or is not a directory: {path}")

    if (path / "result.json").is_file():
        return [path]

    result_dirs = sorted(
        result_json.parent
        for result_json in path.rglob("result.json")
        if result_json.is_file()
    )
    return result_dirs


def validate_one_result_dir(
    result_dir: Path,
    config: PlausibilityConfig,
) -> FileReport:
    checks: list[CheckResult] = []
    result_path = result_dir / "result.json"

    try:
        with result_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception as error:
        return FileReport(
            name=result_dir.name,
            status="FAIL",
            checks=[CheckResult("result_json_readable", "FAIL", str(error), "valid result.json")],
            result_dir=result_dir,
        )

    input_name = str(data.get("input_file_name") or result_dir.name)

    status_text = str(data.get("status", ""))
    status_ok = status_text.lower() in {"ok", "completed", "success"}
    checks.append(
        CheckResult(
            "pipeline_status",
            "PASS" if status_ok else "FAIL",
            status_text,
            "ok/completed/success",
        )
    )

    output_files = data.get("output_files") or {}
    labels_name = output_files.get("label_map_npy", "labels.npy")
    labels_path = result_dir / labels_name

    try:
        labels = np.load(labels_path, allow_pickle=False)
    except Exception as error:
        checks.append(
            CheckResult("labels_readable", "FAIL", str(error), f"readable {labels_name}")
        )
        return FileReport(input_name, "FAIL", checks, result_dir)

    checks.append(check_bool("labels_are_2d", labels.ndim == 2, "2-D label map"))
    if labels.ndim != 2:
        return FileReport(input_name, "FAIL", checks, result_dir)

    actual_labels = set(np.unique(labels).tolist())
    checks.append(
        check_bool(
            "label_values",
            actual_labels.issubset({0, 1, 2}),
            "labels subset of {0,1,2}",
            details=f"actual={sorted(actual_labels)}",
        )
    )

    pivot_box = bbox_from_label(labels, 1)
    xpander_box = bbox_from_label(labels, 2)
    checks.append(check_bool("pivot_present", pivot_box is not None, "label 1 exists"))
    checks.append(check_bool("xpander_present", xpander_box is not None, "label 2 exists"))

    if pivot_box is not None:
        checks.append(check_range("pivot_width_px", pivot_box.width, config.pivot_width))
        checks.append(check_range("pivot_height_px", pivot_box.height, config.pivot_height))
        checks.append(
            check_range(
                "pivot_width_over_height",
                pivot_box.width / max(pivot_box.height, 1),
                config.pivot_aspect_ratio,
            )
        )
        pivot_component_fraction = largest_connected_component_fraction(labels == 1)
        checks.append(
            check_min(
                "pivot_largest_component_fraction",
                pivot_component_fraction,
                config.largest_component_fraction,
            )
        )

    if xpander_box is not None:
        checks.append(check_range("xpander_width_px", xpander_box.width, config.xpander_width))
        checks.append(check_range("xpander_height_px", xpander_box.height, config.xpander_height))
        checks.append(
            check_range(
                "xpander_width_over_height",
                xpander_box.width / max(xpander_box.height, 1),
                config.xpander_aspect_ratio,
            )
        )
        xpander_component_fraction = largest_connected_component_fraction(labels == 2)
        checks.append(
            check_min(
                "xpander_largest_component_fraction",
                xpander_component_fraction,
                config.largest_component_fraction,
            )
        )

    if pivot_box is not None and xpander_box is not None:
        checks.append(
            check_range(
                "xpander_width_over_pivot_width",
                xpander_box.width / max(pivot_box.width, 1),
                config.xpander_to_pivot_width,
            )
        )
        checks.append(
            check_range(
                "xpander_height_over_pivot_height",
                xpander_box.height / max(pivot_box.height, 1),
                config.xpander_to_pivot_height,
            )
        )

        checks.append(
            check_bool(
                "xpander_wider_than_pivot",
                xpander_box.width > pivot_box.width,
                "Xpander width > Pivot width",
            )
        )
        checks.append(
            check_bool(
                "xpander_taller_than_pivot",
                xpander_box.height > pivot_box.height,
                "Xpander height > Pivot height",
            )
        )
        checks.append(
            check_bool(
                "xpander_spans_pivot_horizontally",
                xpander_box.x_min < pivot_box.x_min
                and xpander_box.x_max > pivot_box.x_max,
                "Xpander extends beyond both Pivot sides",
            )
        )

        center_x_difference = abs(xpander_box.center_x - pivot_box.center_x)
        center_x_fraction = center_x_difference / max(pivot_box.width, 1)
        checks.append(
            check_max(
                "xpander_pivot_center_x_difference_over_pivot_width",
                center_x_fraction,
                config.xpander_pivot_center_x_fraction_of_pivot_width,
            )
        )

        left_extension = pivot_box.x_min - xpander_box.x_min
        right_extension = xpander_box.x_max - pivot_box.x_max
        mean_extension = max((left_extension + right_extension) / 2.0, 1e-8)
        extension_asymmetry = abs(left_extension - right_extension) / mean_extension
        checks.append(
            check_max(
                "xpander_left_right_extension_asymmetry",
                extension_asymmetry,
                config.xpander_left_right_extension_asymmetry,
            )
        )

        vertical_gap = pivot_box.y_min - xpander_box.y_max
        checks.append(
            check_range(
                "pivot_top_minus_xpander_bottom_px",
                vertical_gap,
                config.xpander_to_pivot_vertical_gap_px,
            )
        )

    cross_pair = parse_cross_centers(data)
    if cross_pair is None:
        checks.append(
            CheckResult(
                "two_cross_centers_present",
                "FAIL",
                data.get("pivot_cross_centers_px"),
                "exactly two valid cross centers",
            )
        )
    else:
        lower_cross, upper_cross = cross_pair
        lower_x, lower_y = lower_cross
        upper_x, upper_y = upper_cross

        delta_x = abs(lower_x - upper_x)
        separation_y = lower_y - upper_y

        checks.append(check_max("cross_center_delta_x_px", delta_x, config.cross_delta_x_px))
        checks.append(
            check_range(
                "cross_vertical_separation_px",
                separation_y,
                config.cross_vertical_separation_px,
            )
        )

        if pivot_box is not None:
            checks.append(
                check_range(
                    "cross_vertical_separation_over_pivot_height",
                    separation_y / max(pivot_box.height, 1),
                    config.cross_vertical_separation_over_pivot_height,
                )
            )
            mean_cross_x = 0.5 * (lower_x + upper_x)
            checks.append(
                check_max(
                    "cross_mean_x_to_pivot_center_x_px",
                    abs(mean_cross_x - pivot_box.center_x),
                    config.cross_to_pivot_center_x_px,
                )
            )

    pivot_height_difference = get_float(data, "pivot_height_difference_um")
    if pivot_height_difference is None:
        checks.append(
            CheckResult(
                "pivot_height_difference_um",
                "FAIL",
                None,
                "finite measurement",
            )
        )
    else:
        checks.append(
            check_range(
                "pivot_height_difference_um",
                pivot_height_difference,
                config.pivot_height_difference_um,
            )
        )

    radius_x = get_float(data, "xpander_radius_x_um")
    radius_y = get_float(data, "xpander_radius_y_um")

    if radius_x is None:
        checks.append(CheckResult("xpander_radius_x_um", "FAIL", None, "finite measurement"))
    else:
        checks.append(check_range("xpander_radius_x_um", radius_x, config.xpander_radius_um))

    if radius_y is None:
        checks.append(CheckResult("xpander_radius_y_um", "FAIL", None, "finite measurement"))
    else:
        checks.append(check_range("xpander_radius_y_um", radius_y, config.xpander_radius_um))

    if radius_x is not None and radius_y is not None and abs(radius_y) > 1e-8:
        checks.append(check_range("radius_x_over_radius_y", radius_x / radius_y, config.radius_ratio))

    for key in ("radius_fit_score_x", "radius_fit_score_y", "radius_fit_score_overall"):
        value = get_float(data, key)
        if value is None:
            checks.append(CheckResult(key, "FAIL", None, "finite fit score"))
        else:
            checks.append(check_min(key, value, config.radius_fit_score))

    faults = data.get("faults")
    if isinstance(faults, list) and faults:
        checks.append(
            CheckResult(
                "reported_faults",
                "WARN",
                len(faults),
                "0 reported faults",
                details=json.dumps(faults, ensure_ascii=False),
            )
        )
    else:
        checks.append(CheckResult("reported_faults", "PASS", 0, "0 reported faults"))

    final_status: Status = "PASS"
    for check in checks:
        final_status = worst_status(final_status, check.status)

    return FileReport(
        name=input_name,
        status=final_status,
        checks=checks,
        result_dir=result_dir,
    )


def print_report(report: FileReport, show_passes: bool) -> None:
    print("\n" + "=" * 118)
    print(f"PLAUSIBILITY CHECK — {report.name} — {report.status}")
    print(f"Output: {report.result_dir}")
    print("-" * 118)

    visible_checks = [
        check for check in report.checks
        if show_passes or check.status != "PASS"
    ]

    if not visible_checks:
        print("All checks passed.")
    else:
        for check in visible_checks:
            detail_suffix = f" | {check.details}" if check.details else ""
            print(
                f"[{check.status:4}] {check.name:<58} "
                f"value={format_number(check.value):<16} "
                f"expected={check.expected}{detail_suffix}"
            )

    pass_count = sum(check.status == "PASS" for check in report.checks)
    warn_count = sum(check.status == "WARN" for check in report.checks)
    fail_count = sum(check.status == "FAIL" for check in report.checks)

    print("-" * 118)
    print(
        f"Summary: PASS={pass_count}, WARN={warn_count}, FAIL={fail_count} "
        f"=> FINAL={report.status}"
    )


def save_csv(reports: list[FileReport], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "input",
                "final_status",
                "check",
                "check_status",
                "value",
                "expected",
                "details",
                "result_dir",
            ]
        )

        for report in reports:
            for check in report.checks:
                writer.writerow(
                    [
                        report.name,
                        report.status,
                        check.name,
                        check.status,
                        format_number(check.value),
                        check.expected,
                        check.details,
                        str(report.result_dir),
                    ]
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate whether saved Teramount outputs are physically/geometrically "
            "plausible using result.json and labels.npy. This is not a regression "
            "comparison against the same file's previous output."
        )
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help=(
            "Output root containing per-input folders (0/result.json, 0/labels.npy, ...) "
            "or one per-input output folder."
        ),
    )
    parser.add_argument(
        "--csv-report",
        type=Path,
        default=None,
        help="Optional path for a detailed CSV report.",
    )
    parser.add_argument(
        "--show-passes",
        action="store_true",
        help="Print PASS checks too. By default only WARN/FAIL checks are printed.",
    )
    parser.add_argument(
        "--warn-as-fail",
        action="store_true",
        help="Return a non-zero exit code when at least one WARN exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = PlausibilityConfig()

    try:
        result_dirs = discover_result_dirs(args.output_path)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2

    if not result_dirs:
        print(f"No result.json files found under {args.output_path}", file=sys.stderr)
        return 2

    reports = [
        validate_one_result_dir(result_dir, config)
        for result_dir in result_dirs
    ]

    for report in reports:
        print_report(report, show_passes=args.show_passes)

    pass_count = sum(report.status == "PASS" for report in reports)
    warn_count = sum(report.status == "WARN" for report in reports)
    fail_count = sum(report.status == "FAIL" for report in reports)

    print("\n" + "=" * 118)
    print("BATCH PLAUSIBILITY SUMMARY")
    print("-" * 118)
    print(f"Files: {len(reports)} | PASS={pass_count} | WARN={warn_count} | FAIL={fail_count}")

    if args.csv_report is not None:
        save_csv(reports, args.csv_report)
        print(f"CSV report: {args.csv_report.resolve()}")

    if fail_count > 0:
        return 1
    if args.warn_as_fail and warn_count > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
