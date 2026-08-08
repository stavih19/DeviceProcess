from __future__ import annotations

"""Regression and sanity checks for the Teramount height-map pipeline.

Two complementary layers are implemented here:

1. Dataset sanity checks
   Broad geometry/measurement limits learned from the supplied samples. These
   catch obviously broken detections even when no baseline exists yet.

2. Golden-baseline regression checks
   Compare the current result of a specific input file against a previously
   approved result for that same file. This is the important layer for detecting
   regressions after algorithm changes.

The module does not depend on the internal stage-result classes. It validates the
public RawAnalysisResult and the final label map, so it can remain stable while
individual stages are refactored.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal
import json
import math

import numpy as np


Severity = Literal["PASS", "WARN", "FAIL"]


@dataclass(frozen=True)
class CheckLimits:
    """Broad dataset-level limits.

    WARN limits are intentionally close to the envelope seen in the known-good
    files. FAIL limits are wider so that a real damaged component is not too
    easily confused with an algorithmic crash.
    """

    # Pivot geometry derived from label 1.
    pivot_width_warn: tuple[int, int] = (255, 320)
    pivot_width_fail: tuple[int, int] = (220, 360)
    pivot_height_warn: tuple[int, int] = (178, 198)
    pivot_height_fail: tuple[int, int] = (150, 230)

    # Xpander geometry derived from label 2.
    xpander_width_warn: tuple[int, int] = (475, 515)
    xpander_width_fail: tuple[int, int] = (420, 580)
    xpander_height_warn: tuple[int, int] = (480, 525)
    xpander_height_fail: tuple[int, int] = (420, 600)

    # Relative geometry is preferable to absolute image coordinates because the
    # assignment explicitly says the channel may not be centred.
    xpander_to_pivot_width_ratio_warn: tuple[float, float] = (1.55, 1.90)
    xpander_to_pivot_width_ratio_fail: tuple[float, float] = (1.35, 2.20)
    xpander_to_pivot_height_ratio_warn: tuple[float, float] = (2.50, 2.85)
    xpander_to_pivot_height_ratio_fail: tuple[float, float] = (2.20, 3.20)

    # Two cross centres should be almost on the same vertical axis.
    cross_x_difference_warn_px: float = 3.0
    cross_x_difference_fail_px: float = 12.0
    cross_y_separation_warn_px: tuple[float, float] = (108.0, 125.0)
    cross_y_separation_fail_px: tuple[float, float] = (80.0, 160.0)

    # Measurements observed around 18.0-18.3 um and R around 810-845 um.
    pivot_height_difference_warn_um: tuple[float, float] = (17.8, 18.5)
    pivot_height_difference_fail_um: tuple[float, float] = (16.5, 20.0)
    radius_warn_um: tuple[float, float] = (790.0, 875.0)
    radius_fail_um: tuple[float, float] = (700.0, 1000.0)
    radius_ratio_warn: tuple[float, float] = (0.96, 1.04)
    radius_ratio_fail: tuple[float, float] = (0.90, 1.10)

    # 6.npy was a valid but harder case with fit confidence around 0.937.
    fit_score_warn_min: float = 0.93
    fit_score_fail_min: float = 0.85

    # A class must not collapse to a tiny number of pixels.
    min_label_pixels_fail: int = 1000


@dataclass(frozen=True)
class RegressionLimits:
    """Tolerance against a previously approved result for the same input."""

    cross_center_warn_px: float = 3.0
    cross_center_fail_px: float = 6.0

    height_difference_warn_um: float = 0.10
    height_difference_fail_um: float = 0.25

    radius_relative_warn: float = 0.015
    radius_relative_fail: float = 0.03

    fit_score_drop_warn: float = 0.03
    fit_score_drop_fail: float = 0.08

    # Current Task2->Task1 comparisons were roughly 0.973-0.993 for Pivot and
    # 0.976-0.982 for Xpander, so these leave useful margin for small edge shifts.
    pivot_iou_warn_min: float = 0.96
    pivot_iou_fail_min: float = 0.92
    xpander_iou_warn_min: float = 0.96
    xpander_iou_fail_min: float = 0.92

    label_area_relative_warn: float = 0.05
    label_area_relative_fail: float = 0.10


@dataclass(frozen=True)
class CheckItem:
    name: str
    severity: Severity
    value: Any
    expected: str
    message: str = ""


@dataclass
class ValidationReport:
    file_name: str
    checks: list[CheckItem] = field(default_factory=list)

    @property
    def severity(self) -> Severity:
        if any(item.severity == "FAIL" for item in self.checks):
            return "FAIL"
        if any(item.severity == "WARN" for item in self.checks):
            return "WARN"
        return "PASS"

    @property
    def failures(self) -> list[CheckItem]:
        return [item for item in self.checks if item.severity == "FAIL"]

    @property
    def warnings(self) -> list[CheckItem]:
        return [item for item in self.checks if item.severity == "WARN"]


@dataclass(frozen=True)
class ResultSnapshot:
    status: str
    pivot_height_difference_um: float | None
    lower_cross_xy: tuple[float, float] | None
    upper_cross_xy: tuple[float, float] | None
    xpander_radius_x_um: float | None
    xpander_radius_y_um: float | None
    radius_fit_score_x: float | None
    radius_fit_score_y: float | None
    radius_fit_score_overall: float | None
    pivot_bbox: tuple[int, int, int, int] | None
    xpander_bbox: tuple[int, int, int, int] | None
    pivot_pixels: int
    xpander_pixels: int


def build_snapshot(raw_result: Any) -> ResultSnapshot:
    """Extract a stable, serialisable regression snapshot from RawAnalysisResult."""
    label_map = _validated_label_map(getattr(raw_result, "label_map", None))

    cross_points = []
    for point in getattr(raw_result, "pivot_cross_centers_px", []) or []:
        x_value = _finite_or_none(getattr(point, "x", None))
        y_value = _finite_or_none(getattr(point, "y", None))
        if x_value is not None and y_value is not None:
            cross_points.append((x_value, y_value))

    # Do not trust list ordering. Upper has the smaller Y, lower the larger Y.
    lower_cross: tuple[float, float] | None = None
    upper_cross: tuple[float, float] | None = None
    if len(cross_points) >= 2:
        ordered = sorted(cross_points[:2], key=lambda item: item[1])
        upper_cross = ordered[0]
        lower_cross = ordered[1]

    pivot_mask = label_map == 1
    xpander_mask = label_map == 2

    return ResultSnapshot(
        status=str(getattr(raw_result, "status", "unknown")),
        pivot_height_difference_um=_finite_or_none(
            getattr(raw_result, "pivot_height_difference_um", None)
        ),
        lower_cross_xy=lower_cross,
        upper_cross_xy=upper_cross,
        xpander_radius_x_um=_finite_or_none(
            getattr(raw_result, "xpander_radius_x_um", None)
        ),
        xpander_radius_y_um=_finite_or_none(
            getattr(raw_result, "xpander_radius_y_um", None)
        ),
        radius_fit_score_x=_finite_or_none(
            getattr(raw_result, "radius_fit_score_x", None)
        ),
        radius_fit_score_y=_finite_or_none(
            getattr(raw_result, "radius_fit_score_y", None)
        ),
        radius_fit_score_overall=_finite_or_none(
            getattr(raw_result, "radius_fit_score_overall", None)
        ),
        pivot_bbox=_bbox_from_mask(pivot_mask),
        xpander_bbox=_bbox_from_mask(xpander_mask),
        pivot_pixels=int(np.count_nonzero(pivot_mask)),
        xpander_pixels=int(np.count_nonzero(xpander_mask)),
    )


def validate_sanity(
    file_name: str,
    raw_result: Any,
    limits: CheckLimits | None = None,
) -> ValidationReport:
    """Run broad invariant/range checks without requiring a stored baseline."""
    limits = limits or CheckLimits()
    snapshot = build_snapshot(raw_result)
    report = ValidationReport(file_name=file_name)

    if "failed" in snapshot.status.lower():
        report.checks.append(
            CheckItem(
                "pipeline_status",
                "FAIL",
                snapshot.status,
                "pipeline completes",
                "A sequential stage failed.",
            )
        )
    else:
        report.checks.append(
            CheckItem("pipeline_status", "PASS", snapshot.status, "pipeline completes")
        )

    _check_bbox_dimension(
        report, "pivot_width", snapshot.pivot_bbox, axis="width",
        warn_range=limits.pivot_width_warn, fail_range=limits.pivot_width_fail,
    )
    _check_bbox_dimension(
        report, "pivot_height", snapshot.pivot_bbox, axis="height",
        warn_range=limits.pivot_height_warn, fail_range=limits.pivot_height_fail,
    )
    _check_bbox_dimension(
        report, "xpander_width", snapshot.xpander_bbox, axis="width",
        warn_range=limits.xpander_width_warn, fail_range=limits.xpander_width_fail,
    )
    _check_bbox_dimension(
        report, "xpander_height", snapshot.xpander_bbox, axis="height",
        warn_range=limits.xpander_height_warn, fail_range=limits.xpander_height_fail,
    )

    if snapshot.pivot_bbox is not None and snapshot.xpander_bbox is not None:
        pivot_width, pivot_height = _bbox_size(snapshot.pivot_bbox)
        xpander_width, xpander_height = _bbox_size(snapshot.xpander_bbox)
        _check_range(
            report,
            "xpander_to_pivot_width_ratio",
            xpander_width / max(pivot_width, 1),
            limits.xpander_to_pivot_width_ratio_warn,
            limits.xpander_to_pivot_width_ratio_fail,
        )
        _check_range(
            report,
            "xpander_to_pivot_height_ratio",
            xpander_height / max(pivot_height, 1),
            limits.xpander_to_pivot_height_ratio_warn,
            limits.xpander_to_pivot_height_ratio_fail,
        )

    if snapshot.upper_cross_xy is None or snapshot.lower_cross_xy is None:
        report.checks.append(
            CheckItem(
                "cross_centers",
                "FAIL",
                None,
                "two detected cross centres",
            )
        )
    else:
        cross_dx = abs(snapshot.upper_cross_xy[0] - snapshot.lower_cross_xy[0])
        cross_dy = snapshot.lower_cross_xy[1] - snapshot.upper_cross_xy[1]
        _check_upper_limit(
            report,
            "cross_x_difference_px",
            cross_dx,
            limits.cross_x_difference_warn_px,
            limits.cross_x_difference_fail_px,
        )
        _check_range(
            report,
            "cross_y_separation_px",
            cross_dy,
            limits.cross_y_separation_warn_px,
            limits.cross_y_separation_fail_px,
        )

    _check_optional_range(
        report,
        "pivot_height_difference_um",
        snapshot.pivot_height_difference_um,
        limits.pivot_height_difference_warn_um,
        limits.pivot_height_difference_fail_um,
    )
    _check_optional_range(
        report,
        "xpander_radius_x_um",
        snapshot.xpander_radius_x_um,
        limits.radius_warn_um,
        limits.radius_fail_um,
    )
    _check_optional_range(
        report,
        "xpander_radius_y_um",
        snapshot.xpander_radius_y_um,
        limits.radius_warn_um,
        limits.radius_fail_um,
    )

    if snapshot.xpander_radius_x_um is not None and snapshot.xpander_radius_y_um is not None:
        radius_ratio = snapshot.xpander_radius_x_um / max(snapshot.xpander_radius_y_um, 1e-8)
        _check_range(
            report,
            "radius_x_over_y",
            radius_ratio,
            limits.radius_ratio_warn,
            limits.radius_ratio_fail,
        )

    for name, value in (
        ("radius_fit_score_x", snapshot.radius_fit_score_x),
        ("radius_fit_score_y", snapshot.radius_fit_score_y),
        ("radius_fit_score_overall", snapshot.radius_fit_score_overall),
    ):
        if value is None:
            report.checks.append(CheckItem(name, "FAIL", None, "finite fit score"))
        elif value < limits.fit_score_fail_min:
            report.checks.append(
                CheckItem(name, "FAIL", value, f">= {limits.fit_score_fail_min:.3f}")
            )
        elif value < limits.fit_score_warn_min:
            report.checks.append(
                CheckItem(name, "WARN", value, f">= {limits.fit_score_warn_min:.3f}")
            )
        else:
            report.checks.append(
                CheckItem(name, "PASS", value, f">= {limits.fit_score_warn_min:.3f}")
            )

    for label_name, pixels in (
        ("pivot_pixels", snapshot.pivot_pixels),
        ("xpander_pixels", snapshot.xpander_pixels),
    ):
        severity: Severity = "PASS" if pixels >= limits.min_label_pixels_fail else "FAIL"
        report.checks.append(
            CheckItem(label_name, severity, pixels, f">= {limits.min_label_pixels_fail}")
        )

    return report


def compare_to_baseline(
    file_name: str,
    raw_result: Any,
    baseline_snapshot: ResultSnapshot,
    baseline_label_map: np.ndarray,
    limits: RegressionLimits | None = None,
) -> ValidationReport:
    """Compare current output with an approved golden result for this same file."""
    limits = limits or RegressionLimits()
    current = build_snapshot(raw_result)
    current_labels = _validated_label_map(getattr(raw_result, "label_map", None))
    baseline_labels = _validated_label_map(baseline_label_map)

    report = ValidationReport(file_name=file_name)

    if "failed" in current.status.lower():
        report.checks.append(
            CheckItem("pipeline_status", "FAIL", current.status, baseline_snapshot.status)
        )
    else:
        report.checks.append(
            CheckItem("pipeline_status", "PASS", current.status, baseline_snapshot.status)
        )

    for name, current_xy, baseline_xy in (
        ("lower_cross_center", current.lower_cross_xy, baseline_snapshot.lower_cross_xy),
        ("upper_cross_center", current.upper_cross_xy, baseline_snapshot.upper_cross_xy),
    ):
        if current_xy is None or baseline_xy is None:
            report.checks.append(CheckItem(name, "FAIL", current_xy, str(baseline_xy)))
            continue
        distance = math.hypot(current_xy[0] - baseline_xy[0], current_xy[1] - baseline_xy[1])
        _check_delta(
            report,
            name + "_distance_px",
            distance,
            limits.cross_center_warn_px,
            limits.cross_center_fail_px,
        )

    _compare_absolute_delta(
        report,
        "pivot_height_difference_delta_um",
        current.pivot_height_difference_um,
        baseline_snapshot.pivot_height_difference_um,
        limits.height_difference_warn_um,
        limits.height_difference_fail_um,
    )

    for name, current_radius, baseline_radius in (
        ("xpander_radius_x_relative_delta", current.xpander_radius_x_um, baseline_snapshot.xpander_radius_x_um),
        ("xpander_radius_y_relative_delta", current.xpander_radius_y_um, baseline_snapshot.xpander_radius_y_um),
    ):
        _compare_relative_delta(
            report,
            name,
            current_radius,
            baseline_radius,
            limits.radius_relative_warn,
            limits.radius_relative_fail,
        )

    for name, current_score, baseline_score in (
        ("fit_score_x_drop", current.radius_fit_score_x, baseline_snapshot.radius_fit_score_x),
        ("fit_score_y_drop", current.radius_fit_score_y, baseline_snapshot.radius_fit_score_y),
        ("fit_score_overall_drop", current.radius_fit_score_overall, baseline_snapshot.radius_fit_score_overall),
    ):
        if current_score is None or baseline_score is None:
            report.checks.append(CheckItem(name, "FAIL", current_score, str(baseline_score)))
            continue
        drop = baseline_score - current_score
        _check_delta(
            report,
            name,
            max(0.0, drop),
            limits.fit_score_drop_warn,
            limits.fit_score_drop_fail,
        )

    if current_labels.shape != baseline_labels.shape:
        report.checks.append(
            CheckItem(
                "label_shape",
                "FAIL",
                current_labels.shape,
                str(baseline_labels.shape),
            )
        )
        return report

    for label_value, name, warn_min, fail_min in (
        (1, "pivot_iou", limits.pivot_iou_warn_min, limits.pivot_iou_fail_min),
        (2, "xpander_iou", limits.xpander_iou_warn_min, limits.xpander_iou_fail_min),
    ):
        score = _class_iou(current_labels, baseline_labels, label_value)
        if score < fail_min:
            severity: Severity = "FAIL"
        elif score < warn_min:
            severity = "WARN"
        else:
            severity = "PASS"
        report.checks.append(
            CheckItem(name, severity, score, f">= {warn_min:.3f} (fail < {fail_min:.3f})")
        )

        current_pixels = int(np.count_nonzero(current_labels == label_value))
        baseline_pixels = int(np.count_nonzero(baseline_labels == label_value))
        area_delta = abs(current_pixels - baseline_pixels) / max(baseline_pixels, 1)
        _check_delta(
            report,
            name.replace("_iou", "_area_relative_delta"),
            area_delta,
            limits.label_area_relative_warn,
            limits.label_area_relative_fail,
        )

    return report


def save_baseline(
    baseline_dir: str | Path,
    file_name: str,
    raw_result: Any,
) -> None:
    root = Path(baseline_dir) / Path(file_name).stem
    root.mkdir(parents=True, exist_ok=True)

    snapshot = build_snapshot(raw_result)
    with (root / "snapshot.json").open("w", encoding="utf-8") as file:
        json.dump(asdict(snapshot), file, indent=2, ensure_ascii=False)

    label_map = _validated_label_map(getattr(raw_result, "label_map", None))
    np.save(root / "labels.npy", label_map, allow_pickle=False)


def load_baseline(
    baseline_dir: str | Path,
    file_name: str,
) -> tuple[ResultSnapshot, np.ndarray]:
    root = Path(baseline_dir) / Path(file_name).stem
    snapshot_path = root / "snapshot.json"
    labels_path = root / "labels.npy"

    if not snapshot_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"No baseline exists for {file_name!r} under {root}. "
            "Run the regression runner once with --update-baseline after visually "
            "approving the current outputs."
        )

    with snapshot_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    for key in ("lower_cross_xy", "upper_cross_xy", "pivot_bbox", "xpander_bbox"):
        if data.get(key) is not None:
            data[key] = tuple(data[key])

    snapshot = ResultSnapshot(**data)
    labels = np.load(labels_path, allow_pickle=False)
    return snapshot, labels


def print_validation_report(report: ValidationReport) -> None:
    print("\n" + "=" * 96)
    print(f"REGRESSION CHECK — {report.file_name} — {report.severity}")
    print("=" * 96)

    for item in report.checks:
        marker = {"PASS": "OK  ", "WARN": "WARN", "FAIL": "FAIL"}[item.severity]
        value = _format_value(item.value)
        suffix = f" | {item.message}" if item.message else ""
        print(f"[{marker}] {item.name:<38} value={value:<16} expected {item.expected}{suffix}")

    print("-" * 96)
    print(
        f"Summary: {len(report.failures)} failure(s), "
        f"{len(report.warnings)} warning(s), final={report.severity}"
    )
    print("=" * 96)


def merge_reports(file_name: str, *reports: ValidationReport) -> ValidationReport:
    merged = ValidationReport(file_name=file_name)
    for report in reports:
        merged.checks.extend(report.checks)
    return merged


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validated_label_map(label_map: Any) -> np.ndarray:
    if label_map is None:
        raise ValueError("Regression checks require a label_map.")
    array = np.asarray(label_map)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D label map, received shape {array.shape}.")
    return array


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    y, x = np.nonzero(mask)
    if x.size == 0:
        return None
    return int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1


def _bbox_size(box: tuple[int, int, int, int]) -> tuple[int, int]:
    x_min, y_min, x_max, y_max = box
    return x_max - x_min, y_max - y_min


def _check_bbox_dimension(
    report: ValidationReport,
    name: str,
    box: tuple[int, int, int, int] | None,
    axis: Literal["width", "height"],
    warn_range: tuple[float, float],
    fail_range: tuple[float, float],
) -> None:
    if box is None:
        report.checks.append(CheckItem(name, "FAIL", None, str(warn_range)))
        return
    width, height = _bbox_size(box)
    _check_range(report, name, width if axis == "width" else height, warn_range, fail_range)


def _check_optional_range(
    report: ValidationReport,
    name: str,
    value: float | None,
    warn_range: tuple[float, float],
    fail_range: tuple[float, float],
) -> None:
    if value is None:
        report.checks.append(CheckItem(name, "FAIL", None, str(warn_range)))
        return
    _check_range(report, name, value, warn_range, fail_range)


def _check_range(
    report: ValidationReport,
    name: str,
    value: float,
    warn_range: tuple[float, float],
    fail_range: tuple[float, float],
) -> None:
    fail_low, fail_high = fail_range
    warn_low, warn_high = warn_range
    if value < fail_low or value > fail_high:
        severity: Severity = "FAIL"
    elif value < warn_low or value > warn_high:
        severity = "WARN"
    else:
        severity = "PASS"
    report.checks.append(
        CheckItem(
            name,
            severity,
            value,
            f"{warn_low:g}..{warn_high:g} (fail outside {fail_low:g}..{fail_high:g})",
        )
    )


def _check_upper_limit(
    report: ValidationReport,
    name: str,
    value: float,
    warn_limit: float,
    fail_limit: float,
) -> None:
    if value > fail_limit:
        severity: Severity = "FAIL"
    elif value > warn_limit:
        severity = "WARN"
    else:
        severity = "PASS"
    report.checks.append(
        CheckItem(name, severity, value, f"<= {warn_limit:g} (fail > {fail_limit:g})")
    )


def _check_delta(
    report: ValidationReport,
    name: str,
    delta: float,
    warn_limit: float,
    fail_limit: float,
) -> None:
    _check_upper_limit(report, name, abs(delta), warn_limit, fail_limit)


def _compare_absolute_delta(
    report: ValidationReport,
    name: str,
    current: float | None,
    baseline: float | None,
    warn_limit: float,
    fail_limit: float,
) -> None:
    if current is None or baseline is None:
        report.checks.append(CheckItem(name, "FAIL", current, str(baseline)))
        return
    _check_delta(report, name, current - baseline, warn_limit, fail_limit)


def _compare_relative_delta(
    report: ValidationReport,
    name: str,
    current: float | None,
    baseline: float | None,
    warn_limit: float,
    fail_limit: float,
) -> None:
    if current is None or baseline is None or abs(baseline) < 1e-8:
        report.checks.append(CheckItem(name, "FAIL", current, str(baseline)))
        return
    relative = abs(current - baseline) / abs(baseline)
    _check_delta(report, name, relative, warn_limit, fail_limit)


def _class_iou(current: np.ndarray, baseline: np.ndarray, label_value: int) -> float:
    current_mask = current == label_value
    baseline_mask = baseline == label_value
    union = int(np.count_nonzero(current_mask | baseline_mask))
    if union == 0:
        return 1.0
    intersection = int(np.count_nonzero(current_mask & baseline_mask))
    return intersection / union


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)
