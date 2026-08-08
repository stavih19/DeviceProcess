from __future__ import annotations

"""Batch regression runner for the Teramount pipeline.

Typical workflow:

1) After visually approving a known-good implementation:
   python regression_runner.py ./data/task1 --baseline-dir ./regression_baselines/task1 --update-baseline

2) After every algorithm change:
   python regression_runner.py ./data/task1 --baseline-dir ./regression_baselines/task1

Exit code is non-zero when a regression FAIL is found, so the script can later be
used from pytest, pre-commit, GitHub Actions, etc.
"""

import argparse
from pathlib import Path
import sys

import numpy as np

from algorithm import analyze_height_map
from regression_checks import (
    compare_to_baseline,
    load_baseline,
    merge_reports,
    print_validation_report,
    save_baseline,
    validate_sanity,
)


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run automatic output-regression checks.")
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--pixel-size-um", type=float, default=0.252)
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--summary-plot", type=parse_bool, nargs="?", const=True, default=False)
    parser.add_argument("--print-debug", type=parse_bool, nargs="?", const=True, default=False)
    parser.add_argument("--fail-on-warning", action="store_true")
    return parser.parse_args()


def find_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() == ".npy" else []
    return sorted(path for path in input_path.glob("*.npy") if path.is_file())


def main() -> int:
    args = parse_args()
    files = find_files(args.input_path)
    if not files:
        print(f"No .npy files found at {args.input_path}", file=sys.stderr)
        return 2

    total_pass = 0
    total_warn = 0
    total_fail = 0

    for input_file in files:
        print(f"\nProcessing regression case: {input_file.name}")
        try:
            height_map = np.load(input_file, allow_pickle=False).astype(np.float32, copy=False)
            result = analyze_height_map(
                height_map=height_map,
                pixel_size_um=args.pixel_size_um,
                input_file_name=input_file.name,
                print_debug=args.print_debug,
                show_debug=False,
                show_corner_debug=False,
                summary_plot=args.summary_plot,
                show_summary_on_failure=args.summary_plot,
                raise_on_failure=False,
            )

            sanity = validate_sanity(input_file.name, result)

            if args.update_baseline:
                print_validation_report(sanity)
                if sanity.severity == "FAIL":
                    print("Baseline NOT updated because sanity checks failed.")
                    total_fail += 1
                    continue

                save_baseline(args.baseline_dir, input_file.name, result)
                print(f"Baseline updated: {args.baseline_dir / input_file.stem}")
                if sanity.severity == "WARN":
                    total_warn += 1
                else:
                    total_pass += 1
                continue

            baseline_snapshot, baseline_labels = load_baseline(
                args.baseline_dir,
                input_file.name,
            )
            regression = compare_to_baseline(
                input_file.name,
                result,
                baseline_snapshot,
                baseline_labels,
            )
            report = merge_reports(input_file.name, sanity, regression)
            print_validation_report(report)

            if report.severity == "FAIL":
                total_fail += 1
            elif report.severity == "WARN":
                total_warn += 1
            else:
                total_pass += 1

        except Exception as error:
            total_fail += 1
            print(f"[FAIL] {input_file.name}: {type(error).__name__}: {error}")

    print("\n" + "#" * 96)
    print("REGRESSION SUITE SUMMARY")
    print(f"PASS={total_pass}  WARN={total_warn}  FAIL={total_fail}")
    print("#" * 96)

    if total_fail > 0:
        return 1
    if args.fail_on_warning and total_warn > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
