from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from algorithm import analyze_height_map
from output_writer import save_batch_summary, save_result_files


def parse_bool(value: str) -> bool:
    """Parse a command-line boolean value case-insensitively."""
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"expected a boolean value, received {value!r}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Teramount analysis pipeline on one .npy file or all "
            ".npy files in a directory."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="An input .npy height-map file or a directory containing them.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for generated outputs. "
            "Default: a sibling directory named after the input path."
        ),
    )
    parser.add_argument(
        "--pixel-size-um",
        type=float,
        default=0.252,
        help="Physical size represented by one pixel, in micrometres. Default: 0.252.",
    )
    parser.add_argument(
        "--print-debug",
        type=parse_bool,
        nargs="?",
        const=True,
        default=False,
        metavar="BOOL",
        help=(
            "Print detailed diagnostics for every algorithm stage. "
            "Accepts true/false; using the option without a value means true."
        ),
    )
    parser.add_argument(
        "--show-debug",
        "--plot-debug",
        dest="show_debug",
        type=parse_bool,
        nargs="?",
        const=True,
        default=False,
        metavar="BOOL",
        help=(
            "Display diagnostic Matplotlib figures for every stage. "
            "Accepts true/false; using the option without a value means true."
        ),
    )
    parser.add_argument(
        "--summary-plot",
        "--summery-plot",
        dest="summary_plot",
        type=parse_bool,
        nargs="?",
        const=True,
        default=False,
        metavar="BOOL",
        help=(
            "Display the final analysis-summary plot. Accepts true/false; "
            "using the option without a value means true."
        ),
    )
    return parser.parse_args()


def resolve_output_dir(input_path: Path, output_dir: Path | None) -> Path:
    if output_dir is None:
        output_name = (
            f"{input_path.stem}_outputs"
            if input_path.is_file()
            else f"{input_path.name}_outputs"
        )
        output_dir = input_path.parent / output_name

    input_path = input_path.resolve()
    output_dir = output_dir.resolve()

    if output_dir == input_path:
        raise ValueError("The output directory must be different from the input path.")

    if input_path.is_dir():
        # Do not write generated files inside a batch input folder.
        try:
            output_dir.relative_to(input_path)
        except ValueError:
            pass
        else:
            raise ValueError(
                "The output directory must not be inside the input directory."
            )

    return output_dir


def find_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() == ".npy" else []
    return sorted(path for path in input_path.glob("*.npy") if path.is_file())


def load_height_map(file_path: Path) -> np.ndarray:
    data = np.load(file_path, allow_pickle=False)

    if data.ndim != 2:
        raise ValueError(
            f"{file_path.name}: expected a 2D height map, received shape {data.shape}."
        )

    if not np.issubdtype(data.dtype, np.number):
        raise TypeError(
            f"{file_path.name}: expected numeric values, received dtype {data.dtype}."
        )

    if not np.isfinite(data).all():
        raise ValueError(f"{file_path.name}: the array contains NaN or infinite values.")

    return data.astype(np.float32, copy=False)


def main() -> int:
    args = parse_args()
    input_path = args.input_path.resolve()

    if not input_path.exists():
        print(f"Input path does not exist: {input_path}", file=sys.stderr)
        return 2

    if not input_path.is_dir() and not input_path.is_file():
        print(f"Input path is not a file or directory: {input_path}", file=sys.stderr)
        return 2

    if input_path.is_file() and input_path.suffix.lower() != ".npy":
        print(f"Input file must have a .npy extension: {input_path}", file=sys.stderr)
        return 2

    try:
        output_dir = resolve_output_dir(input_path, args.output_dir)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2

    input_files = find_input_files(input_path)
    if not input_files:
        print(f"No .npy files were found at {input_path}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for input_file in input_files:
        print(f"Processing {input_file.name}...")

        try:
            height_map = load_height_map(input_file)

            # The real algorithm will be implemented later.
            raw_result = analyze_height_map(
                height_map=height_map,
                pixel_size_um=args.pixel_size_um,
                input_file_name=input_file.name,
                print_debug=args.print_debug,
                show_debug=args.show_debug,
                summary_plot=args.summary_plot,
            )

            result = save_result_files(
                input_file_name=input_file.name,
                input_shape=height_map.shape,
                raw_result=raw_result,
                output_root=output_dir,
            )
            results.append(result)

        except Exception as error:
            print(f"Failed to process {input_file.name}: {error}", file=sys.stderr)

    if not results:
        print("No input file was processed successfully.", file=sys.stderr)
        return 1

    save_batch_summary(results=results, output_root=output_dir)

    print(f"Finished. Outputs were written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
