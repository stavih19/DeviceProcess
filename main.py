from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from algorithm import analyze_height_map
from output_writer import save_batch_summary, save_result_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Teramount analysis pipeline on all .npy files in a directory."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing input .npy height-map files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for generated outputs. "
            "Default: a sibling directory named <input_dir_name>_outputs."
        ),
    )
    parser.add_argument(
        "--pixel-size-um",
        type=float,
        default=0.252,
        help="Physical size represented by one pixel, in micrometres. Default: 0.252.",
    )
    return parser.parse_args()


def resolve_output_dir(input_dir: Path, output_dir: Path | None) -> Path:
    if output_dir is None:
        output_dir = input_dir.parent / f"{input_dir.name}_outputs"

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    if output_dir == input_dir:
        raise ValueError("The output directory must be different from the input directory.")

    # Do not write generated files inside the input folder.
    try:
        output_dir.relative_to(input_dir)
    except ValueError:
        pass
    else:
        raise ValueError("The output directory must not be inside the input directory.")

    return output_dir


def find_input_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.glob("*.npy") if path.is_file())


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
    input_dir = args.input_dir.resolve()

    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 2

    if not input_dir.is_dir():
        print(f"Input path is not a directory: {input_dir}", file=sys.stderr)
        return 2

    try:
        output_dir = resolve_output_dir(input_dir, args.output_dir)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2

    input_files = find_input_files(input_dir)
    if not input_files:
        print(f"No .npy files were found in {input_dir}", file=sys.stderr)
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
