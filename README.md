# Teramount Height-Map Analysis

This project analyzes 2-D NumPy height maps (`.npy`) and extracts the required Pivot and Xpander measurements.

## Installation

### Requirements

- Python **3.10 or newer**
- `pip`
- Recommended: a Python virtual environment

The project uses the following external Python packages:

```text
numpy
scipy
matplotlib
Pillow
```

### Ubuntu / Debian

Install Python, `pip`, and virtual-environment support if they are not already installed:

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv
```

From the project root, create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade `pip`:

```bash
python -m pip install --upgrade pip
```

If the project contains a `requirements.txt` file, install the dependencies with:

```bash
pip install -r requirements.txt
```

Otherwise, install the required packages directly:

```bash
pip install numpy scipy matplotlib Pillow
```

To verify the installation:

```bash
python -c "import numpy, scipy, matplotlib, PIL; print('Dependencies installed successfully')"
```

When opening a new terminal later, reactivate the environment before running the project:

```bash
source .venv/bin/activate
```

---

## Running the Analysis

### Run on a directory

To process all `.npy` files directly inside a directory:

```bash
python3 -B main.py <input_directory> --output-dir <output_directory>
```

Example:

```bash
python3 -B main.py \
    ./Teramount_Home_Assignment/15_files_task_1 \
    --output-dir ./outputs/task1
```

The program scans the input directory for `.npy` files, processes each file independently, and creates a separate output subdirectory for every input file.

Useful optional flags:

```bash
--print-debug True
--show-debug True
--show-corner-debug True
--summary-plot True
--pixel-size-um 0.252
```

For example:

```bash
python3 -B main.py \
    ./Teramount_Home_Assignment/15_files_task_1 \
    --output-dir ./outputs/task1 \
    --print-debug True \
    --show-debug False \
    --summary-plot False
```

---

### Run on a single file

To process one `.npy` height map:

```bash
python3 -B main.py <input_file.npy> --output-dir <output_directory>
```

Example:

```bash
python3 -B main.py \
    ./Teramount_Home_Assignment/15_files_task_2/11.npy \
    --output-dir ./outputs/task2
```

The input file is processed through the same analysis pipeline used for directory processing.

---

## Output Structure

When an output directory is supplied, results are written separately from the input files.

For example:

```text
outputs/task1/
├── 0/
│   ├── result.json
│   ├── labels.npy
│   └── labels.png
├── 1/
│   ├── result.json
│   ├── labels.npy
│   └── labels.png
├── ...
├── results.json
└── results.csv
```

Each input file receives its own subdirectory, named after the input filename without the `.npy` extension.

### Per-file outputs

#### `result.json`

Contains the numeric analysis results for one input file.

Main fields:

- `input_file_name`  
  Name of the analyzed input file.

- `status`  
  Processing status of the input.

- `pivot_bounding_box_px`  
  Four Pivot bounding-box corner points in pixel coordinates.

- `xpander_bounding_box_px`  
  Four Xpander bounding-box corner points in pixel coordinates.

The bounding-box points are stored in the following order:

```text
1. Top-left
2. Top-right
3. Bottom-right
4. Bottom-left
```

Each point contains:

```json
{
  "x": 512.0,
  "y": 772.0
}
```

- `pivot_height_difference_um`  
  Height difference between the two Pivot flat surfaces, measured from the regions surrounding the two detected crosses. The value is reported in micrometres (`μm`).

- `pivot_cross_centers_px`  
  Centers of the two detected Pivot crosses in pixel coordinates.

  The list order is:

  ```text
  1. Lower cross
  2. Upper cross
  ```

- `xpander_radius_x_um`  
  Estimated Xpander radius of curvature along the X axis, in micrometres.

- `xpander_radius_y_um`  
  Estimated Xpander radius of curvature along the Y axis, in micrometres.

- `radius_fit_score_x`  
  Fit-quality/confidence score for the X-axis curvature calculation.

- `radius_fit_score_y`  
  Fit-quality/confidence score for the Y-axis curvature calculation.

- `radius_fit_score_overall`  
  Overall curvature-fit confidence score.

- `tilt_x_deg`  
  Applied X-axis tilt correction, in degrees.

- `tilt_y_deg`  
  Applied Y-axis tilt correction, in degrees.

- `rotation_deg`  
  Applied in-plane rotation correction, in degrees.

- `faults`  
  List of detected processing or measurement faults, if any.

- `output_files`  
  Names of the files generated for this input.

#### `labels.npy`

NumPy label map with the same image dimensions as the analyzed height map.

Label convention:

```text
0 = Background
1 = Pivot
2 = Xpander
```

#### `labels.png`

RGB visualization of the label map:

```text
Black = Background
Red   = Pivot
Green = Xpander
```

---

## Aggregate Outputs

When multiple inputs are processed, the output root also contains:

### `results.json`

Contains the complete `result.json`-style result object for every processed input in one JSON array.

This includes:

- Pivot bounding-box points.
- Xpander bounding-box points.
- Pivot height difference.
- Lower and upper cross centers.
- Xpander X/Y curvature radii.
- Curvature fit scores.
- Tilt and rotation corrections.
- Fault information.
- Generated output filenames.

### `results.csv`

A flattened tabular version of the results, convenient for inspection in Excel, Python, or other analysis tools.

The CSV contains separate columns for:

- Pivot bounding-box corner coordinates.
- Xpander bounding-box corner coordinates.
- Pivot height difference.
- Lower-cross X/Y center.
- Upper-cross X/Y center.
- Xpander radius in X and Y.
- Radius fit scores.
- Tilt X/Y.
- Rotation.
- Fault information.
- Generated output filenames.

---

## Measurement Units

- X/Y coordinates and bounding-box coordinates: **pixels**
- Pixel pitch used for physical X/Y conversion: **0.252 μm/pixel** by default
- Height values: **μm**
- Pivot height difference: **μm**
- Xpander curvature radii: **μm**
- Tilt and rotation: **degrees**
