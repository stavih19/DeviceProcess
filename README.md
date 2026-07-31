# Teramount assignment — initial setup

This is a setup-only pipeline for:

1. Reading every `.npy` height map from a directory.
2. Calling one analysis function for each input.
3. Returning a structured result object.
4. Saving all generated files for each input in a dedicated output folder.
5. Saving aggregate `results.json` and `results.csv` files.

The actual Pivot/Xpander analysis is not implemented yet.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Pass the input directory as a positional parameter:

```bash
python main.py ./data/task1
```

Optionally specify a separate output directory:

```bash
python main.py ./data/task1 --output-dir ./outputs/task1
```

Optionally change the physical pixel size:

```bash
python main.py ./data/task1 --pixel-size-um 0.252
```

If `--output-dir` is omitted and the input directory is `data/task1`, the
default output directory is the sibling folder:

```text
data/task1_outputs/
```

The output directory is not allowed to be the input directory or a directory
inside it.

## Output structure

For input files `0.npy` and `1.npy`:

```text
task1_outputs/
├── results.csv
├── results.json
├── 0/
│   ├── result.json
│   ├── labels.npy
│   └── labels.png
└── 1/
    ├── result.json
    ├── labels.npy
    └── labels.png
```

Label convention:

```text
0 = background
1 = Pivot
2 = Xpander
```

`labels.npy` stores exact labels.  
`labels.png` is an RGB preview.

## Result objects

`RawAnalysisResult` is returned by the future algorithm and may contain NumPy
arrays.

`AnalysisResult` is the final lightweight result. It contains all scalar/list
measurements, but for generated files it stores only the file names:

- `result.json`
- `labels.npy`
- `labels.png`

## Implementing the algorithm

Implement the body of:

```python
analyze_height_map(...)
```

inside `algorithm.py`.
