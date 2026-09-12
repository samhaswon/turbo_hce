# Contributing to Turbo HCE

Thanks for helping improve Turbo HCE. This project is a Python extension backed
by C++ and a minimal, statically linked OpenCV build. Changes should preserve
the Python-facing behavior and the exactness guarantees tested in this repo.

## Set up a development environment

You need Python 3.10+, NumPy 1.20+, CMake 3.16+, and a C++17 compiler. On
x86-64, the extension is compiled for AVX2, SSE4.2, and FMA-capable CPUs.

Build and install the project in editable mode:

```bash
pip install --force-reinstall -e .
```

This command configures and builds the included `opencv` submodule as static
`core`, `imgproc`, and `geometry` libraries when they are not already present.

For the full reference and model-evaluation suite, install the optional tools:

```bash
pip install -e ".[test]"
```

That extra includes `onnxruntime`, `opencv-python`, Pillow, scikit-image, and
`tqdm`.

Linux x86_64 is the configuration we test. ARM64 has build support but has not
been tested; contributions that validate or improve it are welcome.

## Test before submitting

Tests use Python's built-in `unittest` runner.

Run the full portable suite:

```bash
python -m unittest discover -s test -v
```

Or run a focused suite while iterating:

```bash
python -m unittest test/test_smoke.py -v
python -m unittest test/test_dtypes.py -v
python -m unittest test/test_skeletonize.py -v
python -m unittest test/test_correctness.py -v
```

`test_smoke.py` covers synthetic shapes, argument boundaries, and exception
safety. `test_dtypes.py` compares the native and reference paths across numeric
dtypes, non-contiguous and unaligned arrays, and non-finite floats.
`test_skeletonize.py` checks bit-exact equivalence with scikit-image for
boundary cases and randomized binary inputs. `test_correctness.py` always runs
portable synthetic integration tests and additionally uses the model and data
below when available.

## Optional model and dataset evaluation

The end-to-end check recognizes these local assets:

- SUNet model: `/home/samuel/ai_data/skin_models/sunet/sunet.onnx`
- Input images: `/home/samuel/da/skindataset/images`
- Ground-truth masks: `/home/samuel/da/skindataset/masks`

Images and masks are paired by matching filenames. The dataset has roughly
1,300 images, so use a representative subset for routine development.

## Benchmarking

Use the benchmark script to compare the Python reference implementation with
the extension at several resolutions:

```bash
python test/benchmark_resolutions.py
```

The published numbers used 10 interleaved runs per resolution and report median
and interquartile range. On an AMD Ryzen 7 4800H, Linux x86_64, Python 3.12.3,
NumPy 2.5.3, and OpenCV 5.0.0, the results were as follows.

### `relax_HCE` (reference Python vs. Turbo HCE)

| Resolution | Python median (IQR) | Native median (IQR) | Speedup | Exact match |
| --- | ---: | ---: | ---: | --- |
| 256×256 | 5.91 (±1.55) ms | 0.56 (±0.15) ms | 10.64× | Yes |
| 512×512 | 19.86 (±1.36) ms | 1.72 (±0.21) ms | 11.53× | Yes |
| 1024×1024 | 83.33 (±4.06) ms | 7.84 (±1.21) ms | 10.64× | Yes |
| 1200×1799 | 173.43 (±4.94) ms | 23.77 (±9.01) ms | 7.30× | Yes |
| 2048×2048 | 377.86 (±87.95) ms | 53.37 (±9.82) ms | 7.08× | Yes |

### `skeletonize` (scikit-image vs. Turbo HCE)

| Resolution | scikit-image median (IQR) | Turbo median (IQR) | Speedup | Exact match |
| --- | ---: | ---: | ---: | --- |
| 256×256 | 4.31 (±0.15) ms | 0.36 (±0.04) ms | 11.85× | Yes |
| 512×512 | 36.72 (±4.55) ms | 1.79 (±0.18) ms | 20.47× | Yes |
| 1024×1024 | 230.50 (±22.31) ms | 7.37 (±0.50) ms | 31.28× | Yes |
| 1200×1799 | 632.46 (±25.86) ms | 15.14 (±0.23) ms | 41.78× | Yes |
| 2048×2048 | 1794.23 (±28.20) ms | 32.77 (±1.98) ms | 54.75× | Yes |

Performance is hardware-dependent. Include the CPU, OS, Python/NumPy versions,
and benchmark method when sharing new measurements.
