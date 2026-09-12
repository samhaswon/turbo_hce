# Turbo HCE

`turbo_hce` provides high-performance C++ implementations of the Human Correction Effort (HCE) metric algorithms, accelerating `relax_HCE` and its helper functions (`filter_bdy_cond`, `approximate_RDP`) using OpenCV.

## Features

- Native C++ implementation linking directly to minimal OpenCV static libraries (`core`, `imgproc`, `geometry`).
- Optimized with architecture-specific compiler flags (`-O3`, `-march=x86-64`, `-mavx2`, `-msse4.2`, `-flto`, `-mfma`, `-ffp-contract=fast`).
- Release of the Global Interpreter Lock (GIL) during native image processing routines to maximize multithreaded throughput.
- Comprehensive input normalization supporting standard numeric dtypes (`uint8`, `int8`, `uint16`, `int16`, `uint32`, `int32`, `uint64`, `int64`, `float16`, `float32`, `float64`, `bool`).
- IEEE 754 floating-point truth semantics: NaN skeleton values are treated as truthy matching NumPy logical operations.
- Enforced memory alignment via `NPY_ARRAY_ALIGNED` preventing undefined behavior on unaligned NumPy buffers.
- Complete `noexcept` C++ exception boundary protecting the CPython ABI against process aborts: OpenCV assertions, memory errors (`std::bad_alloc`), and sequence length errors (`std::length_error`) are safely converted into standard Python exceptions.
- Exact output match with reference Python implementation across tested dtypes, boundary edge cases, synthetic shapes, and dataset samples.
- Native C++ Zhang-Suen 2D morphological skeletonization algorithm (`turbo_hce.skeletonize`) matching `skimage.morphology.skeletonize` output bit-for-bit, delivering 11.8x to 54.8x speedup via dirty-tile frontier scheduling, AVX2 256-bit bitset evaluation, ROI cropping, and direct buffer output.
- Both original (`relax_HCE`, `approximate_RDP`) and PEP8-compliant (`relax_hce`, `approximate_rdp`) APIs.
- Portable synthetic integration tests runnable on any environment without external model or dataset assets.
- Measured 7.1x to 11.5x median speedup for `relax_HCE` and 11.8x to 54.8x median speedup for `skeletonize` across multiple image resolutions.

## System Requirements

- **Supported OS**: Linux (tested on Ubuntu 24.04 / glibc 2.39).
- **Minimum CPU Baseline**: x86-64 with AVX2, SSE4.2, and FMA support (Intel Haswell / AMD Zen 1 or newer), or ARM64 with NEON.
- **Python**: >= 3.10 with NumPy >= 1.20.
- **Build Tools**: C++17 compatible compiler (GCC >= 9 or Clang >= 10) and CMake >= 3.16.

## Installation

Build and install in editable mode:

```bash
pip install --force-reinstall -e .
```

To install optional dependencies needed for reference testing and model evaluation (`onnxruntime`, `opencv-python`, `pillow`, `scikit-image`, `tqdm`):

```bash
pip install -e ".[test]"
```

The build system automatically compiles minimal OpenCV static libraries from the included `opencv` submodule via CMake and links the C++ extension module.

## Testing

Using Python's standard `unittest` framework:

### Quick Smoke Tests (< 1s)

Runs unit tests on synthetic shapes, parameter boundaries, exception safety, and error handling:

```bash
python -m unittest test/test_smoke.py -v
```

### Dtype Differential Tests

Runs table-driven differential tests comparing Python reference vs C++ across all numeric dtypes, signed negatives, values around 128, non-contiguous views, unaligned buffers, and non-finite floats:

```bash
python -m unittest test/test_dtypes.py -v
```

### Skeletonize Equivalence Tests

Runs unit tests verifying `turbo_hce.skeletonize` bit-exact output equivalence with `skimage.morphology.skeletonize` across boundary conditions, degenerate/empty arrays, non-contiguous views, and randomized binary patterns:

```bash
python -m unittest test/test_skeletonize.py -v
```

### Full Correctness and Portable Integration Tests

Runs deterministic synthetic integration tests (no external files required) plus full end-to-end evaluation using the SUNet ONNX model and dataset images when available:

```bash
python -m unittest test/test_correctness.py -v
```

### Multi-Resolution Performance Benchmark

Runs performance benchmarking comparing Python reference vs C++ across multiple resolutions with interleaved runs and dispersion metrics (median and IQR):

```bash
python test/benchmark_resolutions.py
```

#### Benchmark Results

- **CPU**: AMD Ryzen 7 4800H with Radeon Graphics
- **Platform**: Linux x86_64 (Kernel 7.0, glibc 2.39)
- **Environment**: Python 3.12.3, NumPy 2.5.3, OpenCV 5.0.0
- **Methodology**: 10 interleaved repetitions per resolution, reporting median and interquartile range (IQR):

##### `relax_HCE` (Python vs Turbo HCE C++)

| Resolution (WxH) | Python Median (IQR) | C++ Median (IQR) | Speedup | Exact Match |
|:-----------------|:--------------------|:-----------------|:--------|:------------|
| 256x256          | 5.91 (±1.55) ms     | 0.56 (±0.15) ms  | 10.64x  | YES         |
| 512x512          | 19.86 (±1.36) ms    | 1.72 (±0.21) ms  | 11.53x  | YES         |
| 1024x1024        | 83.33 (±4.06) ms    | 7.84 (±1.21) ms  | 10.64x  | YES         |
| 1200x1799        | 173.43 (±4.94) ms   | 23.77 (±9.01) ms | 7.30x   | YES         |
| 2048x2048        | 377.86 (±87.95) ms  | 53.37 (±9.82) ms | 7.08x   | YES         |

##### `skeletonize` (`skimage.morphology.skeletonize` vs `turbo_hce.skeletonize`)

| Resolution (WxH) | skimage Median (IQR) | Turbo Median (IQR) | Speedup | Exact Match |
|:-----------------|:---------------------|:-------------------|:--------|:------------|
| 256x256          | 4.31 (±0.15) ms      | 0.36 (±0.04) ms    | 11.85x  | YES         |
| 512x512          | 36.72 (±4.55) ms     | 1.79 (±0.18) ms    | 20.47x  | YES         |
| 1024x1024        | 230.50 (±22.31) ms   | 7.37 (±0.50) ms    | 31.28x  | YES         |
| 1200x1799        | 632.46 (±25.86) ms   | 15.14 (±0.23) ms   | 41.78x  | YES         |
| 2048x2048        | 1794.23 (±28.20) ms  | 32.77 (±1.98) ms   | 54.75x  | YES         |

### Run All Tests

```bash
python -m unittest discover -s test -v
```

## Quick Start

```python
import cv2 as cv
import numpy as np
import turbo_hce

gt = cv.imread("/path/to/ground_truth.png", cv.IMREAD_GRAYSCALE)
pred = cv.imread("/path/to/prediction.png", cv.IMREAD_GRAYSCALE)
ske = turbo_hce.skeletonize(gt > 128)

fp_pts, fp_indep, fn_pts, fn_indep = turbo_hce.relax_HCE(gt, pred, ske)
print(f"FP points: {fp_pts}, FP indep: {fp_indep}, FN points: {fn_pts}, FN indep: {fn_indep}")
```

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
