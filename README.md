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
- Native C++ Zhang-Suen 2D morphological skeletonization algorithm (`turbo_hce.skeletonize`) matching `skimage.morphology.skeletonize` output bit-for-bit, delivering 6.4x to 17.5x speedup via AVX2 256-bit bitset evaluation, ROI cropping, O(1) bound contraction, and direct buffer output.
- Both original (`relax_HCE`, `approximate_RDP`) and PEP8-compliant (`relax_hce`, `approximate_rdp`) APIs.
- Portable synthetic integration tests runnable on any environment without external model or dataset assets.
- Measured 6.1x to 12.9x median speedup for `relax_HCE` and 6.4x to 17.5x median speedup for `skeletonize` across multiple image resolutions.

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
| 256x256          | 6.35 (±1.57) ms     | 0.60 (±0.21) ms  | 10.62x  | YES         |
| 512x512          | 22.10 (±1.58) ms    | 1.71 (±0.46) ms  | 12.94x  | YES         |
| 1024x1024        | 92.96 (±2.21) ms    | 8.41 (±1.31) ms  | 11.06x  | YES         |
| 1200x1799        | 192.86 (±7.41) ms   | 29.57 (±5.81) ms | 6.52x   | YES         |
| 2048x2048        | 438.15 (±127.59) ms | 65.87 (±11.25) ms| 6.65x   | YES         |

##### `skeletonize` (`skimage.morphology.skeletonize` vs `turbo_hce.skeletonize`)

| Resolution (WxH) | skimage Median (IQR) | Turbo Median (IQR) | Speedup | Exact Match |
|:-----------------|:---------------------|:-------------------|:--------|:------------|
| 256x256          | 4.41 (±0.17) ms      | 0.68 (±0.02) ms    | 6.44x   | YES         |
| 512x512          | 31.05 (±1.01) ms     | 3.23 (±0.10) ms    | 9.62x   | YES         |
| 1024x1024        | 207.97 (±3.25) ms    | 15.21 (±0.28) ms   | 13.67x  | YES         |
| 1200x1799        | 678.04 (±34.78) ms   | 38.82 (±1.94) ms   | 17.47x  | YES         |
| 2048x2048        | 1915.20 (±122.43) ms | 160.29 (±56.38) ms | 11.95x  | YES         |

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
