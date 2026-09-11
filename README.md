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
- Native C++ Zhang-Suen 2D morphological skeletonization algorithm (`turbo_hce.skeletonize`) matching `skimage.morphology.skeletonize` output bit-for-bit.
- Both original (`relax_HCE`, `approximate_RDP`) and PEP8-compliant (`relax_hce`, `approximate_rdp`) APIs.
- Portable synthetic integration tests runnable on any environment without external model or dataset assets.
- Measured 6.1x to 12.7x median speedup for relax_HCE across multiple image resolutions.

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
| 256x256          | 7.00 (±0.94) ms     | 0.55 (±0.10) ms  | 12.65x  | YES         |
| 512x512          | 20.10 (±1.26) ms    | 1.59 (±0.11) ms  | 12.62x  | YES         |
| 1024x1024        | 91.49 (±3.68) ms    | 9.10 (±1.53) ms  | 10.05x  | YES         |
| 1200x1799        | 182.46 (±5.28) ms   | 26.97 (±7.15) ms | 6.76x   | YES         |
| 2048x2048        | 395.99 (±41.86) ms  | 64.62 (±7.94) ms | 6.13x   | YES         |

##### `skeletonize` (`skimage.morphology.skeletonize` vs `turbo_hce.skeletonize`)

| Resolution (WxH) | skimage Median (IQR) | Turbo Median (IQR) | Speedup | Exact Match |
|:-----------------|:---------------------|:-------------------|:--------|:------------|
| 256x256          | 4.39 (±0.12) ms      | 4.39 (±0.15) ms    | 1.00x   | YES         |
| 512x512          | 31.39 (±0.80) ms     | 33.99 (±3.16) ms   | 0.92x   | YES         |
| 1024x1024        | 234.70 (±24.86) ms   | 268.82 (±46.38) ms | 0.87x   | YES         |
| 1200x1799        | 841.31 (±28.70) ms   | 872.34 (±40.82) ms | 0.96x   | YES         |
| 2048x2048        | 1837.32 (±27.38) ms  | 1867.78 (±160.48) ms| 0.98x  | YES         |

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
