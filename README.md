# Turbo HCE

Turbo HCE is a fast native implementation of the **Human Correction Effort**
(HCE) metric from Qin et al.'s ECCV 2022 paper, [*Highly Accurate Dichotomous
Image Segmentation*](https://arxiv.org/abs/2203.03041). It accelerates
`relax_HCE`, `filter_bdy_cond`, and `approximate_RDP` with OpenCV while keeping
the Python API familiar.

It also includes `turbo_hce.skeletonize`: a 2D Zhang-Suen skeletonizer that
matches `skimage.morphology.skeletonize` bit-for-bit in the covered tests.

## What HCE measures

DIS is binary, high-accuracy foreground segmentation: the goal is not merely to
locate a class, but to preserve the object’s outline and small structures well
enough for use in background removal, image editing, 3D reconstruction, or
human-machine interaction. A conventional overlap score can say that two masks
are similarly good while hiding a practical difference: one may take far longer
for a person to repair.

HCE estimates that repair cost as a number of clicks. It separates false-positive
and false-negative areas and models two common editing actions:

- Selecting dominant boundary points to redraw an erroneous contour.
- Selecting a self-contained erroneous region to add or remove it.

The metric counts the points needed by polygon approximation and the independent
regions that need selection. Lower is better: `0` means no correction is needed.
`relax_HCE` implements the paper’s relaxed HCE (`HCEγ`): small errors can be
ignored at a chosen tolerance, while the ground-truth skeleton keeps thin,
important structures such as cables or nets from being discarded during that
relaxation.

## Why Turbo HCE

HCE is useful precisely on detailed masks, and its contour, morphology, and
component work can become a bottleneck across a dataset. Turbo HCE keeps that
work in C++, releases the GIL while it runs, and builds only the OpenCV pieces it
needs (`core`, `imgproc`, and `geometry`). The result is a small extension with
no separate system OpenCV installation required.

The library accepts the usual NumPy numeric dtypes, including booleans and
floating-point arrays. Its truth handling follows NumPy semantics: a `NaN` in a
skeleton is truthy. Native failures, including OpenCV errors and allocation
failures, are translated to Python exceptions instead of escaping across the
CPython boundary.

## Requirements

- Linux is the tested platform (Ubuntu 24.04, x86_64).
- x86-64 builds require AVX2, SSE4.2, and FMA (Haswell / Zen 1 or newer).
- Python 3.10+ and NumPy 1.20+.
- A C++17 compiler (GCC 9+ or Clang 10+) and CMake 3.16+.

ARM64 has build support, but it has **not been tested**. Please report results
or issues if you try it.

## Quick start

```python
import cv2 as cv
import turbo_hce

ground_truth = cv.imread("/path/to/ground_truth.png", cv.IMREAD_GRAYSCALE)
prediction = cv.imread("/path/to/prediction.png", cv.IMREAD_GRAYSCALE)
skeleton = turbo_hce.skeletonize(ground_truth > 128)

fp_points, fp_indep, fn_points, fn_indep = turbo_hce.relax_HCE(
    ground_truth, prediction, skeleton
)
hce = fp_points + fp_indep + fn_points + fn_indep
print(f"HCE: {hce}")  # Lower means less estimated manual correction.
```

Both the original API names (`relax_HCE`, `approximate_RDP`) and conventional
Python aliases (`relax_hce`, `approximate_rdp`) are available.

`relax_HCE` returns the four terms separately—false-positive boundary points,
false-positive region selections, false-negative boundary points, and
false-negative region selections—so callers can see what is driving the editing
cost. The defaults, `relax=5` and `epsilon=2.0`, match the reference
implementation.

## Performance

On an AMD Ryzen 7 4800H running Linux x86_64, `relax_HCE` was 7.1–11.5× faster
than the reference implementation across 256×256 to 2048×2048 masks.
`skeletonize` was 11.8–54.8× faster than scikit-image over the same range.
These are benchmark results, not a guarantee; see [CONTRIBUTING.md](CONTRIBUTING.md)
to reproduce them and run the correctness checks.

## Contributing

Build, test, benchmark, and optional model-evaluation instructions are in
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

Licensed under the [MIT License](LICENSE).
