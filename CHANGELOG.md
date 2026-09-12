# Turbo HCE 

## v0.1.0

First public release of Turbo HCE, a native C++ implementation of the relaxed
Human Correction Effort (HCE) metric for high-accuracy binary segmentation.

- Fast implementations of `relax_HCE`, `filter_bdy_cond`, and `approximate_RDP`.
- Native Zhang-Suen `skeletonize`, validated against scikit-image’s 2D behavior.
- Support for standard NumPy numeric dtypes, non-contiguous inputs, and safe
  conversion of native errors to Python exceptions.
- Original API names and PEP 8 aliases are both available.
- Linux x86_64 is tested; x86-64 builds require AVX2, SSE4.2, and FMA. ARM64
  has build support but has not yet been tested.

See the README for background on HCE and `CONTRIBUTING.md` for development,
testing, and benchmark details.
