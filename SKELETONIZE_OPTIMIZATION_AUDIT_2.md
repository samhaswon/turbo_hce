# Skeletonize optimization audit, second pass

## Scope and executive summary

This is a read-only optimization audit of the implementation at commit
`2d8369d4c77a255ac00a103a36612e66429a6d8c`. No implementation, test, or build
file was changed. The required benchmark and the existing skeletonization unit
tests were run against the installed in-place extension.

The current implementation is already fast on the supplied real mask: it is
6.70x to 16.94x faster than scikit-image and was bit-exact at every benchmark
resolution. The best remaining opportunity is not another isolated SIMD
instruction change. It is reducing how much of the image is revisited after
each synchronous deletion pass. The current loop scans every row and 32-pixel
chunk in one global active rectangle, even though only pixels near a deletion
can change eligibility. This is especially expensive for thick objects and at
large resolutions.

The recommendations, in order, are:

1. Prototype an exact per-subpass frontier or dirty-tile scheduler. This is the
   highest-upside redesign.
2. Make the deletion queue compact and update row/column metadata when an item
   is queued. This is a comparatively easy memory-traffic win.
3. Stop zeroing output pixels that the final ROI copy will overwrite, and
   contract bounds after each subpass. These are low-risk incremental wins.
4. Separate/vectorize ROI normalization and column-count construction instead
   of running a scalar `tzcnt` loop once per foreground pixel.
5. Only after reducing work, revisit the nine overlapping neighbor loads,
   padded-row alignment, and optional large-image multithreading.

Claims labeled **Measured** below are direct observations from this checkout.
Claims labeled **Inference** are hypotheses that need the listed experiments.

## Required benchmark

Command run exactly through the project virtual environment:

```text
.venv/bin/python test/benchmark_resolutions.py
```

Environment reported by the benchmark or queried in the same environment:

- Date: 2026-09-12 01:02:25 UTC
- CPU: AMD Ryzen 7 4800H with Radeon Graphics, 8 cores / 16 threads
- Cache: 8 x 32 KiB L1d, 8 x 512 KiB L2, 2 x 4 MiB L3
- OS: Linux 7.0.0-31-generic, x86-64, glibc 2.39
- Python 3.12.3; NumPy 2.5.3; OpenCV 5.0.0; scikit-image 0.26.0
- GCC/G++ 13.3.0
- Git commit: `2d8369d`
- Data: real skin dataset sample `00001.png`
- Method: 10 repetitions in interleaved implementation order, reporting the
  median and IQR

### Skeletonize results

| Resolution | scikit-image median (IQR) | Turbo median (IQR) | Speedup | Exact match |
|---|---:|---:|---:|:---:|
| 256x256 | 4.21 (0.03) ms | 0.63 (0.02) ms | 6.70x | yes |
| 512x512 | 30.83 (0.99) ms | 3.24 (0.08) ms | 9.52x | yes |
| 1024x1024 | 209.27 (10.74) ms | 15.21 (0.24) ms | 13.76x | yes |
| 1200x1799 | 667.41 (58.07) ms | 39.40 (2.59) ms | 16.94x | yes |
| 2048x2048 | 1950.07 (95.83) ms | 189.84 (53.07) ms | 10.27x | yes |

**Measured:** all five warmup comparisons were exact. The 2048x2048 Turbo
median was 4.82 times the 1200x1799 median for only 1.94 times as many pixels,
and its 53.07 ms IQR was 28% of its median. Some of this may be system noise,
frequency scaling, or allocator/cache effects; one ten-sample run cannot assign
the cause.

The same required script also reported the following `relax_HCE` results. They
are included for completeness but were not used to rank skeletonization work.

| Resolution | Python median (IQR) | C++ median (IQR) | Speedup | Exact match |
|---|---:|---:|---:|:---:|
| 256x256 | 5.46 (0.42) ms | 0.51 (0.09) ms | 10.61x | yes |
| 512x512 | 20.05 (0.60) ms | 1.71 (0.25) ms | 11.70x | yes |
| 1024x1024 | 87.90 (2.78) ms | 8.67 (1.76) ms | 10.13x | yes |
| 1200x1799 | 183.69 (5.90) ms | 27.80 (9.63) ms | 6.61x | yes |
| 2048x2048 | 388.69 (81.29) ms | 60.82 (16.42) ms | 6.39x | yes |

### Benchmark-mask characteristics

The benchmark creates a contiguous boolean mask before timing
(`test/benchmark_resolutions.py:319-322`), so it exercises the binding's
zero-copy boolean fast path (`src/turbo_hce/_turbo_hce.cpp:1340-1350`).

**Measured:** across the five resized masks, foreground density was
32.78%-32.85%, while the foreground bounding rectangle covered
77.59%-77.98% of the full canvas. Thus ROI cropping helps this workload but
still leaves about 78% of the image inside the repeatedly scanned rectangle.

### Focused timing observations

These exploratory figures used medians from repeated calls and are not a
replacement for the required benchmark:

| 1024x1024 input | Turbo median |
|---|---:|
| blank boolean | 0.072 ms |
| one foreground pixel | 0.073 ms |
| centered 64x64 filled square | 0.261 ms |
| full filled square | 57.547 ms |
| 50%-density random boolean | 4.013 ms |

**Measured:** filled-square medians at sizes 128, 256, 512, 768, 1024, and
1536 were respectively 0.478, 2.134, 10.193, 27.153, 57.297, and 163.332 ms.
Time per source pixel rose from 29.2 ns to 69.2 ns over that range. In contrast,
50%-random-mask time per source pixel only rose from 2.83 ns to 5.21 ns.

**Inference:** masks that need many thinning rounds expose repeated active-area
scans; masks that converge in few rounds are much closer to linear in image
size. Instrumenting iteration and chunk counts is needed to quantify this.

For one fixed 1024x1024 50%-random logical mask, measured medians were 4.305 ms
for contiguous boolean, 4.278 ms for contiguous uint8 containing 0/1, 4.360 ms
for uint8 containing 0/255, 4.801 ms for contiguous float32, 5.180 ms for
contiguous float64, 4.847 ms for a strided uint8 view, and 6.358 ms for a
Fortran-order boolean array. These figures make conversion/layout worth fixing,
but secondary to convergence scheduling for the benchmark's boolean path.

Eight independent 1024x1024 random boolean calls took 34.54 ms sequentially,
24.63 ms with two Python worker threads, 15.16 ms with four, and 12.57 ms with
eight. This confirms useful call-level concurrency from the existing GIL
release, although it is not a controlled scalability study.

Hardware-counter profiling was attempted, but this host has
`perf_event_paranoid=4` and exposed no supported events. Cache/branch claims in
this report are therefore hypotheses, not measured counter results.

## Current work and data flow

The native path performs these phases:

1. Validate dimensions and pointers (`skeletonize.cpp:90-108`).
2. Scan the entire input to find the foreground bounding box
   (`skeletonize.cpp:110-158`).
3. Clear the entire output (`skeletonize.cpp:160-167`).
4. Allocate and zero a padded ROI plus 32-bit row and column count arrays
   (`skeletonize.cpp:169-182`).
5. Scan the ROI again, normalize nonzero bytes to 1, and build counts
   (`skeletonize.cpp:184-230`).
6. Repeatedly run two synchronous Zhang-Suen subpasses over the current global
   active rectangle (`skeletonize.cpp:264-438`).
7. Copy the final ROI into the already-cleared output
   (`skeletonize.cpp:440-445`).

For each nonempty AVX2 center chunk, the hot scan issues a center load followed
by eight overlapping unaligned neighbor loads (`skeletonize.cpp:297-321`). It
then recognizes all-interior chunks (`skeletonize.cpp:323-340`), constructs
eight-bit neighborhood codes (`skeletonize.cpp:342-350`), evaluates the
pass-specific bitset in registers (`skeletonize.cpp:352-368`), and serializes
set bits into a deletion vector (`skeletonize.cpp:370-374`). The current binary's
disassembly confirms AVX2 loads, `vptest`, `vpshufb`, `vpmovmskb`, `popcnt`, and
`tzcnt` are present in the function. There is no scalar-LUT spill in the 32-byte
hot loop.

The binding directly passes aligned, C-contiguous bool/uint8 arrays
(`_turbo_hce.cpp:1342-1345`). Other inputs are copied if needed, converted into
an OpenCV-owned uint8 buffer, and binarized (`_turbo_hce.cpp:1346-1350` and
`517-677`). Output is allocated directly as a NumPy boolean array
(`_turbo_hce.cpp:1352-1359`). The GIL is released around the core call
(`_turbo_hce.cpp:1364-1384`), but not around fallback input preparation.

## Ranked recommendations

### 1. Exact frontier or dirty-tile scheduling

- Likely impact: **very high** on thick, large, or spatially sparse shapes
- Effort: **medium to high**
- Correctness risk: **medium to high**

The active bounds only contract when complete outer rows or columns empty
(`skeletonize.cpp:268-277`). Every remaining row is visited and every chunk
between the same global `c_start` and `c_end` is tested on both subpasses
(`skeletonize.cpp:279-397`). A zero chunk is cheap, but it is still loaded and
looped over; an all-interior foreground chunk incurs all neighbor loads before
being rejected.

After a synchronous subpass, a foreground pixel whose 3x3 neighborhood did not
change has the same deletion decision the next time that pass is evaluated.
Only neighbors of deleted pixels need reconsideration. Preserve exact behavior
with two pass-specific frontiers: read an unchanged image while collecting the
entire deletion set, batch-apply it, then enqueue affected neighbors for the
opposite subpass. Deduplicate with an epoch/stamp array or per-tile bitset.

A lower-risk intermediate design is dirty 32xN or 32x32 tiles. Initially all
tiles are dirty; after a batch deletion, mark its tile and all tiles touched by
the affected 3x3 neighborhoods. This retains the current vector kernel while
eliminating stable tiles. A pixel frontier should remove more work but has more
queueing and deduplication overhead.

Validation experiment:

- Add temporary counters for outer iterations, chunks tested, zero chunks,
  interior chunks, LUT-tested chunks, deletions, and peak frontier size.
- Benchmark filled rectangles/disks over thickness and resolution, the supplied
  real mask, sparse long components, rings, random densities, and already-thin
  lines.
- Differential-test every intermediate prototype against scikit-image,
  especially patterns spanning tile/SIMD boundaries.
- Compare dense scan, dirty tile, and pixel frontier with an adaptive crossover;
  frontier bookkeeping may lose on small or highly random masks.

### 2. Compact the deletion batch

- Likely impact: **medium**, especially on deletion-heavy early passes
- Effort: **low to medium**
- Correctness risk: **low** if metadata timing is kept separate from image writes

Each queued deletion stores three `size_t` fields (`skeletonize.cpp:256-260`),
which is 24 bytes on this x86-64 build. The initial reserve can therefore claim
3 MiB for 131,072 entries (`skeletonize.cpp:261-262`), and a larger deletion
wave grows beyond it. Each entry is written during discovery and read again
during application (`skeletonize.cpp:370-374` and `428-435`).

Store only the linear padded index. Row and column counts can be decremented at
queue time because they are metadata: the image must remain unchanged until the
subpass finishes, but current code does not consult the column counts within a
subpass and only consults a row count at that row's entry. Updating after that
entry cannot change which pixels in the row are examined. Alternatively, keep
an 8- or 16-byte `(row, column)` representation where supported and derive the
linear index.

Do not clear `pad` while discovering deletions. That would change synchronous
Zhang-Suen semantics.

Validation experiment: record peak deletion count and allocation bytes per
subpass; compare 24-byte, 8-byte, and bitmap batches. Run ASan/UBSan and the full
differential suite after implementation.

### 3. Avoid redundant output stores and contract per subpass

- Likely impact: **small to medium**, larger when setup/memory traffic dominates
- Effort: **low**
- Correctness risk: **low**

For nonblank input, the entire output is zeroed at `skeletonize.cpp:166-167`,
then every ROI byte is overwritten at `skeletonize.cpp:440-445`. Zero only the
top and bottom regions and the left/right strips outside the ROI, then copy the
ROI. This always removes exactly one ROI-sized redundant output write; if the
ROI is the full image, the initial clear disappears entirely. Keep the existing
full clear for blank input.

Bounds are contracted only once before both subpasses (`skeletonize.cpp:264-279`).
Contract again after applying pass 0 before scanning pass 1, and after pass 1
before the next pass 0. Empty rows/columns cannot contain a valid deletion, so
this should be behavior-preserving and can save recently emptied edges.

Validation experiment: separate blank, tiny translated ROI, full-image ROI, and
border-touching timings. Use guard-filled output buffers in a direct C++ test to
prove all output bytes are assigned.

### 4. Rework normalization and foreground-count construction

- Likely impact: **medium** on few-iteration and dense masks
- Effort: **medium**
- Correctness risk: **low to medium**

During ROI population, every nonzero bit runs a scalar `tzcnt` loop and performs
a 32-bit column increment (`skeletonize.cpp:191-205`). That is O(foreground
pixels), adds a data-dependent loop, and mixes normalization stores with count
traffic. The padded byte vector is also zero-initialized before nonzero bytes
are stored, so dense ROI bytes are written twice (`skeletonize.cpp:179-200`).

Evaluate a two-stage construction:

- write normalized 0/1 bytes for every ROI byte with straight vector loops,
  while obtaining row counts from vector masks/popcounts;
- build column counts in a separate contiguous vertical accumulation pass that
  the compiler can vectorize, or accumulate wider lanes and flush in blocks;
- allocate pad storage without value-initializing its interior, explicitly
  zeroing only the halo, only if this can be done with simple exception-safe
  ownership.

Although this adds an apparent pass, it replaces per-set-bit scalar work and
data-dependent branches with regular streams. It must be density-benchmarked;
the current skip-store behavior may remain best for nearly blank ROIs.

Also consider whether full row/column counts are still needed after adopting a
frontier. Removing metadata is better than accelerating its construction.

### 5. Reduce neighbor-load and alignment cost after scheduling is fixed

- Likely impact: **medium** in the remaining dense hot path
- Effort: **medium to high**
- Correctness risk: **medium**

The hot block currently uses nine unaligned 32-byte loads, with left/center/right
windows heavily overlapping (`skeletonize.cpp:297-321`). Explore a sliding
three-row window: load aligned/current chunks and synthesize shifted neighbors,
or reuse previous/current/next vectors between adjacent chunks. AVX2's separate
128-bit shuffle lanes make cross-lane bytes easy to get wrong, so the instruction
count and generated assembly must be checked rather than assumed better.

Round `padded_cols` up to a 32- or 64-byte stride and align the allocation. The
current `roi_w + 2` stride makes row alignment drift and can create cache-line
split loads (`skeletonize.cpp:173-180`). Alignment increases padding slightly
but may make center/vertical streams cheaper. Test troublesome strides around
cache-line and page multiples, not just square powers of two.

The current in-register bitset predicate is already a sound AVX2 design. Direct
boolean-condition evaluation or a scalar LUT should only replace it if assembly
and end-to-end measurements win.

### 6. Add optional intra-image parallelism only for large active work

- Likely impact: **medium to high** for large masks with substantial per-pass work
- Effort: **high**
- Correctness risk: **medium**

Rows within a subpass can be scanned in parallel because `pad` is read-only until
the batch deletion barrier. Use per-thread deletion buffers, concatenate or
apply them after the barrier, and reduce row/column count updates without races.
There are two synchronization points per outer iteration, so spawning threads
per pass is not viable; use a persistent pool/OpenMP runtime or parallelize only
above a high work threshold.

The measured 8-call thread experiment shows that the existing call-level GIL
release already supplies useful throughput. Intracall threads can therefore
oversubscribe applications that already parallelize over images. Make the policy
controllable or coordinate through one library-level pool. On this Ryzen, two
4 MiB L3 slices also make cross-core shared-buffer behavior worth measuring.

Frontier/dirty-tile scheduling should precede this work. Parallelizing scans
that should not happen is a weaker design.

### 7. Optimize non-bool input preparation selectively

- Likely impact: **small for the supplied benchmark; medium for dtype/stride-heavy users**
- Effort: **medium**
- Correctness risk: **medium**

Fallback inputs may first become contiguous through NumPy
(`_turbo_hce.cpp:562-569`), then are converted into a second OpenCV uint8 buffer
(`_turbo_hce.cpp:576-677`), all before the GIL is released. The focused timings
showed about 0.5-2.1 ms overhead relative to contiguous boolean on a random
1024x1024 mask, depending on dtype/layout.

Potential improvements are to release the GIL around raw-data binarization after
all Python objects and pointers are secured, to template the initial scan/ROI
normalization over contiguous numeric input and avoid `cv::Mat`, or to support
simple positive row strides directly. General arbitrary-stride support includes
negative strides, byte order, alignment, and overflow cases and should only be
added if real workloads justify it.

The benchmark itself uses contiguous boolean input, so none of these changes
explains or fixes its 2048x2048 behavior.

### 8. Scratch reuse and compiler tuning

- Likely impact: **small to medium**
- Effort: **low to medium**
- Correctness risk: **low to medium**

Repeated calls allocate pad, two count vectors, and the deletion vector
(`skeletonize.cpp:179-182` and `256-262`). A thread-local scratch object or an
internal reusable workspace can amortize allocator costs while retaining
thread safety. Cap retained capacity so one unusually large image does not
permanently pin tens of megabytes per worker thread.

The x86 build already uses `-O3`, AVX2, SSE4.2, FMA, and LTO
(`setup.py:94-105`). The binary confirms that the intended AVX2 path was emitted.
Profile-guided optimization or a suitable `-mtune` may improve scheduling and
branch layout, but is unlikely to rival work reduction. `-march=native` is not
appropriate for distributable wheels. The current build also has no runtime
dispatch and requires AVX2 on x86; adding dispatch would improve portability,
not this host's hot-path speed.

## Correctness and API hazards

Any optimization must retain the following:

- **Synchronous subpasses:** collect the complete deletion set from one unchanged
  image, then apply it. In-place discovery changes results.
- **Pass order and LUT encoding:** the pass-specific bit masks and clockwise
  neighbor-bit mapping at `skeletonize.cpp:382-393` define pixel-exact output.
  A topologically equivalent skeleton is not sufficient.
- **Zero halo and borders:** ROI cropping must preserve the one-pixel zero halo
  (`skeletonize.cpp:169-180`), including objects touching source borders.
- **Nonzero truth semantics:** all supported numeric nonzero values, including
  negative values and floating NaNs, are foreground; signed zero is background
  (`_turbo_hce.cpp:582-672`).
- **Shape/dtype/errors:** Python accepts only two-dimensional numeric arrays,
  rejects object arrays, returns `bool` with the same shape, and supports empty
  dimensions (`_turbo_hce.cpp:1309-1338`). Preserve exception translation and
  allocation-failure handling (`_turbo_hce.cpp:1361-1399`).
- **Concurrency:** global mutable scratch/LUT state would break the existing
  thread-safe, GIL-released API.
- **Counter width:** row and column counts are `uint32_t`
  (`skeletonize.cpp:181-182`). A row or column containing more than 2^32-1
  foreground pixels can wrap and be mistaken for empty even though dimensions
  are `size_t`. This is impractical for normal images but is a real semantic
  edge. A redesign should use `size_t`, reject impossible dimensions explicitly,
  or prove a narrower bound rather than silently preserving the overflow.
- **Direct C++ aliasing:** the pointer overload clears output before finishing
  its reads (`skeletonize.cpp:166-187`), so input/output aliasing is not currently
  supported even though the header does not state this. Do not accidentally
  treat aliasing as supported without defining and testing it.

The existing ten `unittest` cases passed after the benchmark. They cover empty
and degenerate shapes, representative topology and borders, randomized densities,
noncontiguous/numeric inputs, SIMD boundary widths, translated ROIs, concurrency,
and invalid dimensions/object dtype (`test/test_skeletonize.py:34-144`). Expand
tests before a scheduler rewrite with exhaustive small masks where feasible,
tile boundaries, very thick solids, long one-pixel lines, disjoint components,
negative strides, NaNs/infinities/signed zero, and allocation/counter limits.

## Areas unlikely to pay off now

- Replacing the core with an OpenCV thinning call: no OpenCV routine appears in
  the core hot loop, the minimal build does not include `ximgproc`, and a
  different implementation may not be pixel-exact with scikit-image.
- GPU offload: iterative global synchronization and host/device transfer are a
  poor fit at these sizes unless masks are already resident and processed in
  batches. It also greatly expands deployment complexity.
- AVX-512: the measured CPU does not support it, and it does not remove repeated
  convergence scans.
- Bit-packing the entire image as a first step: it could reduce cache/TLB traffic,
  but neighborhood shifts, cross-word boundaries, deletion queues, and tail
  handling make it a high-risk redesign. Dirty scheduling should be evaluated
  first.
- Branch hints, validation micro-optimizations, exception-path tuning, or Python
  wrapper call reduction: measured calls are millisecond-scale on target masks,
  while wrapper overhead is tiny.
- Huge pages, prefetching, or non-temporal stores before hardware-counter data:
  the algorithm revisits `pad`, and unproven cache-bypass changes can regress it.
- Parallelizing blank/small masks: the 1024x1024 blank and single-pixel cases were
  about 0.07 ms; thread scheduling would dominate.

## Suggested implementation sequence

1. Add temporary phase/chunk/iteration/deletion instrumentation in a development
   branch and rerun the required benchmark plus the shape/density matrix.
2. Land the outside-ROI output clearing and per-subpass contraction with
   differential tests.
3. Compact the deletion batch and measure queue peaks/allocation traffic.
4. Prototype dirty tiles, then a pixel frontier if tile granularity leaves too
   much work. Keep an adaptive dense path.
5. Reassess profiles. Optimize normalization/count setup for few-iteration masks
   and neighbor loads/alignment for the remaining scan-heavy cases.
6. Add thresholded intra-image parallelism only if single-image latency still
   warrants its synchronization and oversubscription costs.

For benchmark decisions, pin the process to a suitable core set, control CPU
frequency/thermal state where possible, run more than ten samples at 2048x2048,
and report full distributions. The current 2048x2048 IQR is too wide for small
percentage claims.
