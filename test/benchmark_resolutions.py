"""Benchmark comparing Python reference vs C++ turbo_hce at various resolutions."""

import os
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple
import cv2 as cv
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import turbo_hce
from utils import hce_metric_main as py_hce

MODEL_PATH = "/home/samuel/ai_data/skin_models/sunet/sunet.onnx"
IMAGE_PATH = "/home/samuel/da/skindataset/images/00001.png"
MASK_PATH = "/home/samuel/da/skindataset/masks/00001.png"

RESOLUTIONS: List[Tuple[int, int]] = [
    (256, 256),
    (512, 512),
    (1024, 1024),
    (1200, 1799),
    (2048, 2048),
]


def get_cpu_model() -> str:
    """Retrieve CPU model name from /proc/cpuinfo or platform."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def get_git_commit() -> str:
    """Retrieve short git commit identifier if available."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def generate_synthetic_base(size: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic ground truth and prediction masks if dataset files are absent.

    :param size: Dimension of the square base mask.
    :return: Tuple of (ground truth, prediction).
    """
    gt = np.zeros((size, size), dtype=np.uint8)
    pred = np.zeros((size, size), dtype=np.uint8)

    cv.circle(gt, (size // 2, size // 2), size // 3, 255, -1)
    cv.rectangle(gt, (size // 4, size // 4), (size * 3 // 4, size // 2), 255, -1)
    cv.circle(gt, (size // 2, size // 2), size // 6, 0, -1)

    cv.circle(pred, (size // 2 + 5, size // 2 - 3), size // 3, 255, -1)
    cv.rectangle(pred, (size // 4 + 8, size // 4 - 4), (size * 3 // 4 - 5, size // 2 + 4), 255, -1)
    cv.circle(pred, (size // 2, size // 2), size // 5, 0, -1)
    cv.circle(pred, (size // 8, size * 7 // 8), size // 10, 255, -1)

    return gt, pred


def generate_test_pair_at_resolution(
    height: int,
    width: int,
    base_gt: np.ndarray,
    base_pred: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resize base masks to the target resolution and generate the corresponding skeleton.

    :param height: Target mask height.
    :param width: Target mask width.
    :param base_gt: Ground truth mask array.
    :param base_pred: Prediction mask array.
    :return: Tuple of resized (gt, pred, skeleton).
    """
    gt_resized = cv.resize(base_gt, (width, height), interpolation=cv.INTER_NEAREST)
    pred_resized = cv.resize(base_pred, (width, height), interpolation=cv.INTER_NEAREST)
    ske = skeletonize(gt_resized > 128)
    return gt_resized, pred_resized, ske


def benchmark_resolution(
    gt: np.ndarray,
    pred: np.ndarray,
    ske: np.ndarray,
    iterations: int = 10,
) -> Dict[str, Any]:
    """
    Benchmark Python and C++ relax_HCE using interleaved execution and robust dispersion metrics.

    :param gt: Ground truth mask.
    :param pred: Prediction mask.
    :param ske: Skeleton mask.
    :param iterations: Number of timed repetitions per implementation.
    :return: Dictionary containing timing distributions, speedup, and correctness.
    """
    # Warmup
    py_warmup = py_hce.relax_HCE(gt, pred, ske)
    c_warmup = turbo_hce.relax_HCE(gt, pred, ske)
    assert py_warmup == c_warmup, f"Warmup mismatch: {py_warmup} vs {c_warmup}"

    py_times: List[float] = []
    c_times: List[float] = []

    for i in range(iterations):
        if i % 2 == 0:
            t0 = time.perf_counter()
            py_hce.relax_HCE(gt, pred, ske)
            t1 = time.perf_counter()
            py_times.append((t1 - t0) * 1000.0)

            t2 = time.perf_counter()
            turbo_hce.relax_HCE(gt, pred, ske)
            t3 = time.perf_counter()
            c_times.append((t3 - t2) * 1000.0)
        else:
            t0 = time.perf_counter()
            turbo_hce.relax_HCE(gt, pred, ske)
            t1 = time.perf_counter()
            c_times.append((t1 - t0) * 1000.0)

            t2 = time.perf_counter()
            py_hce.relax_HCE(gt, pred, ske)
            t3 = time.perf_counter()
            py_times.append((t3 - t2) * 1000.0)

    py_median = float(np.median(py_times))
    c_median = float(np.median(c_times))
    py_q25, py_q75 = float(np.percentile(py_times, 25)), float(np.percentile(py_times, 75))
    c_q25, c_q75 = float(np.percentile(c_times, 25)), float(np.percentile(c_times, 75))
    py_iqr = py_q75 - py_q25
    c_iqr = c_q75 - c_q25

    speedup = py_median / max(c_median, 1e-6)

    return {
        "py_median_ms": py_median,
        "py_iqr_ms": py_iqr,
        "c_median_ms": c_median,
        "c_iqr_ms": c_iqr,
        "speedup": speedup,
        "matches": py_warmup == c_warmup,
    }


def benchmark_skeletonize_resolution(
    mask: np.ndarray,
    iterations: int = 10,
) -> Dict[str, Any]:
    """
    Benchmark skimage vs turbo_hce skeletonize using interleaved execution.

    :param mask: Binary mask array to skeletonize.
    :param iterations: Number of timed repetitions per implementation.
    :return: Dictionary containing timing distributions, speedup, and correctness.
    """
    sk_warmup = skeletonize(mask)
    tb_warmup = turbo_hce.skeletonize(mask)
    matches = bool(np.array_equal(sk_warmup, tb_warmup))

    sk_times: List[float] = []
    tb_times: List[float] = []

    for i in range(iterations):
        if i % 2 == 0:
            t0 = time.perf_counter()
            skeletonize(mask)
            t1 = time.perf_counter()
            sk_times.append((t1 - t0) * 1000.0)

            t2 = time.perf_counter()
            turbo_hce.skeletonize(mask)
            t3 = time.perf_counter()
            tb_times.append((t3 - t2) * 1000.0)
        else:
            t0 = time.perf_counter()
            turbo_hce.skeletonize(mask)
            t1 = time.perf_counter()
            tb_times.append((t1 - t0) * 1000.0)

            t2 = time.perf_counter()
            skeletonize(mask)
            t3 = time.perf_counter()
            sk_times.append((t3 - t2) * 1000.0)

    sk_median = float(np.median(sk_times))
    tb_median = float(np.median(tb_times))
    sk_q25, sk_q75 = (
        float(np.percentile(sk_times, 25)),
        float(np.percentile(sk_times, 75)),
    )
    tb_q25, tb_q75 = (
        float(np.percentile(tb_times, 25)),
        float(np.percentile(tb_times, 75)),
    )
    sk_iqr = sk_q75 - sk_q25
    tb_iqr = tb_q75 - tb_q25

    speedup = sk_median / max(tb_median, 1e-6)

    return {
        "skimage_median_ms": sk_median,
        "skimage_iqr_ms": sk_iqr,
        "turbo_median_ms": tb_median,
        "turbo_iqr_ms": tb_iqr,
        "speedup": speedup,
        "matches": matches,
    }


def run_all_benchmarks(iterations: int = 10) -> Dict[str, Any]:
    """
    Execute resolution benchmark suite and print markdown summary tables with metadata.

    :param iterations: Number of iterations per resolution.
    :return: Benchmark results mapped by category and resolution tuple.
    """
    has_resources = (
        os.path.exists(MODEL_PATH)
        and os.path.exists(IMAGE_PATH)
        and os.path.exists(MASK_PATH)
    )

    if has_resources:
        from utils.session import Session

        session = Session(model_path=MODEL_PATH)
        pil_img = Image.open(IMAGE_PATH)
        base_gt = cv.imread(MASK_PATH, cv.IMREAD_GRAYSCALE)
        pred_full = np.array(session.remove(pil_img, do_sigmoid=False, simple_norm=True))
        base_pred = pred_full[:, :, 3]
        source_desc = "real skin dataset sample 00001.png"
    else:
        base_gt, base_pred = generate_synthetic_base(size=512)
        source_desc = "synthetic complex mask fixture"

    cpu_model = get_cpu_model()
    git_rev = get_git_commit()
    date_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    print("\n" + "=" * 80)
    print("Turbo HCE Performance Benchmark (relax_HCE)")
    print("=" * 80)
    print(f"Date:         {date_str}")
    print(f"CPU:          {cpu_model}")
    print(f"Platform:     {platform.platform()}")
    print(f"Python:       {sys.version.split()[0]}")
    print(f"NumPy:        {np.__version__}")
    print(f"OpenCV:       {cv.__version__}")
    print(f"Git commit:   {git_rev}")
    print(f"Data source:  {source_desc}")
    print(f"Repetitions:  {iterations} (interleaved order)")
    print("=" * 80)

    header = (
        f"| {'Resolution (WxH)':<18} | {'Python Median (IQR)':<22} | "
        f"{'C++ Median (IQR)':<20} | {'Speedup':<10} | {'Exact Match':<11} |"
    )
    sep = (
        f"|{'-' * 20}|{'-' * 24}|{'-' * 22}"
        f"|{'-' * 12}|{'-' * 13}|"
    )
    print(header)
    print(sep)

    all_results: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for width, height in RESOLUTIONS:
        gt_r, pred_r, ske_r = generate_test_pair_at_resolution(
            height, width, base_gt, base_pred
        )
        stats = benchmark_resolution(gt_r, pred_r, ske_r, iterations=iterations)
        all_results[(width, height)] = stats

        res_str = f"{width}x{height}"
        match_str = "YES" if stats["matches"] else "NO"
        py_str = f"{stats['py_median_ms']:.2f} (±{stats['py_iqr_ms']:.2f}) ms"
        c_str = f"{stats['c_median_ms']:.2f} (±{stats['c_iqr_ms']:.2f}) ms"
        row = (
            f"| {res_str:<18} | {py_str:>20} | {c_str:>18} "
            f"| {stats['speedup']:>8.2f}x | {match_str:^11} |"
        )
        print(row)

    print("=" * 80 + "\n")

    print("=" * 80)
    print("Skeletonize Performance Benchmark (scikit-image vs Turbo HCE)")
    print("=" * 80)
    sk_header = (
        f"| {'Resolution (WxH)':<18} | {'skimage Median (IQR)':<22} | "
        f"{'Turbo Median (IQR)':<20} | {'Speedup':<10} | {'Exact Match':<11} |"
    )
    print(sk_header)
    print(sep)

    skel_results: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for width, height in RESOLUTIONS:
        gt_r = cv.resize(base_gt, (width, height), interpolation=cv.INTER_NEAREST)
        bin_mask = gt_r > 128
        sk_stats = benchmark_skeletonize_resolution(bin_mask, iterations=iterations)
        skel_results[(width, height)] = sk_stats

        res_str = f"{width}x{height}"
        match_str = "YES" if sk_stats["matches"] else "NO"
        sk_str = f"{sk_stats['skimage_median_ms']:.2f} (±{sk_stats['skimage_iqr_ms']:.2f}) ms"
        tb_str = f"{sk_stats['turbo_median_ms']:.2f} (±{sk_stats['turbo_iqr_ms']:.2f}) ms"
        row = (
            f"| {res_str:<18} | {sk_str:>20} | {tb_str:>18} "
            f"| {sk_stats['speedup']:>8.2f}x | {match_str:^11} |"
        )
        print(row)

    print("=" * 80 + "\n")

    return {"relax_hce": all_results, "skeletonize": skel_results}


if __name__ == "__main__":
    run_all_benchmarks(iterations=10)

