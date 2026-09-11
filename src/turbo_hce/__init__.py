"""Fast C extension for Human Correction Effort (HCE) metric."""

from typing import List, Sequence, Tuple
import numpy as np

from ._turbo_hce import (
    relax_HCE as _relax_HCE,
    relax_hce as _relax_hce,
    filter_bdy_cond as _filter_bdy_cond,
    approximate_RDP as _approximate_RDP,
    approximate_rdp as _approximate_rdp,
    skeletonize as _skeletonize,
)

__version__ = "0.1.0"


def relax_HCE(
    gt: np.ndarray,
    rs: np.ndarray,
    gt_ske: np.ndarray,
    relax: int = 5,
    epsilon: float = 2.0,
) -> Tuple[int, float, int, float]:
    """
    Compute relaxed Human Correction Effort (HCE) components.

    :param gt: Ground truth mask array (2D or 3D; channel 0 used if 3D).
    :param rs: Prediction mask array (2D or 3D; channel 0 used if 3D).
    :param gt_ske: Ground truth skeleton array (strictly 2D).
    :param relax: Number of relaxation iterations for morphological operations (>= 0).
    :param epsilon: Epsilon parameter for polygon approximation (finite, >= 0.0).
    :return: Tuple of (poly_fp_point_cnt, indep_cnt_fp, poly_fn_point_cnt, indep_cnt_fn).
    """
    return _relax_HCE(gt, rs, gt_ske, relax=relax, epsilon=epsilon)


def relax_hce(
    gt: np.ndarray,
    rs: np.ndarray,
    gt_ske: np.ndarray,
    relax: int = 5,
    epsilon: float = 2.0,
) -> Tuple[int, float, int, float]:
    """
    PEP8 alias for relax_HCE.

    :param gt: Ground truth mask array (2D or 3D; channel 0 used if 3D).
    :param rs: Prediction mask array (2D or 3D; channel 0 used if 3D).
    :param gt_ske: Ground truth skeleton array (strictly 2D).
    :param relax: Number of relaxation iterations for morphological operations (>= 0).
    :param epsilon: Epsilon parameter for polygon approximation (finite, >= 0.0).
    :return: Tuple of (poly_fp_point_cnt, indep_cnt_fp, poly_fn_point_cnt, indep_cnt_fn).
    """
    return _relax_hce(gt, rs, gt_ske, relax=relax, epsilon=epsilon)


def filter_bdy_cond(
    bdy_: Sequence[np.ndarray],
    mask: np.ndarray,
    cond: np.ndarray,
) -> Tuple[List[np.ndarray], float]:
    """
    Filter boundary conditions and count independent regions requiring correction.

    :param bdy_: Sequence of boundary contour arrays with shape (N, 1, 2) or (N, 2).
    :param mask: 2D binary mask array identifying connected regions.
    :param cond: 2D condition mask array for boundary filtering.
    :return: Tuple of (boundaries, indep_cnt).
    """
    return _filter_bdy_cond(bdy_, mask, cond)


def approximate_RDP(
    boundaries: Sequence[np.ndarray],
    epsilon: float = 1.0,
) -> Tuple[List[np.ndarray], List[int], int]:
    """
    Approximate boundary polygons using the Ramer-Douglas-Peucker algorithm.

    :param boundaries: Sequence of boundary contour arrays with shape (N, 1, 2) or (N, 2).
    :param epsilon: Approximation accuracy epsilon (finite, >= 0.0).
    :return: Tuple of (approximated_boundaries, boundary_lengths, total_pixel_cnt).
    """
    return _approximate_RDP(boundaries, epsilon=epsilon)


def approximate_rdp(
    boundaries: Sequence[np.ndarray],
    epsilon: float = 1.0,
) -> Tuple[List[np.ndarray], List[int], int]:
    """
    PEP8 alias for approximate_RDP.

    :param boundaries: Sequence of boundary contour arrays with shape (N, 1, 2) or (N, 2).
    :param epsilon: Approximation accuracy epsilon (finite, >= 0.0).
    :return: Tuple of (approximated_boundaries, boundary_lengths, total_pixel_cnt).
    """
    return _approximate_rdp(boundaries, epsilon=epsilon)


def skeletonize(image: np.ndarray) -> np.ndarray:
    """
    Compute the 2D Zhang-Suen morphological skeleton of a binary image.

    Matches the default 2D behavior of skimage.morphology.skeletonize,
    treating every nonzero element as foreground and returning a boolean array.

    :param image: 2D array of boolean or numeric dtype.
    :return: 2D boolean array of the skeleton.
    """
    return _skeletonize(image)


__all__ = [
    "relax_HCE",
    "relax_hce",
    "filter_bdy_cond",
    "approximate_RDP",
    "approximate_rdp",
    "skeletonize",
    "__version__",
]

