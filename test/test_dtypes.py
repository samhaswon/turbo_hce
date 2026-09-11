"""Table-driven differential tests comparing turbo_hce against reference across all dtypes."""

import os
import sys
import unittest

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import turbo_hce
from utils import hce_metric_main as py_hce

DTYPES = [
    np.uint8,
    np.int8,
    np.uint16,
    np.int16,
    np.uint32,
    np.int32,
    np.uint64,
    np.int64,
    np.float16,
    np.float32,
    np.float64,
    bool,
]


class TestDtypesDifferential(unittest.TestCase):
    """Differential tests comparing C++ extension against Python reference across dtypes."""

    def test_all_dtypes_equivalence(self):
        """Verify identical relax_HCE output across all standard NumPy numeric dtypes."""
        shape = (40, 40)
        ske = np.zeros(shape, dtype=bool)
        ske[20, 10:30] = True

        for dtype in DTYPES:
            with self.subTest(dtype=dtype.__name__):
                gt = np.zeros(shape, dtype=dtype)
                rs = np.zeros(shape, dtype=dtype)
                if dtype == np.int8:
                    gt[10:25, 10:25] = 100
                    rs[15:30, 15:30] = 100
                elif dtype == bool:
                    gt[10:25, 10:25] = True
                    rs[15:30, 15:30] = True
                else:
                    gt[10:25, 10:25] = 200
                    rs[15:30, 15:30] = 200

                py_res = py_hce.relax_HCE(gt, rs, ske)
                c_res = turbo_hce.relax_HCE(gt, rs, ske)
                self.assertEqual(py_res, c_res)

    def test_signed_negative_values(self):
        """Verify signed negative values (such as -1) match reference behavior."""
        shape = (30, 30)
        ske = np.zeros(shape, dtype=bool)
        ske[15, 5:25] = True

        for signed_dtype in [np.int8, np.int16, np.int32, np.int64, np.float32, np.float64]:
            with self.subTest(dtype=signed_dtype.__name__):
                gt = np.full(shape, -1, dtype=signed_dtype)
                rs = np.full(shape, -1, dtype=signed_dtype)
                gt[10:20, 10:20] = 200 if signed_dtype != np.int8 else 50

                py_res = py_hce.relax_HCE(gt, rs, ske)
                c_res = turbo_hce.relax_HCE(gt, rs, ske)
                self.assertEqual(py_res, c_res)

    def test_boundary_values_around_128(self):
        """Verify thresholding behavior at 127, 128, and 129 matches reference."""
        shape = (30, 30)
        ske = np.zeros(shape, dtype=bool)

        for val in [127, 128, 129]:
            with self.subTest(val=val):
                gt = np.full(shape, val, dtype=np.uint8)
                rs = np.full(shape, val, dtype=np.uint8)
                py_res = py_hce.relax_HCE(gt, rs, ske)
                c_res = turbo_hce.relax_HCE(gt, rs, ske)
                self.assertEqual(py_res, c_res)

    def test_non_contiguous_slices(self):
        """Verify non-contiguous sliced views match reference behavior."""
        big_gt = np.zeros((60, 60), dtype=np.uint8)
        big_gt[20:50, 20:50] = 255
        big_rs = np.zeros((60, 60), dtype=np.uint8)
        big_rs[25:55, 25:55] = 255

        gt_slice = big_gt[::2, ::2]
        rs_slice = big_rs[::2, ::2]
        ske = np.zeros(gt_slice.shape, dtype=bool)
        ske[15, :] = True

        self.assertFalse(gt_slice.flags["C_CONTIGUOUS"])
        py_res = py_hce.relax_HCE(gt_slice, rs_slice, ske)
        c_res = turbo_hce.relax_HCE(gt_slice, rs_slice, ske)
        self.assertEqual(py_res, c_res)

    def test_three_channel_inputs(self):
        """Verify 3D (H, W, C) inputs extract channel 0 correctly."""
        shape = (30, 30, 3)
        gt_3d = np.zeros(shape, dtype=np.uint8)
        rs_3d = np.zeros(shape, dtype=np.uint8)
        gt_3d[10:20, 10:20, 0] = 255
        gt_3d[10:20, 10:20, 1:] = 100
        rs_3d[12:22, 12:22, 0] = 255

        ske = np.zeros((30, 30), dtype=bool)
        py_res = py_hce.relax_HCE(gt_3d, rs_3d, ske)
        c_res = turbo_hce.relax_HCE(gt_3d, rs_3d, ske)
        self.assertEqual(py_res, c_res)

    def test_non_finite_float_values(self):
        """Verify NaN and infinity follow NumPy comparison and truth semantics."""
        shape = (30, 30)

        for dtype in [np.float16, np.float32, np.float64]:
            with self.subTest(dtype=dtype.__name__):
                gt = np.zeros(shape, dtype=dtype)
                rs = np.zeros(shape, dtype=dtype)
                ske = np.zeros(shape, dtype=dtype)

                gt[5:10, 5:10] = np.nan
                gt[12:18, 12:18] = np.inf
                gt[20:25, 20:25] = -np.inf
                rs[7:12, 7:12] = np.nan
                rs[14:20, 14:20] = np.inf
                rs[22:27, 22:27] = -np.inf
                ske[15, 5:25] = np.nan
                ske[20, 5:15] = np.inf
                ske[21, 15:25] = -np.inf

                py_res = py_hce.relax_HCE(gt, rs, ske)
                c_res = turbo_hce.relax_HCE(gt, rs, ske)
                self.assertEqual(py_res, c_res)

    def test_targeted_nan_skeleton_audit_case(self):
        """Verify targeted 20x20 NaN skeleton case from audit produces identical output."""
        shape = (20, 20)
        for dtype in [np.float16, np.float32, np.float64]:
            with self.subTest(dtype=dtype.__name__):
                gt = np.zeros(shape, dtype=dtype)
                rs = np.zeros(shape, dtype=dtype)
                ske = np.zeros(shape, dtype=dtype)
                ske[10, 5:7] = np.nan

                py_res = py_hce.relax_HCE(gt, rs, ske)
                c_res = turbo_hce.relax_HCE(gt, rs, ske)
                self.assertEqual(py_res, c_res)
                self.assertEqual(c_res[0], 0)
                self.assertEqual(c_res[2], 2)


    def test_unaligned_arrays(self):
        """Verify multi-byte dtypes are safe when the NumPy data pointer is unaligned."""
        shape = (30, 30)
        dtypes = [
            np.uint16,
            np.int16,
            np.uint32,
            np.int32,
            np.uint64,
            np.int64,
            np.float16,
            np.float32,
            np.float64,
        ]

        for dtype in dtypes:
            with self.subTest(dtype=dtype.__name__):
                item_count = int(np.prod(shape))
                item_size = np.dtype(dtype).itemsize
                gt_buffer = bytearray(item_count * item_size + 1)
                rs_buffer = bytearray(item_count * item_size + 1)
                ske_buffer = bytearray(item_count * item_size + 1)
                gt = np.ndarray(shape, dtype=dtype, buffer=gt_buffer, offset=1)
                rs = np.ndarray(shape, dtype=dtype, buffer=rs_buffer, offset=1)
                ske = np.ndarray(shape, dtype=dtype, buffer=ske_buffer, offset=1)

                self.assertTrue(gt.flags["C_CONTIGUOUS"])
                self.assertFalse(gt.flags["ALIGNED"])

                gt.fill(0)
                rs.fill(0)
                ske.fill(0)
                gt[5:20, 5:20] = 200
                rs[10:25, 10:25] = 200
                ske[15, 5:25] = 1

                py_res = py_hce.relax_HCE(gt, rs, ske)
                c_res = turbo_hce.relax_HCE(gt, rs, ske)
                self.assertEqual(py_res, c_res)

    def test_non_native_byte_order(self):
        """Verify byte-swapped integer and float masks are normalized correctly."""
        shape = (30, 30)
        native_order = "<" if sys.byteorder == "little" else ">"
        swapped_order = ">" if native_order == "<" else "<"

        for dtype in [np.uint16, np.int32, np.uint64, np.float32, np.float64]:
            with self.subTest(dtype=dtype.__name__):
                swapped_dtype = np.dtype(dtype).newbyteorder(swapped_order)
                gt = np.zeros(shape, dtype=swapped_dtype)
                rs = np.zeros(shape, dtype=swapped_dtype)
                ske = np.zeros(shape, dtype=swapped_dtype)
                gt[5:20, 5:20] = 200
                rs[10:25, 10:25] = 200
                ske[15, 5:25] = 1

                self.assertFalse(gt.dtype.isnative)
                py_res = py_hce.relax_HCE(gt, rs, ske)
                c_res = turbo_hce.relax_HCE(gt, rs, ske)
                self.assertEqual(py_res, c_res)


if __name__ == "__main__":
    unittest.main()
