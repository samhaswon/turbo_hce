"""Smoke tests for turbo_hce C extension module."""

import os
import subprocess
import sys
import unittest

import numpy as np
import cv2 as cv

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import turbo_hce
from utils import hce_metric_main as py_hce


class TestSmokeTurboHCE(unittest.TestCase):
    """Smoke tests comparing turbo_hce C extension with Python reference."""

    def test_module_attributes(self):
        """Verify module exports required functions and version."""
        self.assertTrue(hasattr(turbo_hce, "__version__"))
        self.assertTrue(hasattr(turbo_hce, "relax_HCE"))
        self.assertTrue(hasattr(turbo_hce, "relax_hce"))
        self.assertTrue(hasattr(turbo_hce, "filter_bdy_cond"))
        self.assertTrue(hasattr(turbo_hce, "approximate_RDP"))
        self.assertTrue(hasattr(turbo_hce, "approximate_rdp"))
        self.assertTrue(hasattr(turbo_hce, "skeletonize"))

    def test_empty_masks(self):
        """Verify relax_HCE returns zeroes for all-zero masks."""
        shape = (100, 100)
        gt = np.zeros(shape, dtype=np.uint8)
        rs = np.zeros(shape, dtype=np.uint8)
        ske = np.zeros(shape, dtype=bool)

        py_res = py_hce.relax_HCE(gt, rs, ske)
        c_res = turbo_hce.relax_HCE(gt, rs, ske)

        self.assertEqual(py_res, c_res)
        self.assertEqual(c_res, (0, 0.0, 0, 0.0))

    def test_identical_masks(self):
        """Verify relax_HCE when gt and rs are identical."""
        gt = np.zeros((100, 100), dtype=np.uint8)
        gt[30:70, 30:70] = 255
        rs = gt.copy()
        ske = np.zeros((100, 100), dtype=bool)
        ske[50, 30:70] = True

        py_res = py_hce.relax_HCE(gt, rs, ske)
        c_res = turbo_hce.relax_HCE(gt, rs, ske)

        self.assertEqual(py_res, c_res)

    def test_disjoint_rectangles(self):
        """Verify relax_HCE on non-overlapping rectangular regions."""
        gt = np.zeros((120, 120), dtype=np.uint8)
        rs = np.zeros((120, 120), dtype=np.uint8)
        gt[10:50, 10:50] = 255
        rs[70:110, 70:110] = 255
        ske = np.zeros((120, 120), dtype=bool)
        ske[30, 10:50] = True

        py_res = py_hce.relax_HCE(gt, rs, ske, relax=3, epsilon=1.5)
        c_res = turbo_hce.relax_HCE(gt, rs, ske, relax=3, epsilon=1.5)

        self.assertEqual(py_res, c_res)

    def test_geometric_circles(self):
        """Verify relax_HCE on overlapping circular regions."""
        gt = np.zeros((150, 150), dtype=np.uint8)
        rs = np.zeros((150, 150), dtype=np.uint8)
        cv.circle(gt, (75, 75), 40, 255, -1)
        cv.circle(rs, (85, 75), 40, 255, -1)
        ske = np.zeros((150, 150), dtype=bool)
        cv.circle(ske.view(np.uint8), (75, 75), 20, 1, 1)

        py_res = py_hce.relax_HCE(gt, rs, ske, relax=4, epsilon=2.0)
        c_res = turbo_hce.relax_HCE(gt, rs, ske, relax=4, epsilon=2.0)

        self.assertEqual(py_res, c_res)

    def test_filter_bdy_cond(self):
        """Verify filter_bdy_cond output matches Python reference."""
        rng = np.random.default_rng(42)
        mask = rng.integers(0, 2, (80, 80), dtype=np.uint8)
        cond = rng.integers(0, 2, (80, 80), dtype=np.uint8)
        ctrs, _ = cv.findContours(mask.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        py_bdies, py_indep = py_hce.filter_bdy_cond(ctrs, mask, cond)
        c_bdies, c_indep = turbo_hce.filter_bdy_cond(ctrs, mask, cond)

        self.assertEqual(py_indep, c_indep)
        self.assertEqual(len(py_bdies), len(c_bdies))
        for pb, cb in zip(py_bdies, c_bdies):
            np.testing.assert_array_equal(pb, cb)

    def test_approximate_rdp(self):
        """Verify approximate_RDP matches Python reference across epsilons."""
        rng = np.random.default_rng(99)
        test_curve = rng.integers(0, 200, (60, 1, 2), dtype=np.int32)
        boundaries = [test_curve, test_curve[:30]]

        for eps in (0.5, 1.0, 2.5, 5.0):
            py_bdies, py_lens, py_cnt = py_hce.approximate_RDP(boundaries, epsilon=eps)
            c_bdies, c_lens, c_cnt = turbo_hce.approximate_RDP(boundaries, epsilon=eps)

            self.assertEqual(py_cnt, c_cnt)
            self.assertEqual(py_lens, c_lens)
            self.assertEqual(len(py_bdies), len(c_bdies))
            for pb, cb in zip(py_bdies, c_bdies):
                np.testing.assert_array_equal(pb, cb)

    def test_pep8_aliases(self):
        """Verify PEP8 alias functions produce identical output."""
        gt = np.zeros((50, 50), dtype=np.uint8)
        gt[10:40, 10:40] = 255
        rs = np.zeros((50, 50), dtype=np.uint8)
        rs[20:45, 20:45] = 255
        ske = np.zeros((50, 50), dtype=bool)

        res_upper = turbo_hce.relax_HCE(gt, rs, ske)
        res_lower = turbo_hce.relax_hce(gt, rs, ske)
        self.assertEqual(res_upper, res_lower)

    def test_error_handling(self):
        """Verify appropriate errors are raised on invalid inputs and parameters."""
        gt = np.zeros((50, 50), dtype=np.uint8)
        rs_mismatched = np.zeros((60, 60), dtype=np.uint8)
        ske = np.zeros((50, 50), dtype=bool)

        with self.assertRaises(ValueError):
            turbo_hce.relax_HCE(gt, rs_mismatched, ske)

        with self.assertRaises(TypeError):
            turbo_hce.relax_HCE("not an array", gt, ske)

        with self.assertRaises(ValueError):
            turbo_hce.relax_HCE(gt, gt, ske, relax=-1)

        with self.assertRaises(ValueError):
            turbo_hce.relax_HCE(gt, gt, ske, epsilon=-1.0)

        with self.assertRaises(ValueError):
            turbo_hce.relax_HCE(gt, gt, ske, epsilon=float("nan"))

        with self.assertRaises(ValueError):
            turbo_hce.relax_HCE(gt, gt, ske, epsilon=float("inf"))

        with self.assertRaises(ValueError):
            turbo_hce.approximate_RDP([], epsilon=-0.5)

        with self.assertRaises(ValueError):
            turbo_hce.approximate_RDP([], epsilon=float("nan"))

        with self.assertRaises(ValueError):
            turbo_hce.relax_HCE(gt, gt, np.zeros((50, 50, 1), dtype=bool))

        with self.assertRaises(ValueError):
            turbo_hce.relax_HCE(np.zeros((0, 50), dtype=np.uint8), gt, ske)

        with self.assertRaises(ValueError):
            turbo_hce.relax_HCE(np.zeros((50, 50, 0), dtype=np.uint8), gt, ske)

        with self.assertRaises(ValueError):
            turbo_hce.relax_HCE(np.zeros((2, 2, 2, 2), dtype=np.uint8), gt, ske)

        with self.assertRaises(TypeError):
            turbo_hce.relax_HCE(gt.astype(np.complex64), gt, ske)

        with self.assertRaises(ValueError):
            bad_cnt = np.zeros((5, 2, 2), dtype=np.int32)
            turbo_hce.filter_bdy_cond([bad_cnt], gt, gt)

        with self.assertRaises(ValueError):
            bad_cnt = np.zeros((5, 3), dtype=np.int32)
            turbo_hce.filter_bdy_cond([bad_cnt], gt, gt)

    def test_subprocess_exception_safety(self):
        """Verify invalid inputs raise exceptions in Python without crashing the process."""
        code = (
            "import sys\n"
            "import turbo_hce, numpy as np\n"
            "class HugeSequence:\n"
            "    def __len__(self):\n"
            "        return sys.maxsize\n"
            "    def __getitem__(self, index):\n"
            "        raise IndexError(index)\n"
            "def require_value_error(callback):\n"
            "    try:\n"
            "        callback()\n"
            "    except ValueError:\n"
            "        return\n"
            "    except Exception as exc:\n"
            "        raise AssertionError(f'unexpected exception: {exc!r}') from exc\n"
            "    raise AssertionError('call did not raise ValueError')\n"
            "gt = np.zeros((10, 10), dtype=np.uint8)\n"
            "ske = np.zeros((10, 10), dtype=bool)\n"
            "require_value_error(\n"
            "    lambda: turbo_hce.relax_HCE(gt, gt, ske, epsilon=-1.0)\n"
            ")\n"
            "require_value_error(\n"
            "    lambda: turbo_hce.approximate_RDP(\n"
            "        [np.zeros((5, 2), dtype=np.int32)], epsilon=-0.5\n"
            "    )\n"
            ")\n"
            "require_value_error(lambda: turbo_hce.approximate_RDP(HugeSequence()))\n"
            "require_value_error(\n"
            "    lambda: turbo_hce.filter_bdy_cond(HugeSequence(), gt, gt)\n"
            ")\n"
        )
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(
            proc.returncode,
            0,
            f"Subprocess failed with code {proc.returncode}:\n{proc.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
