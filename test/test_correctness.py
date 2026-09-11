"""Full correctness tests comparing turbo_hce against Python reference on skin dataset."""

import os
import sys
import time
import unittest
import cv2 as cv
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import turbo_hce
from utils import hce_metric_main as py_hce
from utils.session import Session

MODEL_PATH = "/home/samuel/ai_data/skin_models/sunet/sunet.onnx"
IMAGE_DIR = "/home/samuel/da/skindataset/images"
MASK_DIR = "/home/samuel/da/skindataset/masks"
TEST_SAMPLES = [f"{i:05d}.png" for i in range(1, 11)]  # 1333 or so for the full test.


class TestSyntheticIntegrationCorrectness(unittest.TestCase):
    """Deterministic integration tests verifying correctness without external dataset files."""

    @staticmethod
    def _create_synthetic_case(seed: int, size: int = 128):
        rng = np.random.default_rng(seed)
        gt = np.zeros((size, size), dtype=np.uint8)
        pred = np.zeros((size, size), dtype=np.uint8)

        # Draw base structures: circles, polygons, holes
        cv.circle(gt, (size // 3, size // 3), size // 5, 255, -1)
        cv.rectangle(gt, (size // 2, size // 2), (size - 15, size - 20), 255, -1)
        cv.circle(gt, (size * 2 // 3, size * 2 // 3), size // 10, 0, -1)

        cv.circle(pred, (size // 3 + 4, size // 3 + 2), size // 5 - 1, 255, -1)
        cv.rectangle(pred, (size // 2 + 5, size // 2 - 5), (size - 25, size - 25), 255, -1)
        cv.circle(pred, (size * 2 // 3, size * 2 // 3), size // 8, 0, -1)
        cv.circle(pred, (size // 6, size * 4 // 5), size // 12, 255, -1)

        ske = skeletonize(gt > 128)
        return gt, pred, ske

    def test_deterministic_synthetic_shapes(self):
        """Compare relax_HCE output across synthetic fixtures with various parameters."""
        for seed in (42, 100, 12345):
            for size in (96, 128):
                gt, pred, ske = self._create_synthetic_case(seed=seed, size=size)
                for relax in (0, 3, 5):
                    for eps in (0.5, 1.5, 3.0):
                        with self.subTest(seed=seed, size=size, relax=relax, eps=eps):
                            py_out = py_hce.relax_HCE(gt, pred, ske, relax=relax, epsilon=eps)
                            c_out = turbo_hce.relax_HCE(gt, pred, ske, relax=relax, epsilon=eps)
                            self.assertEqual(py_out, c_out)

    def test_known_fixed_fixture_values(self):
        """Verify relax_HCE against fixed precomputed output for seed 12345."""
        gt, pred, ske = self._create_synthetic_case(seed=12345, size=128)
        c_out = turbo_hce.relax_HCE(gt, pred, ske, relax=3, epsilon=1.5)
        py_out = py_hce.relax_HCE(gt, pred, ske, relax=3, epsilon=1.5)
        self.assertEqual(c_out, (5, 1.0, 28, 0.0))
        self.assertEqual(py_out, c_out)

    def test_synthetic_helpers_correctness(self):
        """Verify filter_bdy_cond and approximate_RDP match on synthetic contours."""
        gt, pred, ske = self._create_synthetic_case(seed=42, size=128)
        kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
        union = np.logical_or(gt > 128, pred > 128).astype(np.uint8)
        union_erode = cv.erode(union, kernel, iterations=3)
        fp = ((pred > 128) & ~(gt > 128)).astype(np.uint8)
        fp_relaxed = np.logical_and(fp, union_erode).astype(np.uint8)

        ctrs, _ = cv.findContours(fp_relaxed.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        cond = (gt > 128).astype(np.uint8)

        py_bdies, py_indep = py_hce.filter_bdy_cond(ctrs, fp_relaxed, cond)
        c_bdies, c_indep = turbo_hce.filter_bdy_cond(ctrs, fp_relaxed, cond)

        self.assertEqual(py_indep, c_indep)
        self.assertEqual(len(py_bdies), len(c_bdies))
        for pb, cb in zip(py_bdies, c_bdies):
            np.testing.assert_array_equal(pb, cb)

        py_rdp = py_hce.approximate_RDP(py_bdies, epsilon=2.0)
        c_rdp = turbo_hce.approximate_RDP(c_bdies, epsilon=2.0)
        self.assertEqual(py_rdp[2], c_rdp[2])
        self.assertEqual(py_rdp[1], c_rdp[1])


class TestCorrectnessTurboHCE(unittest.TestCase):
    """Full correctness test comparing Python and C implementations on real data."""

    @classmethod
    def setUpClass(cls):
        """Initialize ONNX inference session and verify test paths exist."""
        cls.has_resources = (
            os.path.exists(MODEL_PATH)
            and os.path.isdir(IMAGE_DIR)
            and os.path.isdir(MASK_DIR)
        )
        if cls.has_resources:
            cls.session = Session(model_path=MODEL_PATH)

    def setUp(self):
        """Skip tests if model or dataset paths are unavailable."""
        if not self.has_resources:
            self.skipTest("Model or dataset resources not found.")

    def test_dataset_samples_correctness(self):
        """Compare relax_HCE on multiple dataset images inferenced through Session."""
        py_total_time = 0.0
        c_total_time = 0.0

        for filename in TEST_SAMPLES:
            img_path = os.path.join(IMAGE_DIR, filename)
            mask_path = os.path.join(MASK_DIR, filename)

            with self.subTest(sample=filename):
                self.assertTrue(os.path.exists(img_path), f"Missing image {img_path}")
                self.assertTrue(os.path.exists(mask_path), f"Missing mask {mask_path}")

                pil_img = Image.open(img_path)
                gt = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)

                result = np.array(
                    self.session.remove(pil_img, do_sigmoid=False, simple_norm=True)
                )
                pred = result[:, :, 3]
                ske = skeletonize(gt > 128)

                t0 = time.perf_counter()
                py_out = py_hce.relax_HCE(gt, pred, ske)
                t1 = time.perf_counter()

                t2 = time.perf_counter()
                c_out = turbo_hce.relax_HCE(gt, pred, ske)
                t3 = time.perf_counter()

                py_total_time += t1 - t0
                c_total_time += t3 - t2

                self.assertEqual(
                    py_out,
                    c_out,
                    f"Mismatch on {filename}: py={py_out} vs c={c_out}",
                )

        speedup = py_total_time / max(c_total_time, 1e-6)
        print(
            f"\nCorrectness verified on {len(TEST_SAMPLES)} samples."
            f" Py time: {py_total_time:.3f}s, C time: {c_total_time:.3f}s"
            f" (Speedup: {speedup:.2f}x)"
        )

    def test_helpers_on_real_sample(self):
        """Verify filter_bdy_cond and approximate_RDP on contours from sample 00001."""
        mask_path = os.path.join(MASK_DIR, "00001.png")
        img_path = os.path.join(IMAGE_DIR, "00001.png")

        pil_img = Image.open(img_path)
        gt = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        result = np.array(
            self.session.remove(pil_img, do_sigmoid=False, simple_norm=True)
        )
        pred = result[:, :, 3]

        kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
        union = np.logical_or(gt > 128, pred > 128).astype(np.uint8)
        union_erode = cv.erode(union, kernel, iterations=5)
        fp = ((pred > 128) & ~(gt > 128)).astype(np.uint8)
        fp_relaxed = np.logical_and(fp, union_erode).astype(np.uint8)

        ctrs, _ = cv.findContours(
            fp_relaxed.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE
        )
        cond = (gt > 128).astype(np.uint8)

        py_bdies, py_indep = py_hce.filter_bdy_cond(ctrs, fp_relaxed, cond)
        c_bdies, c_indep = turbo_hce.filter_bdy_cond(ctrs, fp_relaxed, cond)

        self.assertEqual(py_indep, c_indep)
        self.assertEqual(len(py_bdies), len(c_bdies))
        for pb, cb in zip(py_bdies, c_bdies):
            np.testing.assert_array_equal(pb, cb)

        py_rdp = py_hce.approximate_RDP(py_bdies, epsilon=2.0)
        c_rdp = turbo_hce.approximate_RDP(c_bdies, epsilon=2.0)

        self.assertEqual(py_rdp[2], c_rdp[2])
        self.assertEqual(py_rdp[1], c_rdp[1])
        self.assertEqual(len(py_rdp[0]), len(c_rdp[0]))
        for pr, cr in zip(py_rdp[0], c_rdp[0]):
            np.testing.assert_array_equal(pr, cr)


if __name__ == "__main__":
    unittest.main()
