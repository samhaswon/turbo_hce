"""Differential tests for the native 2D skeletonize binding."""

import unittest

import numpy as np
from skimage.morphology import skeletonize as skimage_skeletonize

import turbo_hce


class TestSkeletonize(unittest.TestCase):
    """Compare the native binding to scikit-image's default 2D path."""

    @staticmethod
    def _native(image: np.ndarray) -> np.ndarray:
        """Invoke the planned public API.

        :param image: Input image to skeletonize.
        :return: Native skeleton image.
        """
        return turbo_hce.skeletonize(image)

    def assert_matches_reference(self, image: np.ndarray) -> None:
        """Assert exact output equivalence and public result invariants.

        :param image: A two-dimensional input image.
        """
        actual = self._native(image)
        expected = skimage_skeletonize(image)
        self.assertEqual(actual.dtype, np.dtype(bool))
        self.assertEqual(actual.shape, image.shape)
        np.testing.assert_array_equal(actual, expected)

    def test_empty_and_degenerate_images(self) -> None:
        """Match reference behavior for empty, one-row, and one-column inputs."""
        for image in (
            np.empty((0, 0), dtype=bool),
            np.empty((0, 7), dtype=bool),
            np.empty((5, 0), dtype=bool),
            np.array([[0, 1, 1, 1, 0]], dtype=bool),
            np.array([[0], [1], [1], [1], [0]], dtype=bool),
        ):
            with self.subTest(shape=image.shape):
                self.assert_matches_reference(image)

    def test_representative_topologies(self) -> None:
        """Match connected components, rings, diagonal paths, and border touches."""
        square = np.zeros((17, 17), dtype=bool)
        square[3:14, 4:13] = True

        ring = np.zeros((17, 17), dtype=bool)
        ring[2:15, 2:15] = True
        ring[5:12, 5:12] = False

        diagonal = np.eye(17, dtype=bool)
        diagonal |= np.eye(17, k=1, dtype=bool)

        border_touching = np.zeros((17, 17), dtype=bool)
        border_touching[:, :5] = True
        border_touching[8:, 5:12] = True

        for name, image in {
            "square": square,
            "ring": ring,
            "diagonal": diagonal,
            "border_touching": border_touching,
        }.items():
            with self.subTest(shape=name):
                self.assert_matches_reference(image)

    def test_randomized_binary_masks(self) -> None:
        """Match the reference on deterministic random masks and densities."""
        generator = np.random.default_rng(971)
        for shape in ((2, 3), (7, 11), (32, 29), (63, 64)):
            for density in (0.05, 0.3, 0.65, 0.95):
                with self.subTest(shape=shape, density=density):
                    self.assert_matches_reference(generator.random(shape) < density)

    def test_non_contiguous_input(self) -> None:
        """Accept non-contiguous views without changing their logical image."""
        base = np.zeros((31, 37), dtype=np.uint8)
        base[3:26, 4:31] = 255
        base[9:20, 12:23] = 0
        image = base[::2, 1::2]
        self.assertFalse(image.flags.c_contiguous)
        self.assert_matches_reference(image)

    def test_nonzero_numeric_values_are_foreground(self) -> None:
        """Use scikit-image's nonzero-to-foreground semantics for supported dtypes."""
        values = np.array(
            [[0, 0, 0, 0, 0], [0, -2, 9, 0, 0], [0, 3, 5, 1, 0],
             [0, 0, 7, 0, 0], [0, 0, 0, 0, 0]],
            dtype=np.int16,
        )
        for image in (values, values.astype(np.uint8), values.astype(np.float32)):
            with self.subTest(dtype=image.dtype):
                self.assert_matches_reference(image)

    def test_simd_lane_boundary_dimensions(self) -> None:
        """Match reference on dimensions surrounding 32-byte SIMD lane boundaries."""
        generator = np.random.default_rng(1337)
        test_sizes = (1, 2, 15, 16, 31, 32, 33, 63, 64, 65)
        for h in (5, 16, 33):
            for w in test_sizes:
                with self.subTest(shape=(h, w)):
                    mask = generator.random((h, w)) > 0.4
                    self.assert_matches_reference(mask)

    def test_translated_small_roi(self) -> None:
        """Verify small ROIs positioned at corners and center of large canvas."""
        canvas_h, canvas_w = 256, 256
        positions = [(0, 0), (120, 120), (canvas_h - 30, canvas_w - 30)]
        for r_pos, c_pos in positions:
            with self.subTest(position=(r_pos, c_pos)):
                canvas = np.zeros((canvas_h, canvas_w), dtype=bool)
                canvas[r_pos:r_pos + 20, c_pos:c_pos + 20] = True
                canvas[r_pos + 5:r_pos + 15, c_pos + 5:c_pos + 15] = False
                self.assert_matches_reference(canvas)

    def test_multithreaded_concurrent_execution(self) -> None:
        """Verify GIL release and thread safety under concurrent calls."""
        from concurrent.futures import ThreadPoolExecutor

        generator = np.random.default_rng(2026)
        masks = [generator.random((128, 128)) > 0.5 for _ in range(16)]
        expected = [skimage_skeletonize(m) for m in masks]

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(turbo_hce.skeletonize, masks))

        for actual_res, exp_res in zip(results, expected):
            np.testing.assert_array_equal(actual_res, exp_res)

    def test_rejects_non_two_dimensional_input(self) -> None:
        """Reject dimensions outside the prototype's explicitly 2D API."""
        for image in (np.zeros(8, dtype=bool), np.zeros((2, 3, 4), dtype=bool)):
            with self.subTest(shape=image.shape):
                with self.assertRaises(ValueError):
                    self._native(image)

    def test_rejects_object_dtype(self) -> None:
        """Reject object arrays rather than applying Python truthiness elementwise."""
        with self.assertRaises(TypeError):
            self._native(np.array([["foreground"]], dtype=object))


if __name__ == "__main__":
    unittest.main()
