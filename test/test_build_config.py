"""Tests for platform-specific native build configuration."""

import tempfile
import unittest
from pathlib import Path

from build_helpers import find_windows_static_library, libraries_exist


class TestBuildConfig(unittest.TestCase):
    """Verify native library discovery without compiling OpenCV."""

    def test_windows_versioned_archive_discovery(self) -> None:
        """Find versioned OpenCV and ITT archives emitted by Visual Studio builds."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_dir = Path(temp_dir)
            opencv_lib_dir = build_dir / "lib" / "Release"
            third_party_lib_dir = build_dir / "3rdparty" / "lib" / "Release"
            opencv_lib_dir.mkdir(parents=True)
            third_party_lib_dir.mkdir(parents=True)

            imgproc_path = opencv_lib_dir / "libopencv_imgproc510.a"
            ittnotify_path = third_party_lib_dir / "libittnotify.a"
            imgproc_path.touch()
            ittnotify_path.touch()

            self.assertEqual(
                find_windows_static_library(str(build_dir), "opencv_imgproc"),
                str(imgproc_path),
            )
            self.assertEqual(
                find_windows_static_library(str(build_dir), "ittnotify"),
                str(ittnotify_path),
            )

    def test_required_libraries_must_exist(self) -> None:
        """Do not treat unresolved or not-yet-built library paths as a valid cache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            existing_path = Path(temp_dir) / "library.a"
            existing_path.touch()

            self.assertTrue(libraries_exist(str(existing_path)))
            self.assertFalse(libraries_exist(None))
            self.assertFalse(
                libraries_exist(str(existing_path), str(existing_path) + ".missing")
            )


if __name__ == "__main__":
    unittest.main()
