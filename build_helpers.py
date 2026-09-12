"""Helpers shared by the native extension build and its tests."""

import glob
import os


def find_windows_static_library(opencv_build_dir: str, library_name: str) -> str | None:
    """Find a release static library produced by a Visual Studio OpenCV build."""
    library_dirs = (
        os.path.join(opencv_build_dir, "lib", "Release"),
        os.path.join(opencv_build_dir, "lib"),
        os.path.join(opencv_build_dir, "3rdparty", "lib", "Release"),
        os.path.join(opencv_build_dir, "3rdparty", "lib"),
    )
    for library_dir in library_dirs:
        patterns = (
            f"{library_name}*.lib",
            f"lib{library_name}*.lib",
            f"{library_name}*.a",
            f"lib{library_name}*.a",
        )
        matches = sorted(
            library_path
            for pattern in patterns
            for library_path in glob.glob(os.path.join(library_dir, pattern))
        )
        release_matches = [
            library_path
            for library_path in matches
            if not os.path.basename(library_path).endswith(("d.lib", "d.a"))
        ]
        if release_matches:
            return release_matches[0]
    return None


def libraries_exist(*library_paths: str | None) -> bool:
    """Return whether every required static library exists."""
    return all(
        library_path is not None and os.path.exists(library_path)
        for library_path in library_paths
    )
