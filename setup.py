import os
import platform
import subprocess
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import numpy as np

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
OPENCV_SRC = os.path.join(REPO_ROOT, "opencv")


def get_opencv_build_dir() -> str:
    """Determine the OpenCV build directory, prioritizing existing build caches."""
    test_cache = os.path.join(REPO_ROOT, "build", "test_opencv")
    default_dir = os.path.join(REPO_ROOT, "build", "opencv")
    lib_name = "libopencv_imgproc.a" if sys.platform != "win32" else "opencv_imgproc.lib"
    if os.path.exists(os.path.join(test_cache, "lib", lib_name)):
        return test_cache
    return default_dir


def build_opencv_if_needed(opencv_build_dir: str):
    """Build OpenCV static libraries via CMake if not already present."""
    lib_prefix = "" if sys.platform == "win32" else "lib"
    lib_ext = ".lib" if sys.platform == "win32" else ".a"

    core_lib = os.path.join(opencv_build_dir, "lib", f"{lib_prefix}opencv_core{lib_ext}")
    imgproc_lib = os.path.join(opencv_build_dir, "lib", f"{lib_prefix}opencv_imgproc{lib_ext}")
    geometry_lib = os.path.join(opencv_build_dir, "lib", f"{lib_prefix}opencv_geometry{lib_ext}")

    if os.path.exists(core_lib) and os.path.exists(imgproc_lib) and os.path.exists(geometry_lib):
        return

    os.makedirs(opencv_build_dir, exist_ok=True)
    cmake_args = [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DBUILD_LIST=core,imgproc,geometry",
        "-DBUILD_opencv_apps=OFF",
        "-DBUILD_DOCS=OFF",
        "-DBUILD_TESTS=OFF",
        "-DBUILD_PERF_TESTS=OFF",
        "-DBUILD_EXAMPLES=OFF",
        "-DWITH_CUDA=OFF",
        "-DWITH_FFMPEG=OFF",
        "-DWITH_GSTREAMER=OFF",
        "-DWITH_GTK=OFF",
        "-DWITH_JASPER=OFF",
        "-DWITH_JPEG=OFF",
        "-DWITH_OPENEXR=OFF",
        "-DWITH_OPENJPEG=OFF",
        "-DWITH_PNG=OFF",
        "-DWITH_PROTOBUF=OFF",
        "-DWITH_TIFF=OFF",
        "-DWITH_WEBP=OFF",
        "-DWITH_IPP=OFF",
        "-DWITH_OPENCL=OFF",
        "-DBUILD_ZLIB=OFF",
        "-DWITH_1394=OFF",
        "-DWITH_EIGEN=OFF",
        "-DWITH_LAPACK=OFF",
        "-DBUILD_JAVA=OFF",
        "-DBUILD_opencv_python2=OFF",
        "-DBUILD_opencv_python3=OFF",
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
    ]

    subprocess.check_call(["cmake", OPENCV_SRC] + cmake_args, cwd=opencv_build_dir)
    num_jobs = str(os.cpu_count() or 4)
    subprocess.check_call(["cmake", "--build", ".", "-j", num_jobs], cwd=opencv_build_dir)


class CustomBuildExt(build_ext):
    """Custom build_ext command ensuring OpenCV submodule is built before compiling."""

    def run(self):
        opencv_build_dir = get_opencv_build_dir()
        build_opencv_if_needed(opencv_build_dir)
        super().run()


arch = platform.machine().lower()
is_x86 = "x86" in arch or "amd64" in arch or "i386" in arch or "i686" in arch
is_arm = "arm" in arch or "aarch64" in arch
extra_compile_args = []
extra_link_args = []

if sys.platform == "win32":
    extra_compile_args += ["/O2", "/arch:AVX2", "/Qpar", "/std:c++17"]
elif is_arm:
    extra_compile_args += ["-std=c++17", "-O3", "-flto", "-ffp-contract=fast"]
    extra_link_args += ["-flto"]
elif is_x86:
    extra_compile_args += [
        "-std=c++17",
        "-O3",
        "-march=x86-64",
        "-mavx2",
        "-msse4.2",
        "-flto",
        "-mfma",
        "-ffp-contract=fast",
    ]
    extra_link_args += ["-flto"]
else:
    extra_compile_args += ["-std=c++17", "-O3", "-flto"]
    extra_link_args += ["-flto"]

opencv_dir = get_opencv_build_dir()
lib_prefix = "" if sys.platform == "win32" else "lib"
lib_ext = ".lib" if sys.platform == "win32" else ".a"

extra_objects = [
    os.path.join(opencv_dir, "lib", f"{lib_prefix}opencv_imgproc{lib_ext}"),
    os.path.join(opencv_dir, "lib", f"{lib_prefix}opencv_geometry{lib_ext}"),
    os.path.join(opencv_dir, "lib", f"{lib_prefix}opencv_core{lib_ext}"),
]
ittnotify = os.path.join(opencv_dir, "3rdparty", "lib", f"{lib_prefix}ittnotify{lib_ext}")
if os.path.exists(ittnotify):
    extra_objects.append(ittnotify)

libraries = []
if sys.platform.startswith("linux"):
    libraries = ["z", "dl", "m", "pthread", "rt"]
elif sys.platform == "darwin":
    libraries = ["z", "m", "pthread"]
elif sys.platform == "win32":
    libraries = []

turbo_hce_ext = Extension(
    name="turbo_hce._turbo_hce",
    sources=[
        os.path.join("src", "turbo_hce", "_turbo_hce.cpp"),
        os.path.join("src", "turbo_hce", "skeletonize.cpp"),
    ],
    include_dirs=[
        os.path.join(REPO_ROOT, "src", "turbo_hce"),
        opencv_dir,
        os.path.join(OPENCV_SRC, "modules", "core", "include"),
        os.path.join(OPENCV_SRC, "modules", "imgproc", "include"),
        os.path.join(OPENCV_SRC, "modules", "geometry", "include"),
        np.get_include(),
    ],
    extra_objects=extra_objects,
    libraries=libraries,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c++",
)

if __name__ == "__main__":
    setup(
        cmdclass={"build_ext": CustomBuildExt},
        ext_modules=[turbo_hce_ext],
    )
