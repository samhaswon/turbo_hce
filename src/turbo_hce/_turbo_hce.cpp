#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/geometry.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__GNUC__) || defined(__clang__)
#define TURBO_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define TURBO_RESTRICT __restrict
#else
#define TURBO_RESTRICT
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace {

static_assert(sizeof(cv::Point) == 2 * sizeof(int32_t), "cv::Point size mismatch");
static_assert(offsetof(cv::Point, x) == 0, "cv::Point x offset mismatch");
static_assert(offsetof(cv::Point, y) == sizeof(int32_t), "cv::Point y offset mismatch");
static_assert(std::is_trivially_copyable<cv::Point>::value, "cv::Point must be trivially copyable");

inline void compute_tp_fp_fn_union(
    const uint8_t* TURBO_RESTRICT p_gt,
    const uint8_t* TURBO_RESTRICT p_rs,
    uint8_t* TURBO_RESTRICT p_tp,
    uint8_t* TURBO_RESTRICT p_fp,
    uint8_t* TURBO_RESTRICT p_fn,
    uint8_t* TURBO_RESTRICT p_union,
    size_t total_pixels
) {
#if defined(__AVX2__)
    size_t i = 0;
    for (; i + 32 <= total_pixels; i += 32) {
        __m256i g = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p_gt + i));
        __m256i r = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p_rs + i));
        __m256i tp = _mm256_and_si256(g, r);
        __m256i fp = _mm256_andnot_si256(g, r);
        __m256i fn = _mm256_andnot_si256(r, g);
        __m256i un = _mm256_or_si256(g, r);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(p_tp + i), tp);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(p_fp + i), fp);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(p_fn + i), fn);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(p_union + i), un);
    }
    for (; i < total_pixels; ++i) {
        uint8_t g = p_gt[i];
        uint8_t r = p_rs[i];
        uint8_t tp = g & r;
        p_tp[i] = tp;
        p_fp[i] = r - tp;
        p_fn[i] = g - tp;
        p_union[i] = g | r;
    }
#else
    for (size_t i = 0; i < total_pixels; ++i) {
        uint8_t g = p_gt[i];
        uint8_t r = p_rs[i];
        uint8_t tp = g & r;
        p_tp[i] = tp;
        p_fp[i] = r - tp;
        p_fn[i] = g - tp;
        p_union[i] = g | r;
    }
#endif
}

inline void compute_complement(
    const uint8_t* TURBO_RESTRICT p_in,
    uint8_t* TURBO_RESTRICT p_out,
    size_t total_pixels
) {
#if defined(__AVX2__)
    size_t i = 0;
    const __m256i ones = _mm256_set1_epi8(1);
    for (; i + 32 <= total_pixels; i += 32) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p_in + i));
        __m256i res = _mm256_sub_epi8(ones, v);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(p_out + i), res);
    }
    for (; i < total_pixels; ++i) {
        p_out[i] = 1 - p_in[i];
    }
#else
    for (size_t i = 0; i < total_pixels; ++i) {
        p_out[i] = 1 - p_in[i];
    }
#endif
}

inline void merge_skeleton_fn(
    const uint8_t* TURBO_RESTRICT p_ske,
    const uint8_t* TURBO_RESTRICT p_tp,
    uint8_t* TURBO_RESTRICT p_fn,
    size_t total_pixels
) {
#if defined(__AVX2__)
    size_t i = 0;
    for (; i + 32 <= total_pixels; i += 32) {
        __m256i s = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p_ske + i));
        __m256i tp = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p_tp + i));
        __m256i fn_val = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p_fn + i));
        __m256i ske_term = _mm256_andnot_si256(tp, s);
        __m256i res = _mm256_or_si256(fn_val, ske_term);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(p_fn + i), res);
    }
    for (; i < total_pixels; ++i) {
        uint8_t s = p_ske[i];
        uint8_t tp = p_tp[i];
        p_fn[i] = p_fn[i] | (s & (1 - tp));
    }
#else
    for (size_t i = 0; i < total_pixels; ++i) {
        uint8_t s = p_ske[i];
        uint8_t tp = p_tp[i];
        p_fn[i] = p_fn[i] | (s & (1 - tp));
    }
#endif
}

inline void compute_cond_fp(
    const uint8_t* TURBO_RESTRICT p_tp,
    const uint8_t* TURBO_RESTRICT p_fn,
    uint8_t* TURBO_RESTRICT p_cond_fp,
    size_t total_pixels
) {
#if defined(__AVX2__)
    size_t i = 0;
    for (; i + 32 <= total_pixels; i += 32) {
        __m256i tp = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p_tp + i));
        __m256i fn_val = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p_fn + i));
        __m256i res = _mm256_or_si256(tp, fn_val);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(p_cond_fp + i), res);
    }
    for (; i < total_pixels; ++i) {
        p_cond_fp[i] = p_tp[i] | p_fn[i];
    }
#else
    for (size_t i = 0; i < total_pixels; ++i) {
        p_cond_fp[i] = p_tp[i] | p_fn[i];
    }
#endif
}

inline void compute_cond_fn(
    const uint8_t* TURBO_RESTRICT p_tp,
    const uint8_t* TURBO_RESTRICT p_fp,
    const uint8_t* TURBO_RESTRICT p_fn,
    uint8_t* TURBO_RESTRICT p_cond_fn,
    size_t total_pixels
) {
#if defined(__AVX2__)
    size_t i = 0;
    const __m256i ones = _mm256_set1_epi8(1);
    for (; i + 32 <= total_pixels; i += 32) {
        __m256i tp = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p_tp + i));
        __m256i fp = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p_fp + i));
        __m256i fn_val = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p_fn + i));
        __m256i any_set = _mm256_or_si256(tp, _mm256_or_si256(fp, fn_val));
        __m256i res = _mm256_sub_epi8(ones, any_set);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(p_cond_fn + i), res);
    }
    for (; i < total_pixels; ++i) {
        p_cond_fn[i] = 1 - (p_tp[i] | p_fp[i] | p_fn[i]);
    }
#else
    for (size_t i = 0; i < total_pixels; ++i) {
        p_cond_fn[i] = 1 - (p_tp[i] | p_fp[i] | p_fn[i]);
    }
#endif
}

inline void extract_u8_nonzero(
    const uint8_t* TURBO_RESTRICT src,
    uint8_t* TURBO_RESTRICT dst,
    size_t total_pixels
) {
#if defined(__AVX2__)
    size_t i = 0;
    const __m256i zero = _mm256_setzero_si256();
    const __m256i one = _mm256_set1_epi8(1);
    for (; i + 32 <= total_pixels; i += 32) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        __m256i is_zero = _mm256_cmpeq_epi8(v, zero);
        __m256i is_nonzero = _mm256_andnot_si256(is_zero, one);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), is_nonzero);
    }
    for (; i < total_pixels; ++i) {
        dst[i] = (src[i] != 0) ? 1 : 0;
    }
#else
    for (size_t i = 0; i < total_pixels; ++i) {
        dst[i] = (src[i] != 0) ? 1 : 0;
    }
#endif
}

inline void extract_u8_gt128(
    const uint8_t* TURBO_RESTRICT src,
    uint8_t* TURBO_RESTRICT dst,
    size_t total_pixels
) {
#if defined(__AVX2__)
    size_t i = 0;
    const __m256i thresh = _mm256_set1_epi8(static_cast<char>(128));
    const __m256i one = _mm256_set1_epi8(1);
    const __m256i zero = _mm256_setzero_si256();
    for (; i + 32 <= total_pixels; i += 32) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        __m256i v_signed = _mm256_xor_si256(v, thresh);
        __m256i is_gt = _mm256_cmpgt_epi8(v_signed, zero);
        __m256i res = _mm256_and_si256(is_gt, one);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), res);
    }
    for (; i < total_pixels; ++i) {
        dst[i] = (src[i] > 128) ? 1 : 0;
    }
#else
    for (size_t i = 0; i < total_pixels; ++i) {
        dst[i] = (src[i] > 128) ? 1 : 0;
    }
#endif
}

template <typename T, typename Pred>
inline void extract_channel0_loop(
    const T* TURBO_RESTRICT src,
    uint8_t* TURBO_RESTRICT dst,
    size_t total_pixels,
    int channels,
    Pred&& pred
) {
    if (channels == 1) {
        for (size_t i = 0; i < total_pixels; ++i) {
            dst[i] = pred(src[i]) ? 1 : 0;
        }
    } else if (channels == 3) {
        for (size_t i = 0, s = 0; i < total_pixels; ++i, s += 3) {
            dst[i] = pred(src[s]) ? 1 : 0;
        }
    } else if (channels == 4) {
        for (size_t i = 0, s = 0; i < total_pixels; ++i, s += 4) {
            dst[i] = pred(src[s]) ? 1 : 0;
        }
    } else {
        const size_t ch = static_cast<size_t>(channels);
        for (size_t i = 0, s = 0; i < total_pixels; ++i, s += ch) {
            dst[i] = pred(src[s]) ? 1 : 0;
        }
    }
}

class ScopedPyRef {
    PyObject* ptr_;

public:
    explicit ScopedPyRef(PyObject* p = nullptr) : ptr_(p) {}
    ~ScopedPyRef() { Py_XDECREF(ptr_); }
    ScopedPyRef(const ScopedPyRef&) = delete;
    ScopedPyRef& operator=(const ScopedPyRef&) = delete;
    ScopedPyRef(ScopedPyRef&& o) noexcept : ptr_(o.release()) {}
    ScopedPyRef& operator=(ScopedPyRef&& o) noexcept {
        if (this != &o) {
            Py_XDECREF(ptr_);
            ptr_ = o.release();
        }
        return *this;
    }

    PyObject* get() const { return ptr_; }
    PyObject* release() {
        PyObject* tmp = ptr_;
        ptr_ = nullptr;
        return tmp;
    }
    void reset(PyObject* p = nullptr) {
        Py_XDECREF(ptr_);
        ptr_ = p;
    }
    operator PyObject*() const { return ptr_; }
};

cv::Mat get_cross_kernel() {
    return cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
}

std::pair<std::vector<std::vector<cv::Point>>, double> filter_bdy_cond_impl(
    const std::vector<std::vector<cv::Point>>& bdy_,
    const cv::Mat& mask,
    const cv::Mat& cond,
    const cv::Mat& kernel
) {
    cv::Mat cond_u8;
    if (cond.type() != CV_8U) {
        cond.convertTo(cond_u8, CV_8U);
    } else {
        cond_u8 = cond;
    }

    cv::Mat cond_dilated;
    cv::dilate(cond_u8, cond_dilated, kernel);

    cv::Mat mask_u8;
    if (mask.type() != CV_8U) {
        mask.convertTo(mask_u8, CV_8U);
    } else {
        mask_u8 = mask;
    }

    cv::Mat labels;
    int num_labels = cv::connectedComponents(mask_u8, labels, 8, CV_32S);

    std::vector<uint8_t> indep(num_labels > 0 ? num_labels : 1, 1);
    if (num_labels > 0) {
        indep[0] = 0;
    }

    int h = cond_u8.rows;
    int w = cond_u8.cols;
    cv::Mat ind_map = cv::Mat::zeros(h, w, CV_8U);
    std::vector<std::vector<cv::Point>> boundaries;

    const uint8_t* TURBO_RESTRICT p_cond_dilated = cond_dilated.ptr<uint8_t>();
    uint8_t* TURBO_RESTRICT p_ind_map = ind_map.ptr<uint8_t>();
    const int32_t* TURBO_RESTRICT p_labels = labels.ptr<int32_t>();

    for (size_t i = 0; i < bdy_.size(); ++i) {
        std::vector<std::vector<cv::Point>> tmp_bdies;
        std::vector<cv::Point> tmp_bdy;
        const auto& contour = bdy_[i];

        for (size_t j = 0; j < contour.size(); ++j) {
            int c = contour[j].x;
            int r = contour[j].y;

            if (r < 0 || r >= h || c < 0 || c >= w) {
                continue;
            }

            size_t pixel_idx = static_cast<size_t>(r) * w + static_cast<size_t>(c);
            if (p_cond_dilated[pixel_idx] == 0 || p_ind_map[pixel_idx] != 0) {
                if (!tmp_bdy.empty()) {
                    tmp_bdies.push_back(std::move(tmp_bdy));
                    tmp_bdy.clear();
                }
                continue;
            }

            tmp_bdy.push_back(cv::Point(c, r));
            p_ind_map[pixel_idx] = 1;
            int lbl = p_labels[pixel_idx];
            if (lbl >= 0 && lbl < num_labels) {
                indep[lbl] = 0;
            }
        }

        if (!tmp_bdy.empty()) {
            tmp_bdies.push_back(std::move(tmp_bdy));
        }

        if (tmp_bdies.size() > 1) {
            int first_x = tmp_bdies.front().front().x;
            int first_y = tmp_bdies.front().front().y;
            int last_x = tmp_bdies.back().back().x;
            int last_y = tmp_bdies.back().back().y;

            int dx = std::abs(first_x - last_x);
            int dy = std::abs(first_y - last_y);

            if ((dx == 1 && dy == 0) || (dx == 0 && dy == 1) || (dx == 1 && dy == 1)) {
                const auto& first = tmp_bdies.front();
                tmp_bdies.back().insert(tmp_bdies.back().end(), first.rbegin(), first.rend());
                tmp_bdies.erase(tmp_bdies.begin());
            }
        }

        for (auto& b : tmp_bdies) {
            if (!b.empty()) {
                boundaries.push_back(std::move(b));
            }
        }
    }

    int64_t indep_sum = 0;
    const uint8_t* TURBO_RESTRICT indep_ptr = indep.data();
    for (int i = 1; i < num_labels; ++i) {
        indep_sum += indep_ptr[i];
    }

    return {boundaries, static_cast<double>(indep_sum)};
}

int approximate_RDP_cnt(const std::vector<std::vector<cv::Point>>& boundaries, double epsilon) {
    int pixel_cnt = 0;
    std::vector<cv::Point> approx;
    for (const auto& b : boundaries) {
        if (!b.empty()) {
            cv::approxPolyDP(b, approx, epsilon, false);
            pixel_cnt += static_cast<int>(approx.size());
        }
    }
    return pixel_cnt;
}

struct HCEOutput {
    int poly_FP_point_cnt = 0;
    double indep_cnt_FP = 0.0;
    int poly_FN_point_cnt = 0;
    double indep_cnt_FN = 0.0;
};

HCEOutput relax_HCE_impl(
    const cv::Mat& gt_bin,
    const cv::Mat& rs_bin,
    const cv::Mat& ske_bin,
    int relax,
    double epsilon
) {
    int h = gt_bin.rows;
    int w = gt_bin.cols;
    size_t total_pixels = static_cast<size_t>(h) * w;
    cv::Mat kernel = get_cross_kernel();

    cv::Mat TP(h, w, CV_8U);
    cv::Mat FP(h, w, CV_8U);
    cv::Mat FN(h, w, CV_8U);
    cv::Mat Union(h, w, CV_8U);

    compute_tp_fp_fn_union(
        gt_bin.data, rs_bin.data,
        TP.data, FP.data, FN.data, Union.data,
        total_pixels
    );

    cv::Mat Union_erode;
    if (relax > 0) {
        cv::erode(Union, Union_erode, kernel, cv::Point(-1, -1), relax);
    } else {
        Union_erode = Union;
    }

    cv::Mat FP_;
    cv::bitwise_and(FP, Union_erode, FP_);

    if (relax > 0) {
        cv::Mat not_TP_or_FN(h, w, CV_8U);
        compute_complement(gt_bin.data, not_TP_or_FN.data, total_pixels);

        for (int i = 0; i < relax; ++i) {
            cv::dilate(FP_, FP_, kernel);
            cv::bitwise_and(FP_, not_TP_or_FN, FP_);
        }
        cv::bitwise_and(FP, FP_, FP_);
    }

    cv::Mat FN_;
    cv::bitwise_and(FN, Union_erode, FN_);

    if (relax > 0) {
        cv::Mat not_TP_or_FP(h, w, CV_8U);
        compute_complement(rs_bin.data, not_TP_or_FP.data, total_pixels);

        for (int i = 0; i < relax; ++i) {
            cv::dilate(FN_, FN_, kernel);
            cv::bitwise_and(FN_, not_TP_or_FP, FN_);
        }
        cv::bitwise_and(FN, FN_, FN_);
    }

    merge_skeleton_fn(ske_bin.data, TP.data, FN_.data, total_pixels);

    std::vector<std::vector<cv::Point>> ctrs_FP;
    std::vector<cv::Vec4i> hier_FP;
    cv::findContours(FP_, ctrs_FP, hier_FP, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

    cv::Mat cond_FP(h, w, CV_8U);
    compute_cond_fp(TP.data, FN_.data, cond_FP.data, total_pixels);

    auto [bdies_FP, indep_cnt_FP] = filter_bdy_cond_impl(ctrs_FP, FP_, cond_FP, kernel);

    std::vector<std::vector<cv::Point>> ctrs_FN;
    std::vector<cv::Vec4i> hier_FN;
    cv::findContours(FN_, ctrs_FN, hier_FN, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

    cv::Mat cond_FN(h, w, CV_8U);
    compute_cond_fn(TP.data, FP_.data, FN_.data, cond_FN.data, total_pixels);

    auto [bdies_FN, indep_cnt_FN] = filter_bdy_cond_impl(ctrs_FN, FN_, cond_FN, kernel);

    int poly_FP_point_cnt = approximate_RDP_cnt(bdies_FP, epsilon);
    int poly_FN_point_cnt = approximate_RDP_cnt(bdies_FN, epsilon);

    return {poly_FP_point_cnt, indep_cnt_FP, poly_FN_point_cnt, indep_cnt_FN};
}

bool extract_binarized_2d(
    PyObject* py_obj,
    const char* arg_name,
    bool is_skeleton,
    bool allow_3d_channel0,
    cv::Mat& out_mat
) {
    if (!PyArray_Check(py_obj)) {
        PyErr_Format(PyExc_TypeError, "%s must be a numpy ndarray", arg_name);
        return false;
    }

    PyArrayObject* orig_arr = reinterpret_cast<PyArrayObject*>(py_obj);
    int orig_ndim = PyArray_NDIM(orig_arr);

    if (allow_3d_channel0) {
        if (orig_ndim != 2 && orig_ndim != 3) {
            PyErr_Format(PyExc_ValueError, "%s must be a 2D or 3D array, got %d dimensions",
                         arg_name, orig_ndim);
            return false;
        }
    } else {
        if (orig_ndim != 2) {
            PyErr_Format(PyExc_ValueError, "%s must be a 2D array, got %d dimensions",
                         arg_name, orig_ndim);
            return false;
        }
    }

    npy_intp* orig_dims = PyArray_SHAPE(orig_arr);
    if (orig_dims[0] <= 0 || orig_dims[1] <= 0) {
        PyErr_Format(PyExc_ValueError, "%s dimensions must be positive, got (%ld, %ld)",
                     arg_name, static_cast<long>(orig_dims[0]), static_cast<long>(orig_dims[1]));
        return false;
    }
    if (orig_dims[0] > std::numeric_limits<int>::max() ||
        orig_dims[1] > std::numeric_limits<int>::max()) {
        PyErr_Format(PyExc_ValueError, "%s dimensions exceed maximum supported size", arg_name);
        return false;
    }
    if (orig_ndim == 3 && orig_dims[2] <= 0) {
        PyErr_Format(PyExc_ValueError, "%s channel dimension must be >= 1", arg_name);
        return false;
    }

    // Require C-contiguous, native byte order, and aligned storage
    ScopedPyRef contig(PyArray_CheckFromAny(
        py_obj, nullptr, 2, allow_3d_channel0 ? 3 : 2,
        NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_NOTSWAPPED | NPY_ARRAY_ALIGNED, nullptr
    ));
    if (!contig) {
        return false;
    }

    int h = static_cast<int>(orig_dims[0]);
    int w = static_cast<int>(orig_dims[1]);
    int channels = (orig_ndim == 3) ? static_cast<int>(orig_dims[2]) : 1;
    size_t total_pixels = static_cast<size_t>(h) * w;

    out_mat.create(h, w, CV_8U);
    uint8_t* dst = out_mat.data;

    int type = PyArray_TYPE(reinterpret_cast<PyArrayObject*>(contig.get()));
    const void* raw_data = PyArray_DATA(reinterpret_cast<PyArrayObject*>(contig.get()));

    if (is_skeleton) {
        switch (type) {
            case NPY_BOOL:
            case NPY_UINT8: {
                const uint8_t* src = static_cast<const uint8_t*>(raw_data);
                if (channels == 1) {
                    extract_u8_nonzero(src, dst, total_pixels);
                } else {
                    extract_channel0_loop(src, dst, total_pixels, channels, [](uint8_t val) {
                        return val != 0;
                    });
                }
                break;
            }
            case NPY_INT8: {
                const int8_t* src = static_cast<const int8_t*>(raw_data);
                extract_channel0_loop(src, dst, total_pixels, channels, [](int8_t val) {
                    return val != 0;
                });
                break;
            }
            case NPY_UINT16: {
                const uint16_t* src = static_cast<const uint16_t*>(raw_data);
                extract_channel0_loop(src, dst, total_pixels, channels, [](uint16_t val) {
                    return val != 0;
                });
                break;
            }
            case NPY_INT16: {
                const int16_t* src = static_cast<const int16_t*>(raw_data);
                extract_channel0_loop(src, dst, total_pixels, channels, [](int16_t val) {
                    return val != 0;
                });
                break;
            }
            case NPY_UINT32: {
                const uint32_t* src = static_cast<const uint32_t*>(raw_data);
                extract_channel0_loop(src, dst, total_pixels, channels, [](uint32_t val) {
                    return val != 0;
                });
                break;
            }
            case NPY_INT32: {
                const int32_t* src = static_cast<const int32_t*>(raw_data);
                extract_channel0_loop(src, dst, total_pixels, channels, [](int32_t val) {
                    return val != 0;
                });
                break;
            }
            case NPY_UINT64: {
                const uint64_t* src = static_cast<const uint64_t*>(raw_data);
                extract_channel0_loop(src, dst, total_pixels, channels, [](uint64_t val) {
                    return val != 0;
                });
                break;
            }
            case NPY_INT64: {
                const int64_t* src = static_cast<const int64_t*>(raw_data);
                extract_channel0_loop(src, dst, total_pixels, channels, [](int64_t val) {
                    return val != 0;
                });
                break;
            }
            case NPY_FLOAT16: {
                ScopedPyRef f32_arr(reinterpret_cast<PyObject*>(
                    PyArray_Cast(reinterpret_cast<PyArrayObject*>(contig.get()), NPY_FLOAT32)
                ));
                if (!f32_arr) {
                    return false;
                }
                const float* src = static_cast<const float*>(
                    PyArray_DATA(reinterpret_cast<PyArrayObject*>(f32_arr.get()))
                );
                extract_channel0_loop(src, dst, total_pixels, channels, [](float val) {
                    return val != 0.0f;
                });
                break;
            }
            case NPY_FLOAT32: {
                const float* src = static_cast<const float*>(raw_data);
                extract_channel0_loop(src, dst, total_pixels, channels, [](float val) {
                    return val != 0.0f;
                });
                break;
            }
            case NPY_FLOAT64: {
                const double* src = static_cast<const double*>(raw_data);
                extract_channel0_loop(src, dst, total_pixels, channels, [](double val) {
                    return val != 0.0;
                });
                break;
            }
            default:
                PyErr_Format(PyExc_TypeError, "Unsupported dtype for %s", arg_name);
                return false;
        }
    } else {
        switch (type) {
            case NPY_BOOL:
            case NPY_INT8:
                std::memset(dst, 0, total_pixels);
                break;
            case NPY_UINT8: {
                const uint8_t* src = static_cast<const uint8_t*>(raw_data);
                if (channels == 1) {
                    extract_u8_gt128(src, dst, total_pixels);
                } else {
                    extract_channel0_loop(src, dst, total_pixels, channels, [](uint8_t val) {
                        return val > 128;
                    });
                }
                break;
            }
            case NPY_UINT16: {
                const uint16_t* src = static_cast<const uint16_t*>(raw_data);
                extract_channel0_loop(src, dst, total_pixels, channels, [](uint16_t val) {
                    return val > 128;
                });
                break;
            }
            case NPY_INT16: {
                const int16_t* src = static_cast<const int16_t*>(raw_data);
                extract_channel0_loop(src, dst, total_pixels, channels, [](int16_t val) {
                    return val > 128;
                });
                break;
            }
            case NPY_UINT32: {
                const uint32_t* src = static_cast<const uint32_t*>(raw_data);
                extract_channel0_loop(src, dst, total_pixels, channels, [](uint32_t val) {
                    return val > 128;
                });
                break;
            }
            case NPY_INT32: {
                const int32_t* src = static_cast<const int32_t*>(raw_data);
                extract_channel0_loop(src, dst, total_pixels, channels, [](int32_t val) {
                    return val > 128;
                });
                break;
            }
            case NPY_UINT64: {
                const uint64_t* src = static_cast<const uint64_t*>(raw_data);
                extract_channel0_loop(src, dst, total_pixels, channels, [](uint64_t val) {
                    return val > 128;
                });
                break;
            }
            case NPY_INT64: {
                const int64_t* src = static_cast<const int64_t*>(raw_data);
                extract_channel0_loop(src, dst, total_pixels, channels, [](int64_t val) {
                    return val > 128;
                });
                break;
            }
            case NPY_FLOAT16: {
                ScopedPyRef f32_arr(reinterpret_cast<PyObject*>(
                    PyArray_Cast(reinterpret_cast<PyArrayObject*>(contig.get()), NPY_FLOAT32)
                ));
                if (!f32_arr) {
                    return false;
                }
                const float* src = static_cast<const float*>(
                    PyArray_DATA(reinterpret_cast<PyArrayObject*>(f32_arr.get()))
                );
                extract_channel0_loop(src, dst, total_pixels, channels, [](float val) {
                    return val > 128.0f;
                });
                break;
            }
            case NPY_FLOAT32: {
                const float* src = static_cast<const float*>(raw_data);
                extract_channel0_loop(src, dst, total_pixels, channels, [](float val) {
                    return val > 128.0f;
                });
                break;
            }
            case NPY_FLOAT64: {
                const double* src = static_cast<const double*>(raw_data);
                extract_channel0_loop(src, dst, total_pixels, channels, [](double val) {
                    return val > 128.0;
                });
                break;
            }
            default:
                PyErr_Format(PyExc_TypeError, "Unsupported dtype for %s", arg_name);
                return false;
        }
    }

    return true;
}

bool parse_contours_list(
    PyObject* py_bdy,
    std::vector<std::vector<cv::Point>>& out_bdy,
    const char* arg_name = "bdy_"
) {
    if (!PySequence_Check(py_bdy)) {
        PyErr_Format(PyExc_TypeError, "%s must be a sequence of contours", arg_name);
        return false;
    }

    Py_ssize_t num_contours = PySequence_Size(py_bdy);
    if (num_contours < 0) {
        return false;
    }
    if (num_contours > 10000000) {
        PyErr_Format(PyExc_ValueError, "Sequence length %zd exceeds maximum supported contour count",
                     num_contours);
        return false;
    }

    out_bdy.clear();
    out_bdy.reserve(static_cast<size_t>(num_contours));

    for (Py_ssize_t i = 0; i < num_contours; ++i) {
        ScopedPyRef item(PySequence_GetItem(py_bdy, i));
        if (!item) {
            return false;
        }

        if (!PyArray_Check(item.get())) {
            PyErr_Format(PyExc_TypeError, "Contour at index %zd must be a numpy ndarray", i);
            return false;
        }

        PyArrayObject* orig_arr = reinterpret_cast<PyArrayObject*>(item.get());
        int ndim = PyArray_NDIM(orig_arr);
        npy_intp* dims = PyArray_SHAPE(orig_arr);

        if (ndim == 2) {
            if (dims[1] != 2) {
                PyErr_Format(PyExc_ValueError,
                    "Contour at index %zd with 2D shape must have shape (N, 2), got (%ld, %ld)",
                    i, static_cast<long>(dims[0]), static_cast<long>(dims[1]));
                return false;
            }
        } else if (ndim == 3) {
            if (dims[1] != 1 || dims[2] != 2) {
                PyErr_Format(PyExc_ValueError,
                    "Contour at index %zd with 3D shape must have shape (N, 1, 2), "
                    "got (%ld, %ld, %ld)",
                    i, static_cast<long>(dims[0]), static_cast<long>(dims[1]),
                    static_cast<long>(dims[2]));
                return false;
            }
        } else {
            PyErr_Format(PyExc_ValueError,
                "Contour at index %zd must have 2 or 3 dimensions, got %d", i, ndim);
            return false;
        }

        if (dims[0] < 0 || dims[0] > std::numeric_limits<int>::max() || dims[0] > 10000000) {
            PyErr_Format(PyExc_ValueError, "Contour at index %zd has invalid length %ld",
                         i, static_cast<long>(dims[0]));
            return false;
        }

        ScopedPyRef contig(PyArray_CheckFromAny(
            item.get(), PyArray_DescrFromType(NPY_INT32), 2, 3,
            NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_NOTSWAPPED | NPY_ARRAY_ALIGNED | NPY_ARRAY_FORCECAST,
            nullptr
        ));
        if (!contig) {
            return false;
        }

        npy_intp n_pts = dims[0];
        const int32_t* data = static_cast<const int32_t*>(
            PyArray_DATA(reinterpret_cast<PyArrayObject*>(contig.get()))
        );

        std::vector<cv::Point> contour(static_cast<size_t>(n_pts));
        if (n_pts > 0) {
            std::memcpy(contour.data(), data, static_cast<size_t>(n_pts) * sizeof(cv::Point));
        }

        out_bdy.push_back(std::move(contour));
    }
    return true;
}

PyObject* boundaries_to_py_list(const std::vector<std::vector<cv::Point>>& boundaries) {
    ScopedPyRef py_list(PyList_New(boundaries.size()));
    if (!py_list) {
        return nullptr;
    }

    for (size_t i = 0; i < boundaries.size(); ++i) {
        npy_intp dims[3] = {static_cast<npy_intp>(boundaries[i].size()), 1, 2};
        PyObject* arr = PyArray_SimpleNew(3, dims, NPY_INT32);
        if (!arr) {
            return nullptr;
        }

        int32_t* data = static_cast<int32_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(arr)));
        if (!boundaries[i].empty()) {
            std::memcpy(data, boundaries[i].data(), boundaries[i].size() * sizeof(cv::Point));
        }

        PyList_SET_ITEM(py_list.get(), i, arr);
    }
    return py_list.release();
}

} // namespace

static PyObject* relax_HCE_inner(PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {"gt", "rs", "gt_ske", "relax", "epsilon", nullptr};
    PyObject* py_gt = nullptr;
    PyObject* py_rs = nullptr;
    PyObject* py_ske = nullptr;
    int relax = 5;
    double epsilon = 2.0;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOO|id:relax_HCE", const_cast<char**>(kwlist),
            &py_gt, &py_rs, &py_ske, &relax, &epsilon)) {
        return nullptr;
    }

    if (relax < 0) {
        PyErr_SetString(PyExc_ValueError, "relax must be a non-negative integer");
        return nullptr;
    }
    if (!std::isfinite(epsilon) || epsilon < 0.0) {
        PyErr_SetString(PyExc_ValueError, "epsilon must be a finite, non-negative number");
        return nullptr;
    }

    cv::Mat gt_bin;
    if (!extract_binarized_2d(py_gt, "gt", false, true, gt_bin)) {
        return nullptr;
    }

    cv::Mat rs_bin;
    if (!extract_binarized_2d(py_rs, "rs", false, true, rs_bin)) {
        return nullptr;
    }

    if (gt_bin.rows != rs_bin.rows || gt_bin.cols != rs_bin.cols) {
        PyErr_Format(PyExc_ValueError, "Shape mismatch between gt (%dx%d) and rs (%dx%d)",
                     gt_bin.rows, gt_bin.cols, rs_bin.rows, rs_bin.cols);
        return nullptr;
    }

    cv::Mat ske_bin;
    if (!extract_binarized_2d(py_ske, "gt_ske", true, false, ske_bin)) {
        return nullptr;
    }

    if (gt_bin.rows != ske_bin.rows || gt_bin.cols != ske_bin.cols) {
        PyErr_Format(PyExc_ValueError, "Shape mismatch between gt (%dx%d) and gt_ske (%dx%d)",
                     gt_bin.rows, gt_bin.cols, ske_bin.rows, ske_bin.cols);
        return nullptr;
    }

    HCEOutput out;
    char err_buf[256] = {0};
    int err_type = 0; // 0=none, 1=bad_alloc, 2=cv_error, 3=runtime_error

    Py_BEGIN_ALLOW_THREADS
    try {
        out = relax_HCE_impl(gt_bin, rs_bin, ske_bin, relax, epsilon);
    } catch (const cv::Exception& e) {
        err_type = 2;
        std::strncpy(err_buf, e.what(), sizeof(err_buf) - 1);
    } catch (const std::bad_alloc&) {
        err_type = 1;
    } catch (const std::exception& e) {
        err_type = 3;
        std::strncpy(err_buf, e.what(), sizeof(err_buf) - 1);
    } catch (...) {
        err_type = 3;
        std::strncpy(err_buf, "Unknown C++ exception occurred during relax_HCE", sizeof(err_buf) - 1);
    }
    Py_END_ALLOW_THREADS

    if (err_type == 1) {
        return PyErr_NoMemory();
    }
    if (err_type == 2) {
        PyErr_SetString(PyExc_ValueError, err_buf);
        return nullptr;
    }
    if (err_type == 3) {
        PyErr_SetString(PyExc_RuntimeError, err_buf);
        return nullptr;
    }

    return Py_BuildValue("idid", out.poly_FP_point_cnt, out.indep_cnt_FP,
                         out.poly_FN_point_cnt, out.indep_cnt_FN);
}

static PyObject* py_relax_HCE(PyObject* /*self*/, PyObject* args, PyObject* kwargs) noexcept {
    try {
        return relax_HCE_inner(args, kwargs);
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    } catch (const cv::Exception& e) {
        if (!PyErr_Occurred()) {
            const char* msg = e.what();
            PyErr_SetString(PyExc_ValueError, msg ? msg : "OpenCV exception in relax_HCE");
        }
        return nullptr;
    } catch (const std::length_error& e) {
        if (!PyErr_Occurred()) {
            const char* msg = e.what();
            PyErr_SetString(PyExc_ValueError, msg ? msg : "Length error in relax_HCE");
        }
        return nullptr;
    } catch (const std::out_of_range& e) {
        if (!PyErr_Occurred()) {
            const char* msg = e.what();
            PyErr_SetString(PyExc_ValueError, msg ? msg : "Out of range error in relax_HCE");
        }
        return nullptr;
    } catch (const std::invalid_argument& e) {
        if (!PyErr_Occurred()) {
            const char* msg = e.what();
            PyErr_SetString(PyExc_ValueError, msg ? msg : "Invalid argument in relax_HCE");
        }
        return nullptr;
    } catch (const std::exception& e) {
        if (!PyErr_Occurred()) {
            const char* msg = e.what();
            PyErr_SetString(PyExc_RuntimeError, msg ? msg : "C++ exception in relax_HCE");
        }
        return nullptr;
    } catch (...) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, "Unknown C++ exception occurred in relax_HCE");
        }
        return nullptr;
    }
}

static PyObject* filter_bdy_cond_inner(PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {"bdy_", "mask", "cond", nullptr};
    PyObject* py_bdy = nullptr;
    PyObject* py_mask = nullptr;
    PyObject* py_cond = nullptr;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOO:filter_bdy_cond", const_cast<char**>(kwlist),
            &py_bdy, &py_mask, &py_cond)) {
        return nullptr;
    }

    std::vector<std::vector<cv::Point>> bdy;
    if (!parse_contours_list(py_bdy, bdy, "bdy_")) {
        return nullptr;
    }

    cv::Mat mask;
    if (!extract_binarized_2d(py_mask, "mask", true, false, mask)) {
        return nullptr;
    }

    cv::Mat cond;
    if (!extract_binarized_2d(py_cond, "cond", true, false, cond)) {
        return nullptr;
    }

    if (mask.rows != cond.rows || mask.cols != cond.cols) {
        PyErr_Format(PyExc_ValueError, "Shape mismatch between mask (%dx%d) and cond (%dx%d)",
                     mask.rows, mask.cols, cond.rows, cond.cols);
        return nullptr;
    }

    std::vector<std::vector<cv::Point>> boundaries;
    double indep_cnt = 0.0;
    char err_buf[256] = {0};
    int err_type = 0;

    cv::Mat kernel = get_cross_kernel();

    Py_BEGIN_ALLOW_THREADS
    try {
        auto res = filter_bdy_cond_impl(bdy, mask, cond, kernel);
        boundaries = std::move(res.first);
        indep_cnt = res.second;
    } catch (const cv::Exception& e) {
        err_type = 2;
        std::strncpy(err_buf, e.what(), sizeof(err_buf) - 1);
    } catch (const std::bad_alloc&) {
        err_type = 1;
    } catch (const std::exception& e) {
        err_type = 3;
        std::strncpy(err_buf, e.what(), sizeof(err_buf) - 1);
    } catch (...) {
        err_type = 3;
        std::strncpy(err_buf, "Unknown C++ exception occurred during filter_bdy_cond", sizeof(err_buf) - 1);
    }
    Py_END_ALLOW_THREADS

    if (err_type == 1) {
        return PyErr_NoMemory();
    }
    if (err_type == 2) {
        PyErr_SetString(PyExc_ValueError, err_buf);
        return nullptr;
    }
    if (err_type == 3) {
        PyErr_SetString(PyExc_RuntimeError, err_buf);
        return nullptr;
    }

    ScopedPyRef py_bdies(boundaries_to_py_list(boundaries));
    if (!py_bdies) {
        return nullptr;
    }

    ScopedPyRef py_indep(PyFloat_FromDouble(indep_cnt));
    if (!py_indep) {
        return nullptr;
    }

    return PyTuple_Pack(2, py_bdies.get(), py_indep.get());
}

static PyObject* py_filter_bdy_cond(PyObject* /*self*/, PyObject* args, PyObject* kwargs) noexcept {
    try {
        return filter_bdy_cond_inner(args, kwargs);
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    } catch (const cv::Exception& e) {
        if (!PyErr_Occurred()) {
            const char* msg = e.what();
            PyErr_SetString(PyExc_ValueError, msg ? msg : "OpenCV exception in filter_bdy_cond");
        }
        return nullptr;
    } catch (const std::length_error& e) {
        if (!PyErr_Occurred()) {
            const char* msg = e.what();
            PyErr_SetString(PyExc_ValueError, msg ? msg : "Length error in filter_bdy_cond");
        }
        return nullptr;
    } catch (const std::out_of_range& e) {
        if (!PyErr_Occurred()) {
            const char* msg = e.what();
            PyErr_SetString(PyExc_ValueError, msg ? msg : "Out of range error in filter_bdy_cond");
        }
        return nullptr;
    } catch (const std::invalid_argument& e) {
        if (!PyErr_Occurred()) {
            const char* msg = e.what();
            PyErr_SetString(PyExc_ValueError, msg ? msg : "Invalid argument in filter_bdy_cond");
        }
        return nullptr;
    } catch (const std::exception& e) {
        if (!PyErr_Occurred()) {
            const char* msg = e.what();
            PyErr_SetString(PyExc_RuntimeError, msg ? msg : "C++ exception in filter_bdy_cond");
        }
        return nullptr;
    } catch (...) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, "Unknown C++ exception occurred in filter_bdy_cond");
        }
        return nullptr;
    }
}

static PyObject* approximate_RDP_inner(PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {"boundaries", "epsilon", nullptr};
    PyObject* py_boundaries = nullptr;
    double epsilon = 1.0;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "O|d:approximate_RDP", const_cast<char**>(kwlist),
            &py_boundaries, &epsilon)) {
        return nullptr;
    }

    if (!std::isfinite(epsilon) || epsilon < 0.0) {
        PyErr_SetString(PyExc_ValueError, "epsilon must be a finite, non-negative number");
        return nullptr;
    }

    std::vector<std::vector<cv::Point>> boundaries;
    if (!parse_contours_list(py_boundaries, boundaries, "boundaries")) {
        return nullptr;
    }

    std::vector<std::vector<cv::Point>> approx_list;
    approx_list.resize(boundaries.size());
    int total_pixel_cnt = 0;
    char err_buf[256] = {0};
    int err_type = 0;

    Py_BEGIN_ALLOW_THREADS
    try {
        for (size_t i = 0; i < boundaries.size(); ++i) {
            if (!boundaries[i].empty()) {
                cv::approxPolyDP(boundaries[i], approx_list[i], epsilon, false);
            }
            total_pixel_cnt += static_cast<int>(approx_list[i].size());
        }
    } catch (const cv::Exception& e) {
        err_type = 2;
        std::strncpy(err_buf, e.what(), sizeof(err_buf) - 1);
    } catch (const std::bad_alloc&) {
        err_type = 1;
    } catch (const std::exception& e) {
        err_type = 3;
        std::strncpy(err_buf, e.what(), sizeof(err_buf) - 1);
    } catch (...) {
        err_type = 3;
        std::strncpy(err_buf, "Unknown C++ exception occurred during approximate_RDP", sizeof(err_buf) - 1);
    }
    Py_END_ALLOW_THREADS

    if (err_type == 1) {
        return PyErr_NoMemory();
    }
    if (err_type == 2) {
        PyErr_SetString(PyExc_ValueError, err_buf);
        return nullptr;
    }
    if (err_type == 3) {
        PyErr_SetString(PyExc_RuntimeError, err_buf);
        return nullptr;
    }

    ScopedPyRef py_boundaries_out(PyList_New(boundaries.size()));
    ScopedPyRef py_lengths_out(PyList_New(boundaries.size()));
    if (!py_boundaries_out || !py_lengths_out) {
        return nullptr;
    }

    for (size_t i = 0; i < approx_list.size(); ++i) {
        int len = static_cast<int>(approx_list[i].size());
        npy_intp dims[3] = {static_cast<npy_intp>(len), 1, 2};
        PyObject* arr = PyArray_SimpleNew(3, dims, NPY_INT32);
        if (!arr) {
            return nullptr;
        }

        int32_t* data = static_cast<int32_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(arr)));
        if (len > 0) {
            std::memcpy(data, approx_list[i].data(), static_cast<size_t>(len) * sizeof(cv::Point));
        }

        PyList_SET_ITEM(py_boundaries_out.get(), i, arr);

        PyObject* py_len = PyLong_FromLong(len);
        if (!py_len) {
            return nullptr;
        }
        PyList_SET_ITEM(py_lengths_out.get(), i, py_len);
    }

    ScopedPyRef py_pixel_cnt(PyLong_FromLong(total_pixel_cnt));
    if (!py_pixel_cnt) {
        return nullptr;
    }

    return PyTuple_Pack(
        3, py_boundaries_out.get(), py_lengths_out.get(), py_pixel_cnt.get()
    );
}

static PyObject* py_approximate_RDP(PyObject* /*self*/, PyObject* args, PyObject* kwargs) noexcept {
    try {
        return approximate_RDP_inner(args, kwargs);
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    } catch (const cv::Exception& e) {
        if (!PyErr_Occurred()) {
            const char* msg = e.what();
            PyErr_SetString(PyExc_ValueError, msg ? msg : "OpenCV exception in approximate_RDP");
        }
        return nullptr;
    } catch (const std::length_error& e) {
        if (!PyErr_Occurred()) {
            const char* msg = e.what();
            PyErr_SetString(PyExc_ValueError, msg ? msg : "Length error in approximate_RDP");
        }
        return nullptr;
    } catch (const std::out_of_range& e) {
        if (!PyErr_Occurred()) {
            const char* msg = e.what();
            PyErr_SetString(PyExc_ValueError, msg ? msg : "Out of range error in approximate_RDP");
        }
        return nullptr;
    } catch (const std::invalid_argument& e) {
        if (!PyErr_Occurred()) {
            const char* msg = e.what();
            PyErr_SetString(PyExc_ValueError, msg ? msg : "Invalid argument in approximate_RDP");
        }
        return nullptr;
    } catch (const std::exception& e) {
        if (!PyErr_Occurred()) {
            const char* msg = e.what();
            PyErr_SetString(PyExc_RuntimeError, msg ? msg : "C++ exception in approximate_RDP");
        }
        return nullptr;
    } catch (...) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, "Unknown C++ exception occurred in approximate_RDP");
        }
        return nullptr;
    }
}

static PyMethodDef TurboHCEMethods[] = {
    {"relax_HCE", reinterpret_cast<PyCFunction>(py_relax_HCE), METH_VARARGS | METH_KEYWORDS,
     "Compute relaxed Human Correction Effort (HCE) components."},
    {"relax_hce", reinterpret_cast<PyCFunction>(py_relax_HCE), METH_VARARGS | METH_KEYWORDS,
     "Compute relaxed Human Correction Effort (HCE) components."},
    {"filter_bdy_cond", reinterpret_cast<PyCFunction>(py_filter_bdy_cond),
     METH_VARARGS | METH_KEYWORDS,
     "Filter boundary conditions based on connected components and condition map."},
    {"approximate_RDP", reinterpret_cast<PyCFunction>(py_approximate_RDP),
     METH_VARARGS | METH_KEYWORDS,
     "Approximate boundaries using the Ramer-Douglas-Peucker algorithm."},
    {"approximate_rdp", reinterpret_cast<PyCFunction>(py_approximate_RDP),
     METH_VARARGS | METH_KEYWORDS,
     "Approximate boundaries using the Ramer-Douglas-Peucker algorithm."},
    {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef turbo_hce_module = {
    PyModuleDef_HEAD_INIT,
    "_turbo_hce",
    "High-performance C++ implementation of HCE metric helpers.",
    -1,
    TurboHCEMethods
};

PyMODINIT_FUNC PyInit__turbo_hce(void) {
    import_array();
    if (PyErr_Occurred()) {
        return nullptr;
    }
    return PyModule_Create(&turbo_hce_module);
}
