#include "skeletonize.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#endif

// 2D Zhang-Suen morphological skeletonization.
// Lookup table and 8-neighbor connectivity encoding derived from scikit-image
// (BSD-3-Clause license, Copyright (c) 2011-2024, the scikit-image team).

namespace turbo_hce::morphology {
namespace {

alignas(64) constexpr std::uint8_t ZHANG_SUEN_LUT[256] = {
    0, 0, 0, 1, 0, 0, 1, 3, 0, 0, 3, 1, 1, 0, 1, 3,
    0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 3, 0, 3, 3,
    0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 2, 2,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0,
    3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 2, 0,
    0, 0, 3, 1, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 3, 1, 3, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    3, 3, 0, 1, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0
};

inline int count_trailing_zeros(const unsigned int mask) noexcept {
#if defined(_MSC_VER)
    unsigned long index = 0;
    _BitScanForward(&index, mask);
    return static_cast<int>(index);
#else
    return __builtin_ctz(mask);
#endif
}

std::size_t checked_pixel_count(const std::size_t rows, const std::size_t columns) {
    if (rows == 0 || columns == 0) {
        return 0;
    }
    if (rows > std::numeric_limits<std::size_t>::max() / columns) {
        throw std::length_error("skeleton image dimensions overflow size_t");
    }
    return rows * columns;
}

}  // namespace

std::vector<std::uint8_t> skeletonize_zhang_suen(const std::uint8_t* input,
                                                  const std::size_t rows,
                                                  const std::size_t columns) {
    const std::size_t pixel_count = checked_pixel_count(rows, columns);
    if (pixel_count == 0) {
        return {};
    }
    if (input == nullptr) {
        throw std::invalid_argument("non-empty skeleton image requires an input buffer");
    }

    if (rows > std::numeric_limits<std::size_t>::max() - 2 ||
        columns > std::numeric_limits<std::size_t>::max() - 2) {
        throw std::length_error("skeleton image dimensions overflow size_t");
    }

    const std::size_t padded_rows = rows + 2;
    const std::size_t padded_cols = columns + 2;
    if (padded_rows > std::numeric_limits<std::size_t>::max() / padded_cols) {
        throw std::length_error("skeleton image dimensions overflow size_t");
    }

    const std::size_t padded_size = padded_rows * padded_cols;
    std::vector<std::uint8_t> pad(padded_size, 0);

    std::size_t min_row = rows + 1;
    std::size_t max_row = 0;
    std::size_t min_col = columns + 1;
    std::size_t max_col = 0;

    for (std::size_t r = 0; r < rows; ++r) {
        const std::uint8_t* in_row = input + r * columns;
        std::uint8_t* pad_row = pad.data() + (r + 1) * padded_cols + 1;
        bool has_fg = false;
        for (std::size_t c = 0; c < columns; ++c) {
            if (in_row[c] != 0) {
                pad_row[c] = 1;
                has_fg = true;
                if (c + 1 < min_col) min_col = c + 1;
                if (c + 1 > max_col) max_col = c + 1;
            }
        }
        if (has_fg) {
            if (min_row > rows) min_row = r + 1;
            max_row = r + 1;
        }
    }

    if (min_row > max_row) {
        return std::vector<std::uint8_t>(pixel_count, 0);
    }

    std::vector<std::size_t> to_delete;
    to_delete.reserve(std::min<std::size_t>(pixel_count / 4, 131072));

    bool pixel_removed = true;
    while (pixel_removed) {
        pixel_removed = false;

#if defined(__AVX2__)
        // Contract active row bounds
        while (min_row <= max_row) {
            const std::uint8_t* row_ptr = pad.data() + min_row * padded_cols + 1;
            std::size_t c = 0;
            bool has_fg = false;
            for (; c + 32 <= columns; c += 32) {
                const __m256i v = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(row_ptr + c));
                if (!_mm256_testz_si256(v, v)) {
                    has_fg = true;
                    break;
                }
            }
            if (!has_fg) {
                for (; c < columns; ++c) {
                    if (row_ptr[c] != 0) {
                        has_fg = true;
                        break;
                    }
                }
            }
            if (has_fg) break;
            ++min_row;
        }

        while (max_row >= min_row) {
            const std::uint8_t* row_ptr = pad.data() + max_row * padded_cols + 1;
            std::size_t c = 0;
            bool has_fg = false;
            for (; c + 32 <= columns; c += 32) {
                const __m256i v = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(row_ptr + c));
                if (!_mm256_testz_si256(v, v)) {
                    has_fg = true;
                    break;
                }
            }
            if (!has_fg) {
                for (; c < columns; ++c) {
                    if (row_ptr[c] != 0) {
                        has_fg = true;
                        break;
                    }
                }
            }
            if (has_fg) break;
            if (max_row == 0) break;
            --max_row;
        }

        if (min_row > max_row) break;

        const std::size_t c_start = (min_col > 32) ? (((min_col - 1) / 32) * 32 + 1) : 1;
        const std::size_t c_end = std::min<std::size_t>(columns, max_col);

        for (std::uint8_t pass = 0; pass < 2; ++pass) {
            const std::uint8_t pass_mask = (pass == 0) ? 1 : 2;

            for (std::size_t r = min_row; r <= max_row; ++r) {
                const std::size_t r_offset = r * padded_cols;
                const std::size_t prev_row = r_offset - padded_cols;
                const std::size_t next_row = r_offset + padded_cols;

                const std::uint8_t* curr_ptr = pad.data() + r_offset;
                const std::uint8_t* prev_ptr = pad.data() + prev_row;
                const std::uint8_t* next_ptr = pad.data() + next_row;

                std::size_t c = c_start;
                for (; c + 32 <= c_end + 1; c += 32) {
                    const __m256i v_center = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(curr_ptr + c));

                    if (_mm256_testz_si256(v_center, v_center)) {
                        continue;
                    }

                    const __m256i n0 = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(prev_ptr + c - 1));
                    const __m256i n1 = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(prev_ptr + c));
                    const __m256i n2 = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(prev_ptr + c + 1));
                    const __m256i n3 = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(curr_ptr + c + 1));
                    const __m256i n4 = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(next_ptr + c + 1));
                    const __m256i n5 = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(next_ptr + c));
                    const __m256i n6 = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(next_ptr + c - 1));
                    const __m256i n7 = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(curr_ptr + c - 1));

                    __m256i interior = _mm256_and_si256(n0, n1);
                    interior = _mm256_and_si256(interior, n2);
                    interior = _mm256_and_si256(interior, n3);
                    interior = _mm256_and_si256(interior, n4);
                    interior = _mm256_and_si256(interior, n5);
                    interior = _mm256_and_si256(interior, n6);
                    interior = _mm256_and_si256(interior, n7);

                    const __m256i mask_center = _mm256_cmpgt_epi8(
                        v_center, _mm256_setzero_si256());
                    const __m256i mask_interior = _mm256_cmpgt_epi8(
                        interior, _mm256_setzero_si256());
                    const __m256i candidate_vec = _mm256_andnot_si256(
                        mask_interior, mask_center);

                    std::uint32_t candidate_mask = _mm256_movemask_epi8(candidate_vec);
                    if (candidate_mask == 0) {
                        continue;
                    }

                    __m256i code_vec = n0;
                    code_vec = _mm256_or_si256(code_vec, _mm256_slli_epi16(n1, 1));
                    code_vec = _mm256_or_si256(code_vec, _mm256_slli_epi16(n2, 2));
                    code_vec = _mm256_or_si256(code_vec, _mm256_slli_epi16(n3, 3));
                    code_vec = _mm256_or_si256(code_vec, _mm256_slli_epi16(n4, 4));
                    code_vec = _mm256_or_si256(code_vec, _mm256_slli_epi16(n5, 5));
                    code_vec = _mm256_or_si256(code_vec, _mm256_slli_epi16(n6, 6));
                    code_vec = _mm256_or_si256(code_vec, _mm256_slli_epi16(n7, 7));

                    alignas(32) std::uint8_t codes[32];
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(codes), code_vec);

                    while (candidate_mask != 0) {
                        const int bit = count_trailing_zeros(candidate_mask);
                        const std::uint8_t code = codes[bit];
                        if (ZHANG_SUEN_LUT[code] & pass_mask) {
                            to_delete.push_back(r_offset + c + bit);
                        }
                        candidate_mask &= candidate_mask - 1;
                    }
                }

                for (; c <= c_end; ++c) {
                    const std::size_t idx = r_offset + c;
                    if (pad[idx] == 0) continue;

                    const std::uint8_t code = static_cast<std::uint8_t>(
                        (pad[prev_row + c - 1]) |
                        (pad[prev_row + c    ] << 1) |
                        (pad[prev_row + c + 1] << 2) |
                        (pad[r_offset + c + 1] << 3) |
                        (pad[next_row + c + 1] << 4) |
                        (pad[next_row + c    ] << 5) |
                        (pad[next_row + c - 1] << 6) |
                        (pad[r_offset + c - 1] << 7)
                    );

                    if (ZHANG_SUEN_LUT[code] & pass_mask) {
                        to_delete.push_back(idx);
                    }
                }
            }

            if (!to_delete.empty()) {
                pixel_removed = true;
                for (const auto del_idx : to_delete) {
                    pad[del_idx] = 0;
                }
                to_delete.clear();
            }
        }
#else
        const std::size_t c_start = (min_col >= 1) ? min_col : 1;
        const std::size_t c_end = std::min<std::size_t>(columns, max_col);

        for (std::uint8_t pass = 0; pass < 2; ++pass) {
            const std::uint8_t pass_mask = (pass == 0) ? 1 : 2;

            for (std::size_t r = min_row; r <= max_row; ++r) {
                const std::size_t r_offset = r * padded_cols;
                const std::size_t prev_row = r_offset - padded_cols;
                const std::size_t next_row = r_offset + padded_cols;

                for (std::size_t c = c_start; c <= c_end; ++c) {
                    const std::size_t idx = r_offset + c;
                    if (pad[idx] == 0) continue;

                    const std::uint8_t code = static_cast<std::uint8_t>(
                        (pad[prev_row + c - 1]) |
                        (pad[prev_row + c    ] << 1) |
                        (pad[prev_row + c + 1] << 2) |
                        (pad[r_offset + c + 1] << 3) |
                        (pad[next_row + c + 1] << 4) |
                        (pad[next_row + c    ] << 5) |
                        (pad[next_row + c - 1] << 6) |
                        (pad[r_offset + c - 1] << 7)
                    );

                    if (ZHANG_SUEN_LUT[code] & pass_mask) {
                        to_delete.push_back(idx);
                    }
                }
            }

            if (!to_delete.empty()) {
                pixel_removed = true;
                for (const auto del_idx : to_delete) {
                    pad[del_idx] = 0;
                }
                to_delete.clear();
            }
        }
#endif
    }

    std::vector<std::uint8_t> result(pixel_count);
    for (std::size_t r = 0; r < rows; ++r) {
        const std::uint8_t* pad_row = pad.data() + (r + 1) * padded_cols + 1;
        std::uint8_t* res_row = result.data() + r * columns;
        std::memcpy(res_row, pad_row, columns);
    }

    return result;
}

}  // namespace turbo_hce::morphology
