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

inline int find_highest_bit(const unsigned int mask) noexcept {
#if defined(_MSC_VER)
    unsigned long index = 0;
    _BitScanReverse(&index, mask);
    return static_cast<int>(index);
#else
    return 31 - __builtin_clz(mask);
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

alignas(32) constexpr std::uint8_t BITMAP_PASS0[32] = {
    0xc8, 0xdc, 0x00, 0xd0, 0x00, 0x01, 0x00, 0x10,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x11,
    0xcc, 0x80, 0x00, 0x80, 0x03, 0x00, 0x00, 0x00,
    0xce, 0x80, 0x00, 0x00, 0x8a, 0x00, 0x0b, 0x00
};

alignas(32) constexpr std::uint8_t BITMAP_PASS1[32] = {
    0x80, 0x84, 0x00, 0xd5, 0x00, 0x01, 0x00, 0xd1,
    0x00, 0x00, 0x00, 0x00, 0x01, 0x11, 0x01, 0x51,
    0x84, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00,
    0x8b, 0x00, 0x00, 0x00, 0x03, 0x00, 0x03, 0x13
};

}  // namespace

void skeletonize_zhang_suen(const std::uint8_t* input,
                            const std::size_t rows,
                            const std::size_t columns,
                            std::uint8_t* output) {
    const std::size_t pixel_count = checked_pixel_count(rows, columns);
    if (pixel_count == 0) {
        return;
    }
    if (input == nullptr) {
        throw std::invalid_argument("non-empty skeleton image requires an input buffer");
    }
    if (output == nullptr) {
        throw std::invalid_argument("non-empty skeleton image requires an output buffer");
    }

    if (rows > std::numeric_limits<std::size_t>::max() - 2 ||
        columns > std::numeric_limits<std::size_t>::max() - 2) {
        throw std::length_error("skeleton image dimensions overflow size_t");
    }

    // 1. Initial scan: discover foreground bounding box
    std::size_t min_row = rows;
    std::size_t max_row = 0;
    std::size_t min_col = columns;
    std::size_t max_col = 0;

#if defined(__AVX2__)
    const __m256i zero256 = _mm256_setzero_si256();
    for (std::size_t r = 0; r < rows; ++r) {
        const std::uint8_t* in_row = input + r * columns;
        std::size_t c = 0;
        for (; c + 32 <= columns; c += 32) {
            const __m256i v = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(in_row + c));
            if (_mm256_testz_si256(v, v)) {
                continue;
            }
            const uint32_t is_zero_mask = static_cast<uint32_t>(
                _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, zero256)));
            uint32_t nonzero_mask = ~is_zero_mask;
            if (r < min_row) min_row = r;
            max_row = r;
            const std::size_t first_c = c + static_cast<std::size_t>(count_trailing_zeros(nonzero_mask));
            if (first_c < min_col) min_col = first_c;
            const std::size_t last_c = c + static_cast<std::size_t>(find_highest_bit(nonzero_mask));
            if (last_c > max_col) max_col = last_c;
        }
        for (; c < columns; ++c) {
            if (in_row[c] != 0) {
                if (r < min_row) min_row = r;
                max_row = r;
                if (c < min_col) min_col = c;
                if (c > max_col) max_col = c;
            }
        }
    }
#else
    for (std::size_t r = 0; r < rows; ++r) {
        const std::uint8_t* in_row = input + r * columns;
        for (std::size_t c = 0; c < columns; ++c) {
            if (in_row[c] != 0) {
                if (r < min_row) min_row = r;
                max_row = r;
                if (c < min_col) min_col = c;
                if (c > max_col) max_col = c;
            }
        }
    }
#endif

    // Completely blank input: return all zeros immediately
    if (min_row > max_row || min_col > max_col) {
        std::memset(output, 0, pixel_count);
        return;
    }

    // Zero-initialize the entire output buffer
    std::memset(output, 0, pixel_count);

    // 2. Crop working image to ROI plus 1-pixel zero-padded halo
    const std::size_t roi_h = max_row - min_row + 1;
    const std::size_t roi_w = max_col - min_col + 1;

    const std::size_t padded_cols = roi_w + 2;
    const std::size_t padded_rows = roi_h + 2;
    if (padded_rows > std::numeric_limits<std::size_t>::max() / padded_cols) {
        throw std::length_error("skeleton image dimensions overflow size_t");
    }

    const std::size_t padded_size = padded_rows * padded_cols;
    std::vector<std::uint8_t> pad(padded_size, 0);
    std::vector<std::uint32_t> row_counts(padded_rows, 0);
    std::vector<std::uint32_t> col_counts(padded_cols, 0);

#if defined(__AVX2__)
    const __m256i one256 = _mm256_set1_epi8(1);
    for (std::size_t r = 0; r < roi_h; ++r) {
        const std::uint8_t* in_row = input + (min_row + r) * columns + min_col;
        std::uint8_t* pad_row = pad.data() + (r + 1) * padded_cols + 1;
        const std::size_t pad_r = r + 1;
        std::size_t c = 0;
        for (; c + 32 <= roi_w; c += 32) {
            const __m256i v = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(in_row + c));
            if (_mm256_testz_si256(v, v)) {
                continue;
            }
            const __m256i is_zero = _mm256_cmpeq_epi8(v, zero256);
            const __m256i is_nz = _mm256_andnot_si256(is_zero, one256);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(pad_row + c), is_nz);
            uint32_t nonzero_mask = ~static_cast<uint32_t>(_mm256_movemask_epi8(is_zero));
            row_counts[pad_r] += static_cast<uint32_t>(_mm_popcnt_u32(nonzero_mask));
            while (nonzero_mask != 0) {
                const int bit = count_trailing_zeros(nonzero_mask);
                col_counts[c + bit + 1]++;
                nonzero_mask &= nonzero_mask - 1;
            }
        }
        for (; c < roi_w; ++c) {
            if (in_row[c] != 0) {
                pad_row[c] = 1;
                row_counts[pad_r]++;
                col_counts[c + 1]++;
            }
        }
    }
#else
    for (std::size_t r = 0; r < roi_h; ++r) {
        const std::uint8_t* in_row = input + (min_row + r) * columns + min_col;
        std::uint8_t* pad_row = pad.data() + (r + 1) * padded_cols + 1;
        const std::size_t pad_r = r + 1;
        for (std::size_t c = 0; c < roi_w; ++c) {
            if (in_row[c] != 0) {
                pad_row[c] = 1;
                const std::size_t pad_c = c + 1;
                row_counts[pad_r]++;
                col_counts[pad_c]++;
            }
        }
    }
#endif

    std::size_t cur_min_r = 1;
    std::size_t cur_max_r = roi_h;
    std::size_t cur_min_c = 1;
    std::size_t cur_max_c = roi_w;

#if defined(__AVX2__)
    const __m256i pow2_lut = _mm256_setr_epi8(
        1, 2, 4, 8, 16, 32, 64, static_cast<char>(128),
        0, 0, 0, 0, 0, 0, 0, 0,
        1, 2, 4, 8, 16, 32, 64, static_cast<char>(128),
        0, 0, 0, 0, 0, 0, 0, 0
    );

    const __m256i t_low_0 = _mm256_broadcastsi128_si256(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(BITMAP_PASS0)));
    const __m256i t_high_0 = _mm256_broadcastsi128_si256(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(BITMAP_PASS0 + 16)));

    const __m256i t_low_1 = _mm256_broadcastsi128_si256(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(BITMAP_PASS1)));
    const __m256i t_high_1 = _mm256_broadcastsi128_si256(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(BITMAP_PASS1 + 16)));
#endif

    struct Deletion {
        std::size_t idx;
        std::size_t r;
        std::size_t c;
    };
    std::vector<Deletion> to_delete;
    to_delete.reserve(std::min<std::size_t>(padded_size / 4, 131072));

    bool pixel_removed = true;
    while (pixel_removed) {
        pixel_removed = false;

        // O(1) Contraction using row and column foreground counts
        while (cur_min_r <= cur_max_r && row_counts[cur_min_r] == 0) ++cur_min_r;
        while (cur_max_r >= cur_min_r && row_counts[cur_max_r] == 0) --cur_max_r;
        while (cur_min_c <= cur_max_c && col_counts[cur_min_c] == 0) ++cur_min_c;
        while (cur_max_c >= cur_min_c && col_counts[cur_max_c] == 0) --cur_max_c;

        if (cur_min_r > cur_max_r || cur_min_c > cur_max_c) break;

        const std::size_t c_start = cur_min_c;
        const std::size_t c_end = cur_max_c;

        for (std::uint8_t pass = 0; pass < 2; ++pass) {
            const std::uint8_t pass_mask = (pass == 0) ? 1 : 2;

#if defined(__AVX2__)
            const __m256i t_low = (pass == 0) ? t_low_0 : t_low_1;
            const __m256i t_high = (pass == 0) ? t_high_0 : t_high_1;

            for (std::size_t r = cur_min_r; r <= cur_max_r; ++r) {
                if (row_counts[r] == 0) continue;

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

                    if (_mm256_testz_si256(candidate_vec, candidate_vec)) {
                        continue;
                    }

                    // Compute neighborhood codes via SIMD shifts and ORs
                    __m256i code_vec = n0;
                    code_vec = _mm256_or_si256(code_vec, _mm256_slli_epi16(n1, 1));
                    code_vec = _mm256_or_si256(code_vec, _mm256_slli_epi16(n2, 2));
                    code_vec = _mm256_or_si256(code_vec, _mm256_slli_epi16(n3, 3));
                    code_vec = _mm256_or_si256(code_vec, _mm256_slli_epi16(n4, 4));
                    code_vec = _mm256_or_si256(code_vec, _mm256_slli_epi16(n5, 5));
                    code_vec = _mm256_or_si256(code_vec, _mm256_slli_epi16(n6, 6));
                    code_vec = _mm256_or_si256(code_vec, _mm256_slli_epi16(n7, 7));

                    // AVX2 256-bit bitset predicate: evaluates exact deletion eligibility in SIMD
                    const __m256i byte_idx = _mm256_and_si256(
                        _mm256_srli_epi16(code_vec, 3), _mm256_set1_epi8(0x1F));
                    const __m256i bit_idx = _mm256_and_si256(code_vec, _mm256_set1_epi8(0x07));

                    const __m256i val_low = _mm256_shuffle_epi8(t_low, byte_idx);
                    const __m256i idx_high = _mm256_sub_epi8(byte_idx, _mm256_set1_epi8(16));
                    const __m256i val_high = _mm256_shuffle_epi8(t_high, idx_high);
                    const __m256i is_high = _mm256_cmpgt_epi8(byte_idx, _mm256_set1_epi8(15));
                    const __m256i val_byte = _mm256_blendv_epi8(val_low, val_high, is_high);

                    const __m256i bit_mask = _mm256_shuffle_epi8(pow2_lut, bit_idx);
                    const __m256i bit_test = _mm256_and_si256(val_byte, bit_mask);
                    const __m256i is_deletable = _mm256_cmpeq_epi8(bit_test, bit_mask);

                    uint32_t delete_mask = _mm256_movemask_epi8(
                        _mm256_and_si256(is_deletable, candidate_vec));

                    while (delete_mask != 0) {
                        const int bit = count_trailing_zeros(delete_mask);
                        to_delete.push_back({r_offset + c + bit, r, c + bit});
                        delete_mask &= delete_mask - 1;
                    }
                }

                // Remainder columns
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
                        to_delete.push_back({idx, r, c});
                    }
                }
            }
#else
            for (std::size_t r = cur_min_r; r <= cur_max_r; ++r) {
                if (row_counts[r] == 0) continue;

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
                        to_delete.push_back({idx, r, c});
                    }
                }
            }
#endif

            if (!to_delete.empty()) {
                pixel_removed = true;
                for (const auto& del : to_delete) {
                    pad[del.idx] = 0;
                    row_counts[del.r]--;
                    col_counts[del.c]--;
                }
                to_delete.clear();
            }
        }
    }

    // Copy thinned ROI back into output buffer at original offset
    for (std::size_t r = 0; r < roi_h; ++r) {
        std::memcpy(output + (min_row + r) * columns + min_col,
                    pad.data() + (r + 1) * padded_cols + 1,
                    roi_w);
    }
}

std::vector<std::uint8_t> skeletonize_zhang_suen(const std::uint8_t* input,
                                                  const std::size_t rows,
                                                  const std::size_t columns) {
    const std::size_t pixel_count = checked_pixel_count(rows, columns);
    if (pixel_count == 0) {
        return {};
    }
    std::vector<std::uint8_t> result(pixel_count);
    skeletonize_zhang_suen(input, rows, columns, result.data());
    return result;
}

}  // namespace turbo_hce::morphology
