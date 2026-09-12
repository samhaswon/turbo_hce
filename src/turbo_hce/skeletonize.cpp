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
            const std::size_t first_c =
                c + static_cast<std::size_t>(count_trailing_zeros(nonzero_mask));
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

    // Zero-initialize output buffer strictly outside the bounding box
    if (min_row > 0) {
        std::memset(output, 0, min_row * columns);
    }
    if (max_row + 1 < rows) {
        std::memset(output + (max_row + 1) * columns, 0, (rows - 1 - max_row) * columns);
    }
    for (std::size_t r = min_row; r <= max_row; ++r) {
        if (min_col > 0) {
            std::memset(output + r * columns, 0, min_col);
        }
        if (max_col + 1 < columns) {
            std::memset(output + r * columns + max_col + 1, 0, columns - 1 - max_col);
        }
    }

    // 2. Crop working image to ROI plus 1-pixel zero-padded halo
    const std::size_t roi_h = max_row - min_row + 1;
    const std::size_t roi_w = max_col - min_col + 1;

    const std::size_t num_chunks_x = (roi_w + 31) / 32;
    const std::size_t padded_cols = (num_chunks_x + 1) * 32;
    const std::size_t padded_rows = roi_h + 2;
    if (padded_rows > std::numeric_limits<std::uint32_t>::max() / padded_cols) {
        throw std::length_error("skeleton image dimensions exceed 32-bit index limit");
    }

    const std::size_t padded_size = padded_rows * padded_cols;
    std::vector<std::uint8_t> pad(padded_size, 0);

    constexpr std::size_t TILE_H = 16;
    const std::size_t num_tiles_y = (roi_h + TILE_H - 1) / TILE_H;
    const std::size_t num_tiles_x = num_chunks_x;
    const std::size_t total_tiles = num_tiles_y * num_tiles_x;

    std::vector<std::uint32_t> tile_counts(total_tiles, 0);
    std::vector<std::uint8_t> dirty_pass0(total_tiles, 0);
    std::vector<std::uint8_t> dirty_pass1(total_tiles, 0);

#if defined(__AVX2__)
    const __m256i one256 = _mm256_set1_epi8(1);
    for (std::size_t r = 0; r < roi_h; ++r) {
        const std::uint8_t* in_row = input + (min_row + r) * columns + min_col;
        std::uint8_t* pad_row = pad.data() + (r + 1) * padded_cols + 1;
        const std::size_t ty = r / TILE_H;
        for (std::size_t tx = 0; tx < num_tiles_x; ++tx) {
            const std::size_t c = tx * 32;
            if (c >= roi_w) break;
            const std::size_t rem = std::min<std::size_t>(32, roi_w - c);
            if (rem == 32) {
                const __m256i v = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(in_row + c));
                if (_mm256_testz_si256(v, v)) continue;
                const __m256i is_zero = _mm256_cmpeq_epi8(v, zero256);
                const __m256i is_nz = _mm256_andnot_si256(is_zero, one256);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(pad_row + c), is_nz);
                const uint32_t nz_mask = ~static_cast<uint32_t>(_mm256_movemask_epi8(is_zero));
                tile_counts[ty * num_tiles_x + tx] +=
                    static_cast<std::uint32_t>(_mm_popcnt_u32(nz_mask));
            } else {
                for (std::size_t i = 0; i < rem; ++i) {
                    if (in_row[c + i] != 0) {
                        pad_row[c + i] = 1;
                        tile_counts[ty * num_tiles_x + tx]++;
                    }
                }
            }
        }
    }
#else
    for (std::size_t r = 0; r < roi_h; ++r) {
        const std::uint8_t* in_row = input + (min_row + r) * columns + min_col;
        std::uint8_t* pad_row = pad.data() + (r + 1) * padded_cols + 1;
        const std::size_t ty = r / TILE_H;
        for (std::size_t tx = 0; tx < num_tiles_x; ++tx) {
            const std::size_t c = tx * 32;
            if (c >= roi_w) break;
            const std::size_t rem = std::min<std::size_t>(32, roi_w - c);
            for (std::size_t i = 0; i < rem; ++i) {
                if (in_row[c + i] != 0) {
                    pad_row[c + i] = 1;
                    tile_counts[ty * num_tiles_x + tx]++;
                }
            }
        }
    }
#endif

    for (std::size_t t = 0; t < total_tiles; ++t) {
        if (tile_counts[t] > 0) {
            dirty_pass0[t] = 1;
            dirty_pass1[t] = 1;
        }
    }

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

    struct DeletionRecord {
        std::uint32_t idx;
        std::uint16_t ty;
        std::uint16_t tx;
        std::uint8_t row_in_tile;
        std::uint8_t col_in_tile;
    };
    std::vector<DeletionRecord> to_delete;
    to_delete.reserve(std::min<std::size_t>(padded_size / 4, 131072));

    bool pixel_removed = true;
    while (pixel_removed) {
        pixel_removed = false;

        for (std::uint8_t pass = 0; pass < 2; ++pass) {
            std::uint8_t* curr_dirty = (pass == 0) ? dirty_pass0.data() : dirty_pass1.data();
            const std::uint8_t pass_mask = (pass == 0) ? 1 : 2;

#if defined(__AVX2__)
            const __m256i t_low = (pass == 0) ? t_low_0 : t_low_1;
            const __m256i t_high = (pass == 0) ? t_high_0 : t_high_1;

            for (std::size_t ty = 0; ty < num_tiles_y; ++ty) {
                const std::size_t r_start = ty * TILE_H + 1;
                const std::size_t r_end = std::min(r_start + TILE_H - 1, roi_h);

                for (std::size_t tx = 0; tx < num_tiles_x; ++tx) {
                    const std::size_t t = ty * num_tiles_x + tx;
                    if (!curr_dirty[t]) continue;
                    curr_dirty[t] = 0;

                    if (tile_counts[t] == 0) continue;

                    const std::size_t c = tx * 32 + 1;

                    for (std::size_t r = r_start; r <= r_end; ++r) {
                        const std::size_t r_offset = r * padded_cols;
                        const std::uint8_t* curr_ptr = pad.data() + r_offset;
                        const std::uint8_t* prev_ptr = pad.data() + r_offset - padded_cols;
                        const std::uint8_t* next_ptr = pad.data() + r_offset + padded_cols;

                        const __m256i v_center = _mm256_loadu_si256(
                            reinterpret_cast<const __m256i*>(curr_ptr + c));
                        if (_mm256_testz_si256(v_center, v_center)) continue;

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
                            v_center, zero256);
                        const __m256i mask_interior = _mm256_cmpgt_epi8(
                            interior, zero256);
                        const __m256i candidate_vec = _mm256_andnot_si256(
                            mask_interior, mask_center);

                        if (_mm256_testz_si256(candidate_vec, candidate_vec)) continue;

                        __m256i code_vec = n0;
                        code_vec = _mm256_or_si256(code_vec, _mm256_slli_epi16(n1, 1));
                        code_vec = _mm256_or_si256(code_vec, _mm256_slli_epi16(n2, 2));
                        code_vec = _mm256_or_si256(code_vec, _mm256_slli_epi16(n3, 3));
                        code_vec = _mm256_or_si256(code_vec, _mm256_slli_epi16(n4, 4));
                        code_vec = _mm256_or_si256(code_vec, _mm256_slli_epi16(n5, 5));
                        code_vec = _mm256_or_si256(code_vec, _mm256_slli_epi16(n6, 6));
                        code_vec = _mm256_or_si256(code_vec, _mm256_slli_epi16(n7, 7));

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
                            const std::size_t del_c = c + static_cast<std::size_t>(bit);
                            to_delete.push_back({
                                static_cast<std::uint32_t>(r_offset + del_c),
                                static_cast<std::uint16_t>(ty),
                                static_cast<std::uint16_t>(tx),
                                static_cast<std::uint8_t>(r - r_start),
                                static_cast<std::uint8_t>(bit)
                            });
                            tile_counts[t]--;
                            delete_mask &= delete_mask - 1;
                        }
                    }
                }
            }
#else
            for (std::size_t ty = 0; ty < num_tiles_y; ++ty) {
                const std::size_t r_start = ty * TILE_H + 1;
                const std::size_t r_end = std::min(r_start + TILE_H - 1, roi_h);

                for (std::size_t tx = 0; tx < num_tiles_x; ++tx) {
                    const std::size_t t = ty * num_tiles_x + tx;
                    if (!curr_dirty[t]) continue;
                    curr_dirty[t] = 0;

                    if (tile_counts[t] == 0) continue;

                    const std::size_t c_start = tx * 32 + 1;
                    const std::size_t c_end = std::min(c_start + 31, roi_w);

                    for (std::size_t r = r_start; r <= r_end; ++r) {
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
                                to_delete.push_back({
                                    static_cast<std::uint32_t>(idx),
                                    static_cast<std::uint16_t>(ty),
                                    static_cast<std::uint16_t>(tx),
                                    static_cast<std::uint8_t>(r - r_start),
                                    static_cast<std::uint8_t>(c - c_start)
                                });
                                tile_counts[t]--;
                            }
                        }
                    }
                }
            }
#endif

            if (!to_delete.empty()) {
                pixel_removed = true;
                for (const auto& del : to_delete) {
                    pad[del.idx] = 0;

                    const std::size_t ty = del.ty;
                    const std::size_t tx = del.tx;
                    const std::size_t t = ty * num_tiles_x + tx;

                    dirty_pass0[t] = 1;
                    dirty_pass1[t] = 1;

                    const bool top_edge = (del.row_in_tile == 0 && ty > 0);
                    const bool bottom_edge =
                        (del.row_in_tile == TILE_H - 1 && ty + 1 < num_tiles_y);
                    const bool left_edge = (del.col_in_tile == 0 && tx > 0);
                    const bool right_edge = (del.col_in_tile == 31 && tx + 1 < num_tiles_x);

                    if (top_edge) {
                        dirty_pass0[t - num_tiles_x] = 1;
                        dirty_pass1[t - num_tiles_x] = 1;
                    }
                    if (bottom_edge) {
                        dirty_pass0[t + num_tiles_x] = 1;
                        dirty_pass1[t + num_tiles_x] = 1;
                    }
                    if (left_edge) {
                        dirty_pass0[t - 1] = 1;
                        dirty_pass1[t - 1] = 1;
                    }
                    if (right_edge) {
                        dirty_pass0[t + 1] = 1;
                        dirty_pass1[t + 1] = 1;
                    }

                    if (top_edge && left_edge) {
                        dirty_pass0[t - num_tiles_x - 1] = 1;
                        dirty_pass1[t - num_tiles_x - 1] = 1;
                    }
                    if (top_edge && right_edge) {
                        dirty_pass0[t - num_tiles_x + 1] = 1;
                        dirty_pass1[t - num_tiles_x + 1] = 1;
                    }
                    if (bottom_edge && left_edge) {
                        dirty_pass0[t + num_tiles_x - 1] = 1;
                        dirty_pass1[t + num_tiles_x - 1] = 1;
                    }
                    if (bottom_edge && right_edge) {
                        dirty_pass0[t + num_tiles_x + 1] = 1;
                        dirty_pass1[t + num_tiles_x + 1] = 1;
                    }
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
