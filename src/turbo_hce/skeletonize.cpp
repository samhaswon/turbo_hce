#include "skeletonize.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

// 2D Zhang-Suen morphological skeletonization.
// Lookup table and 8-neighbor connectivity encoding derived from scikit-image
// (BSD-3-Clause license, Copyright (c) 2011-2024, the scikit-image team).

namespace turbo_hce::morphology {
namespace {

constexpr std::uint8_t ZHANG_SUEN_LUT[256] = {
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

    for (std::size_t r = 0; r < rows; ++r) {
        const std::uint8_t* in_row = input + r * columns;
        std::uint8_t* pad_row = pad.data() + (r + 1) * padded_cols + 1;
        for (std::size_t c = 0; c < columns; ++c) {
            pad_row[c] = (in_row[c] != 0) ? 1 : 0;
        }
    }

    std::vector<std::uint8_t> cleaned = pad;

    bool pixel_removed = true;
    while (pixel_removed) {
        pixel_removed = false;
        for (std::uint8_t pass = 0; pass < 2; ++pass) {
            const bool first_pass = (pass == 0);
            for (std::size_t r = 1; r <= rows; ++r) {
                const std::size_t r_offset = r * padded_cols;
                const std::size_t prev_row = r_offset - padded_cols;
                const std::size_t next_row = r_offset + padded_cols;

                for (std::size_t c = 1; c <= columns; ++c) {
                    const std::size_t idx = r_offset + c;
                    if (pad[idx] == 0) {
                        continue;
                    }

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

                    const std::uint8_t n = ZHANG_SUEN_LUT[code];
                    if (n == 3 || (n == 1 && first_pass) || (n == 2 && !first_pass)) {
                        cleaned[idx] = 0;
                        pixel_removed = true;
                    }
                }
            }
            pad = cleaned;
        }
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
