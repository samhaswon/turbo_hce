#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace turbo_hce::morphology {

/**
 * Compute the 2D Zhang-Suen skeleton of a binary image.
 *
 * Every nonzero input byte is treated as foreground. The returned image has
 * ``rows * columns`` elements in row-major order and contains only 0 and 1.
 * A zero-sized dimension returns an empty image. ``input`` may be null only
 * when the requested image has zero elements.
 *
 * :param input: Row-major input pixels.
 * :param rows: Number of image rows.
 * :param columns: Number of image columns.
 * :return: Row-major binary skeleton.
 * :throws std::invalid_argument: If a non-empty input has a null pointer.
 * :throws std::length_error: If rows * columns cannot be represented.
 */
void skeletonize_zhang_suen(const std::uint8_t* input,
                            std::size_t rows,
                            std::size_t columns,
                            std::uint8_t* output);

std::vector<std::uint8_t> skeletonize_zhang_suen(const std::uint8_t* input,
                                                  std::size_t rows,
                                                  std::size_t columns);

}  // namespace turbo_hce::morphology
