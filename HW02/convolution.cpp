#include "convolution.h"

void convolve(const float* image, float* output, std::size_t n,
              const float* mask, std::size_t m) {
    const std::ptrdiff_t radius = static_cast<std::ptrdiff_t>(m / 2);
    const std::ptrdiff_t side = static_cast<std::ptrdiff_t>(n);

    for (std::size_t x = 0; x < n; ++x) {
        for (std::size_t y = 0; y < n; ++y) {
            float sum = 0.0f;

            for (std::size_t i = 0; i < m; ++i) {
                for (std::size_t j = 0; j < m; ++j) {
                    const std::ptrdiff_t row =
                        static_cast<std::ptrdiff_t>(x) +
                        static_cast<std::ptrdiff_t>(i) - radius;
                    const std::ptrdiff_t col =
                        static_cast<std::ptrdiff_t>(y) +
                        static_cast<std::ptrdiff_t>(j) - radius;

                    const bool row_inside = row >= 0 && row < side;
                    const bool col_inside = col >= 0 && col < side;

                    float pixel = 0.0f;
                    if (row_inside && col_inside) {
                        pixel = image[static_cast<std::size_t>(row) * n +
                                      static_cast<std::size_t>(col)];
                    } else if (row_inside || col_inside) {
                        pixel = 1.0f;
                    }

                    sum += mask[i * m + j] * pixel;
                }
            }

            output[x * n + y] = sum;
        }
    }
}
