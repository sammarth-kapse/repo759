#include "matmul.h"

#include <algorithm>

void mmul(const float* A, const float* B, float* C, const std::size_t n) {
    // Each thread owns complete rows of C, so accumulation needs no locks.
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t row = i * n;
        std::fill_n(C + row, n, 0.0f);

        // Keep the HW02 mmul2 i-k-j order for contiguous accesses to B and C.
        for (std::size_t k = 0; k < n; ++k) {
            const float a = A[row + k];
            for (std::size_t j = 0; j < n; ++j) {
                C[row + j] += a * B[k * n + j];
            }
        }
    }
}
