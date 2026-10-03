#include "matmul.h"

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <random>
#include <vector>

int main() {
    constexpr unsigned int n = 1024;
    const std::size_t count = static_cast<std::size_t>(n) * n;
    std::vector<double> A(count), B(count), C(count);

    std::random_device seed;
    std::mt19937 rng(seed());
    std::uniform_int_distribution<int> value(-2, 2);

    for (std::size_t p = 0; p < count; ++p) {
        A[p] = value(rng);
        B[p] = value(rng);
    }

    using Clock = std::chrono::high_resolution_clock;
    std::printf("%u\n", n);

    auto start = Clock::now();
    mmul1(A.data(), B.data(), C.data(), n);
    auto stop = Clock::now();
    std::printf("%f\n",
                std::chrono::duration<double, std::milli>(stop - start).count());
    std::printf("%f\n", C.back());
    const std::vector<double> reference = C;

    start = Clock::now();
    mmul2(A.data(), B.data(), C.data(), n);
    stop = Clock::now();
    if (C != reference) {
        std::fprintf(stderr, "mmul2 produced a different matrix\n");
        return 1;
    }
    std::printf("%f\n",
                std::chrono::duration<double, std::milli>(stop - start).count());
    std::printf("%f\n", C.back());

    start = Clock::now();
    mmul3(A.data(), B.data(), C.data(), n);
    stop = Clock::now();
    if (C != reference) {
        std::fprintf(stderr, "mmul3 produced a different matrix\n");
        return 1;
    }
    std::printf("%f\n",
                std::chrono::duration<double, std::milli>(stop - start).count());
    std::printf("%f\n", C.back());

    start = Clock::now();
    mmul4(A, B, C.data(), n);
    stop = Clock::now();
    if (C != reference) {
        std::fprintf(stderr, "mmul4 produced a different matrix\n");
        return 1;
    }
    std::printf("%f\n",
                std::chrono::duration<double, std::milli>(stop - start).count());
    std::printf("%f\n", C.back());

    return 0;
}
