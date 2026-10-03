#include "scan.h"

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <random>
#include <string>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        return 1;
    }

    const std::size_t n = std::stoull(argv[1]);
    if (n == 0) {
        return 1;
    }

    std::random_device seed;
    std::mt19937 rng(seed());
    std::uniform_real_distribution<float> pick(-1.0f, 1.0f);

    float* arr = new float[n];
    float* output = new float[n];

    for (std::size_t i = 0; i < n; ++i) {
        arr[i] = pick(rng);
    }

    const auto start = std::chrono::high_resolution_clock::now();
    scan(arr, output, n);
    const auto stop = std::chrono::high_resolution_clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(stop - start).count();

    std::printf("%f\n", elapsed_ms);
    std::printf("%f\n", output[0]);
    std::printf("%f\n", output[n - 1]);

    delete[] arr;
    delete[] output;
    return 0;
}
