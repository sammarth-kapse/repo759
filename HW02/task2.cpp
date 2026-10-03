#include "convolution.h"

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <random>
#include <string>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        return 1;
    }

    const std::size_t n = std::stoull(argv[1]);
    const std::size_t m = std::stoull(argv[2]);
    if (n == 0 || m == 0 || m % 2 == 0) {
        return 1;
    }

    std::random_device seed;
    std::mt19937 rng(seed());
    std::uniform_real_distribution<float> image_values(-10.0f, 10.0f);
    std::uniform_real_distribution<float> mask_values(-1.0f, 1.0f);

    float* image = new float[n * n];
    for (std::size_t i = 0; i < n * n; ++i) {
        image[i] = image_values(rng);
    }

    float* mask = new float[m * m];
    for (std::size_t i = 0; i < m * m; ++i) {
        mask[i] = mask_values(rng);
    }

    float* output = new float[n * n];

    const auto start = std::chrono::high_resolution_clock::now();
    convolve(image, output, n, mask, m);
    const auto stop = std::chrono::high_resolution_clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(stop - start).count();

    std::printf("%f\n", elapsed_ms);
    std::printf("%f\n", output[0]);
    std::printf("%f\n", output[n * n - 1]);

    delete[] image;
    delete[] mask;
    delete[] output;
    return 0;
}
