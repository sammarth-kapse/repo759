#include "convolution.h"

#include <chrono>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::fprintf(stderr, "Usage: %s n t (n > 0, 1 <= t <= 20)\n", argv[0]);
        return 1;
    }

    try {
        const std::string n_text(argv[1]);
        const std::string t_text(argv[2]);
        if (n_text.empty() || t_text.empty() ||
            n_text.find_first_not_of("0123456789") != std::string::npos ||
            t_text.find_first_not_of("0123456789") != std::string::npos) {
            throw std::invalid_argument("n and t must be positive integers");
        }

        const auto n_input = std::stoull(n_text);
        const auto t_input = std::stoull(t_text);
        if (n_input == 0 || n_input > std::numeric_limits<std::size_t>::max() ||
            n_input > static_cast<unsigned long long>(
                          std::numeric_limits<std::ptrdiff_t>::max()) ||
            t_input < 1 || t_input > 20) {
            throw std::invalid_argument("require n > 0 and 1 <= t <= 20");
        }
        const std::size_t n = static_cast<std::size_t>(n_input);
        if (n > std::vector<float>().max_size() / n) {
            throw std::length_error("image is too large");
        }

        constexpr std::size_t m = 3;
        const std::size_t count = n * n;
        std::vector<float> image(count), output(count), mask(m * m);
        std::mt19937 rng(759);
        std::uniform_real_distribution<float> image_values(-10.0f, 10.0f);
        std::uniform_real_distribution<float> mask_values(-1.0f, 1.0f);
        for (float& pixel : image) {
            pixel = image_values(rng);
        }
        for (float& weight : mask) {
            weight = mask_values(rng);
        }

        omp_set_dynamic(0);
        omp_set_num_threads(static_cast<int>(t_input));
        const auto start = std::chrono::steady_clock::now();
        convolve(image.data(), output.data(), n, mask.data(), m);
        const auto stop = std::chrono::steady_clock::now();
        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(stop - start).count();

        std::printf("%f\n%f\n%f\n", output.front(), output.back(), elapsed_ms);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
