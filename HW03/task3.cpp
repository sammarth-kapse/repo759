#include "msort.h"

#include <charconv>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

std::size_t positive_integer(const char* text) {
    std::size_t value = 0;
    const char* const end = text + std::strlen(text);
    const auto result = std::from_chars(text, end, value);
    if (result.ec != std::errc{} || result.ptr != end || value == 0) {
        throw std::invalid_argument("arguments must be positive integers");
    }
    return value;
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " n t ts\n";
        return 1;
    }

    try {
        const std::size_t n = positive_integer(argv[1]);
        const std::size_t threads = positive_integer(argv[2]);
        const std::size_t threshold = positive_integer(argv[3]);
        if (threads > 20) {
            throw std::invalid_argument("thread count must be between 1 and 20");
        }

        omp_set_dynamic(0);
        omp_set_num_threads(static_cast<int>(threads));
        std::mt19937 generator(759);
        std::uniform_int_distribution<int> distribution(-1000, 1000);
        std::vector<int> arr(n);
        for (int& value : arr) {
            value = distribution(generator);
        }

        const auto start = std::chrono::steady_clock::now();
        msort(arr.data(), n, threshold);
        const auto stop = std::chrono::steady_clock::now();
        const double milliseconds =
            std::chrono::duration<double, std::milli>(stop - start).count();

        std::cout << arr.front() << '\n'
                  << arr.back() << '\n'
                  << std::fixed << std::setprecision(6) << milliseconds << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
