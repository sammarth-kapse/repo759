#include "msort.h"

#include <algorithm>
#include <vector>

namespace {

void merge_sort(int* arr, int* scratch, const std::size_t begin,
                const std::size_t end, const std::size_t threshold) {
    const std::size_t length = end - begin;
    if (length <= 1) {
        return;
    }
    if (length < threshold) {
        // Heapsort is iterative and avoids creating tasks for small subarrays.
        std::make_heap(arr + begin, arr + end);
        std::sort_heap(arr + begin, arr + end);
        return;
    }

    const std::size_t middle = begin + length / 2;

#pragma omp task default(none) firstprivate(arr, scratch, begin, middle, threshold)
    {
        merge_sort(arr, scratch, begin, middle, threshold);
    }

#pragma omp task default(none) firstprivate(arr, scratch, middle, end, threshold)
    {
        merge_sort(arr, scratch, middle, end, threshold);
    }

#pragma omp taskwait
    // Sibling tasks use disjoint slices of the same temporary array.
    std::merge(arr + begin, arr + middle, arr + middle, arr + end,
               scratch + begin);
    std::copy(scratch + begin, scratch + end, arr + begin);
}

}  // namespace

void msort(int* arr, const std::size_t n, const std::size_t threshold) {
    if (n <= 1) {
        return;
    }

    std::vector<int> scratch(n);
    int* const buffer = scratch.data();

    // One team executes the entire task tree; recursion never creates teams.
#pragma omp parallel default(none) firstprivate(arr, buffer, n, threshold)
    {
#pragma omp single
        {
            merge_sort(arr, buffer, 0, n, threshold);
        }
    }
}
