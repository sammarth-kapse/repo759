# ECE 759 — HW03

## Build on Euler

```sh
g++ task1.cpp matmul.cpp -Wall -O3 -std=c++17 -o task1 -fopenmp
g++ task2.cpp convolution.cpp -Wall -O3 -std=c++17 -o task2 -fopenmp
g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp
```

The three drivers accept `n t`, `n t`, and `n t ts`, respectively. Each prints
the first result, last result, and kernel time in milliseconds on separate lines.
The fixed seed gives identical inputs for each thread count. On this Mac,
use `g++-15` to compile with OpenMP.

## Collect the required timings

From the directory containing these files on Euler:

```sh
sbatch benchmark.sh
```

The script requests 20 CPUs and runs all prescribed parameter sweeps. It records
three runs per configuration in `benchmark.csv`, selects the threshold using the
median at eight threads, and records the node/job/settings in `benchmark.json`.
Data from the threshold and thread sweeps are kept separate by the `phase` column.
Both benchmark entry points reject full runs outside the required Slurm allocation.
Submit with `sbatch`; the script starts the measurements inside `srun` on the
allocated compute node. Do not run the benchmark programs on an Euler login node.

After copying the results back to a machine with LaTeX and pgfplots:

```sh
python3 build_report.py . --output-dir . --compile
```

This regenerates the editable standalone `assignment3.tex`, `assignment3.pdf`,
and `task1.pdf`, `task2.pdf`, `task3_ts.pdf`, `task3_t.pdf`. All plot coordinates
are embedded in the main LaTeX source, so the report does not depend on image files.

## Ideas to understand

- **Matrix multiplication:** keep the HW02 `i-k-j` loop order; parallelize `i`.
  A thread owns an entire output row. The inner loop accesses consecutive
  elements of B and C, while reusing one value of A.
- **Convolution:** output pixels are independent. Each pixel's accumulator is
  local, and the image/mask are shared read-only. Preserve the HW02 edge-strip
  padding of 1 and corner padding of 0.
- **Merge sort:** a single OpenMP team executes recursive tasks. Both children
  must finish before their parent merges. Scratch regions are disjoint between
  sibling tasks. Arrays strictly smaller than the cutoff use iterative heapsort.
- **Cutoff:** smaller values expose more parallel tasks but cost more scheduling;
  larger values reduce scheduling while leaving more serial work in each leaf.
  The chosen value is the fastest tested median on Euler, not a universal constant.
- **Scaling:** speedup is `time_at_one_thread / time_at_t_threads`. Team startup,
  task scheduling, shared hardware, and the serial merge at the root can prevent
  ideal scaling. The timing plots alone cannot identify each contribution.

The three provided headers are unchanged official course copies. Submission uses
the six `.cpp` files in HW03 on GitHub main, the report, and four plot PDFs on
Canvas. Publication and Canvas submission are separate from running the code.

## Completion checklist

- [x] All six required C++ source files are present directly in HW03.
- [x] All three supplied headers match the official originals byte for byte.
- [x] Matrix multiplication and convolution match full serial reference outputs.
- [x] Merge sort passes 367 comparisons against `std::sort`, including boundary sizes.
- [x] Euler Slurm job 91407 completed with exit code 0 and 20 allocated CPUs.
- [x] All 70 prescribed configurations have three timing measurements (210 total).
- [x] Best tested merge-sort threshold is 256, selected using median runtime.
- [x] All plot coordinates and report numbers match the recorded measurements.
- [x] The standalone LaTeX report compiles; all three report pages and four plots were visually checked.
- [x] LaTeX, PDFs, measurements, and reproduction tools are included alongside the source.
- [ ] Upload the five PDFs below to Canvas and confirm the submission receipt.

GitHub submission location:
https://github.com/sammarth-kapse/repo759/tree/main/HW03

Upload these five files separately to Canvas (do not zip):

1. `assignment3.pdf`
2. `task1.pdf`
3. `task2.pdf`
4. `task3_ts.pdf`
5. `task3_t.pdf`
