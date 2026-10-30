# ECE 759 — Assignment 4

The completed serial and parallel programs are `task2.cpp` and `task3.cpp`.
The supplied `nbody.py` and `plot_positions.py` remain unchanged.

## Submission

The code and supporting files belong in HW04 on GitHub main:
https://github.com/sammarth-kapse/repo759/tree/main/HW04

Upload these **three separate files** to Canvas, without a zip:

1. `assignment4.pdf` — report and explanations.
2. `task1.png` — the final image from the original Python simulation.
3. `task4.pdf` — three plots for static, dynamic, and guided scheduling.

The report and scheduling plots have editable, standalone LaTeX sources:
`assignment4.tex` and `task4.tex`. All plot coordinates are embedded in the sources.

## Build and run

These are the assignment's exact build commands on Euler:

```sh
g++ task2.cpp -Wall -O3 -std=c++17 -o task2
g++ task3.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp
```

The required interfaces are:

```sh
./task2 number_of_particles simulation_end_time
./task3 number_of_particles simulation_end_time num_threads
```

`task3` defaults to static scheduling. Set the standard `OMP_SCHEDULE` environment
variable to switch policies while keeping the same interface:

```sh
OMP_SCHEDULE=static ./task3 256 100 4
OMP_SCHEDULE=dynamic ./task3 256 100 4
OMP_SCHEDULE=guided ./task3 256 100 4
```

On Euler, execute computation through Slurm. The supplied batch script requests
**exactly eight CPUs**, runs correctness checks and compilation on the compute
node, then runs the complete experiment inside `srun`:

```sh
sbatch benchmark.sh
```

Both full benchmark entry points reject execution outside the required allocation.
The fixed C++ seed is 17; every policy/thread count receives the same initial data.
Each iteration owns one particle's output, so scheduling does not change summation
order within its acceleration.

For the Python image, install NumPy and Matplotlib in your Python environment,
run the unchanged `nbody.py` locally, and rename `nbody-python.png` to `task1.png`.
The submitted image uses the supplied defaults: N=100, end time=10, dt=0.01.
The Python and C++ random distributions produce different samples despite the same seed.

## Validation and measurements

Euler Slurm job **108475** completed with exit code 0 and eight allocated CPUs,
using its default GNU compiler and no additional modules.

- 60 acceleration checks passed against a long-double force reference, including
  single particles, coincident positions, and nonuniform masses.
- 50 complete trajectory checks passed against an independent direct Python
  implementation, covering all policies and 1, 2, and 8 threads.
- 10 invalid-input checks passed.
- Maximum tested position difference was 4.440892098500626e-16.
- Normal builds were checked to produce no positions CSV file.
- All 24 required scheduling configurations were measured three times:
  N=256, end time=100, dt=0.01, 10000 timesteps, threads=1 through 8.
- Three serial baseline measurements are also recorded: **75 measurements total**.

The raw data and provenance are in `benchmark.csv`, `benchmark.json`,
`validation.json`, and `source_manifest.json`. CSV writes are disabled in measured
C++ programs. Timing includes allocation/initialization, the initial acceleration,
and all simulation steps; argument parsing, printing, and deallocation are outside it.

To redo the checks locally with an OpenMP-capable GNU compiler:

```sh
python3 verify_hw04.py --compiler g++-15 --source-dir . --work-dir /tmp/hw04-validation
```

After collecting new data on Euler, copy `benchmark.csv`, `benchmark.json`, and
`validation.json` back to the same local folder. With LaTeX/pgfplots available,
regenerate the report and plot PDF:

```sh
python3 build_report.py . --output-dir . --compile
```

Only the Python standard library is needed for the C++ verification and benchmark
scripts. NumPy and Matplotlib are needed by the original simulation; the optional
original `plot_positions.py` also imports pandas.

## Learning checklist for redoing the work

1. Implement one particle's acceleration. Use the source particle's mass `mass[j]`,
   skip self-interaction, and add softening squared before taking the inverse cube.
2. Implement half kick, drift, coordinate clamping, force update, and half kick in
   that order. The C++ timestep count retains the supplied skeleton's truncation.
3. Parallelize the outer particle loops. Each particle writes its own acceleration,
   position, and velocity; a barrier must finish each stage before the next begins.
4. Use `schedule(runtime)` with `OMP_SCHEDULE` to compare policies fairly. Static
   has low scheduling overhead; dynamic/guided trade dispatch work for load balancing.
5. Check trajectories before timing. Build normally for measurements; define
   `NBODY_SAVE_POSITIONS` only for optional debug traces, never for benchmarks.
