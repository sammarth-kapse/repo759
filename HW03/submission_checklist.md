# HW03 submission checklist

- [x] Six required C++ source files prepared in HW03, with no subdirectories.
- [x] Official headers preserved unchanged.
- [x] Full-array matrix/convolution reference checks pass.
- [x] Merge sort passes 367 reference comparisons.
- [x] Euler Slurm job 91407 completed successfully with 20 allocated CPUs.
- [x] All 210 benchmark runs and all four plot datasets validated.
- [x] Best tested merge-sort threshold: 256.
- [x] LaTeX report compiled and all PDF pages visually checked.
- [x] Report, plot PDFs, editable LaTeX, and raw benchmark data are ready.
- [ ] Upload the five PDFs listed below to Canvas and verify its submission receipt.

The GitHub destination for the source and supporting files is:
https://github.com/sammarth-kapse/repo759/tree/main/HW03

Canvas requires five separate files (no zip):

1. assignment3.pdf
2. task1.pdf
3. task2.pdf
4. task3_ts.pdf
5. task3_t.pdf

Use `sbatch benchmark.sh` for any future Euler benchmark run. The script requests
20 CPUs and runs computation in an `srun` step. The full benchmark entry points
reject execution outside that allocation. Local smoke tests are explicitly separate.
