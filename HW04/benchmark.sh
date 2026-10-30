#!/usr/bin/env bash
#SBATCH --partition=instruction
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --job-name=hw04
#SBATCH --output=hw04-%j.out
#SBATCH --error=hw04-%j.err

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || "${SLURM_CPUS_PER_TASK:-}" != "8" ]]; then
    echo "Submit with sbatch benchmark.sh; an eight-CPU Slurm job is required." >&2
    exit 1
fi
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores
hostname
date -u
lscpu
# Correctness tests and all compilation also execute on the allocated node.
srun --ntasks=1 --cpus-per-task=8 python3 verify_hw04.py --source-dir . --work-dir validation
cp validation/task2 task2
cp validation/task3 task3
cp validation/validation.json validation.json
srun --ntasks=1 --cpus-per-task=8 python3 collect_benchmarks.py --repeats 3
