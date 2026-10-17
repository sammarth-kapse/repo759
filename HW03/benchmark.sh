#!/usr/bin/env bash
#SBATCH --partition=instruction
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --job-name=hw03
#SBATCH --output=hw03-%j.out
#SBATCH --error=hw03-%j.err

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || "${SLURM_CPUS_PER_TASK:-}" != "20" ]]; then
    echo "Submit this script with sbatch benchmark.sh; it requires a 20-CPU Slurm job." >&2
    exit 1
fi
cd "${SLURM_SUBMIT_DIR:-.}"
g++ task1.cpp matmul.cpp -Wall -O3 -std=c++17 -o task1 -fopenmp
g++ task2.cpp convolution.cpp -Wall -O3 -std=c++17 -o task2 -fopenmp
g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp

export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores
hostname
date -u
lscpu
srun --ntasks=1 --cpus-per-task=20 python3 collect_benchmarks.py --repeats 3
