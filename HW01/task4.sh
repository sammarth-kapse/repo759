#!/usr/bin/env bash
#SBATCH -p instruction
#SBATCH --cpus-per-task=2
#SBATCH --job-name=FirstSlurm
#SBATCH --output=FirstSlurm.out
#SBATCH -p instruction
#SBATCH --error=FirstSlurm.err

hostname
