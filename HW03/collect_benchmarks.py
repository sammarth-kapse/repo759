#!/usr/bin/env python3
"""Collect kernel timings; select the merge-sort cutoff by median runtime."""

import argparse
import csv
import json
import math
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--smoke", action="store_true", help="Small local check only")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not args.smoke and (
        not os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_CPUS_PER_TASK") != "20"
        or not os.environ.get("SLURM_STEP_ID")
    ):
        parser.error("Full benchmarks require a 20-CPU Slurm step; use sbatch benchmark.sh")

    threads = [1, 2, 4] if args.smoke else list(range(1, 21))
    thresholds = [2, 16, 128] if args.smoke else [2**p for p in range(1, 11)]
    matrix_n = 64 if args.smoke else 1024
    sort_n = 2048 if args.smoke else 1_000_000
    sort_threads = 4 if args.smoke else 8
    suffix = "_smoke" if args.smoke else ""
    result_path = Path("benchmark" + suffix + ".csv")
    start = time.time()
    metadata = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "smoke": args.smoke,
        "repeats": args.repeats,
        "omp_proc_bind": os.environ.get("OMP_PROC_BIND"),
        "omp_places": os.environ.get("OMP_PLACES"),
    }
    fields = ["phase", "task", "n", "threads", "threshold", "trial", "first", "last", "time_ms"]
    with result_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()

        def run(task, n, t, threshold=0, phase="threads"):
            times = []
            for trial in range(1, args.repeats + 1):
                command = ["./task" + str(task), str(n), str(t)]
                if task == 3:
                    command.append(str(threshold))
                output = subprocess.check_output(command, text=True).splitlines()
                if len(output) != 3:
                    raise RuntimeError("Expected exactly three output lines: " + repr(output))
                first, last, elapsed = map(float, output)
                if not all(map(math.isfinite, (first, last, elapsed))) or elapsed <= 0:
                    raise RuntimeError("Invalid numeric output from " + repr(command))
                writer.writerow(dict(zip(fields, [phase, task, n, t, threshold, trial,
                                                  first, last, elapsed])))
                stream.flush()
                times.append(elapsed)
            median = statistics.median(times)
            print("task%d n=%d threads=%d threshold=%d median_ms=%.6f" %
                  (task, n, t, threshold, median), flush=True)
            return median

        for task in (1, 2):
            for t in threads:
                run(task, matrix_n, t)
        threshold_times = {ts: run(3, sort_n, sort_threads, ts, phase="threshold")
                           for ts in thresholds}
        best = min(threshold_times, key=threshold_times.get)
        metadata["best_threshold"] = best
        metadata["threshold_median_ms"] = threshold_times
        for t in threads:
            run(3, sort_n, t, best)

    metadata["elapsed_seconds"] = time.time() - start
    Path("benchmark" + suffix + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
    Path("best_threshold" + suffix + ".txt").write_text(str(best) + "\n")
    print("Saved", result_path, "with best threshold", best, flush=True)


if __name__ == "__main__":
    main()
