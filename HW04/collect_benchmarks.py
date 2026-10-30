#!/usr/bin/env python3
"""Run the required HW04 scheduling sweeps inside an eight-CPU Slurm step."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import socket
import statistics
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_CPUS_PER_TASK') != '8' or not os.environ.get('SLURM_STEP_ID'):
        parser.error('Use sbatch benchmark.sh; benchmarks require an eight-CPU Slurm step')
    n, end_time = 256, 100
    metadata = dict(hostname=socket.getfqdn(), slurm_job_id=os.environ['SLURM_JOB_ID'],
                    cpus_per_task=8, particles=n, end_time=end_time, dt=0.01,
                    timesteps=10000, repeats=args.repeats, seed=17,
                    omp_proc_bind=os.environ.get('OMP_PROC_BIND'),
                    omp_places=os.environ.get('OMP_PLACES'), csv_output=False)
    start = time.time()
    with Path('benchmark.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['schedule', 'particles', 'end_time', 'threads', 'trial', 'time_ms'])

        def measure(schedule, threads, trial):
            env = os.environ.copy()
            env['OMP_SCHEDULE'] = 'static' if schedule == 'serial' else schedule
            command = ['./task2', str(n), str(end_time)] if schedule == 'serial' else [
                './task3', str(n), str(end_time), str(threads)]
            output = subprocess.check_output(command, env=env, text=True)
            match = re.fullmatch(r'time: ([0-9]+(?:\.[0-9]+)?)ms\n', output)
            if not match:
                raise RuntimeError('Unexpected program output: ' + repr(output))
            elapsed = float(match.group(1))
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise RuntimeError('Invalid simulation duration')
            writer.writerow([schedule, n, end_time, threads, trial, elapsed])
            stream.flush()
            print('%s t=%d trial=%d: %.6f ms' % (schedule, threads, trial, elapsed), flush=True)

        for trial in range(1, args.repeats+1):
            measure('serial', 1, trial)
            # Interleave policies within each repeat rather than timing them in separate jobs.
            for threads in range(1, 9):
                for schedule in ('static', 'dynamic', 'guided'):
                    measure(schedule, threads, trial)
    metadata['wall_seconds'] = time.time()-start
    metadata['compiler'] = subprocess.check_output(['g++', '--version'], text=True).splitlines()[0]
    Path('benchmark.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print('Completed all 24 scheduling configurations, plus serial baseline.', flush=True)


if __name__ == '__main__':
    main()
