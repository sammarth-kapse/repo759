#!/usr/bin/env python3
"""Validate HW04 measurements and generate standalone LaTeX submission files."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import tempfile

POLICIES = ('static', 'dynamic', 'guided')


def escape(value):
    return str(value).replace('_', r'\_').replace('&', r'\&').replace('%', r'\%')


def number(value):
    return format(value, '.5g')


def scientific(value):
    mantissa, exponent = format(value, '.2e').split('e')
    return mantissa + r'\times 10^{' + str(int(exponent)) + '}'


def read_data(folder):
    info = json.loads((folder/'benchmark.json').read_text())
    validation = json.loads((folder/'validation.json').read_text())
    if not re.fullmatch(r'[1-9][0-9]*', str(info.get('slurm_job_id', ''))):
        raise ValueError('A recorded Slurm job ID is required')
    if info.get('cpus_per_task') != 8 or info.get('particles', 0) < 100 or info.get('end_time', 0) < 100:
        raise ValueError('Required eight-CPU, N>=100, tEnd>=100 conditions are not met')
    if info.get('dt') != 0.01 or info.get('timesteps') != int(info['end_time']/0.01):
        raise ValueError('Recorded timestep count is inconsistent')
    if info.get('csv_output') is not False or info.get('seed') != 17:
        raise ValueError('Expected disabled CSV output and reproducible seed17')
    if validation.get('status') != 'passed' or validation.get('slurm_job_id') != info['slurm_job_id']:
        raise ValueError('Matching successful Euler correctness checks are required')
    repeats = info.get('repeats')
    if type(repeats) is not int or repeats < 3:
        raise ValueError('At least three measurements per configuration are required')
    groups = {(p, t): {} for p in POLICIES for t in range(1, 9)}
    groups[('serial', 1)] = {}
    with (folder/'benchmark.csv').open(newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['schedule', 'particles', 'end_time', 'threads', 'trial', 'time_ms']:
            raise ValueError('Unexpected benchmark columns')
        for row in reader:
            key = (row['schedule'], int(row['threads']))
            trial = int(row['trial'])
            elapsed = float(row['time_ms'])
            if key not in groups or trial in groups[key] or not 1 <= trial <= repeats:
                raise ValueError('Unexpected or duplicate benchmark configuration')
            if int(row['particles']) != info['particles'] or float(row['end_time']) != info['end_time']:
                raise ValueError('Benchmark dimensions disagree with metadata')
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError('Invalid elapsed time')
            groups[key][trial] = elapsed
    if any(set(times) != set(range(1, repeats+1)) for times in groups.values()):
        raise ValueError('Incomplete benchmark sweep')
    medians = {key: statistics.median(times.values()) for key, times in groups.items()}
    return info, validation, medians


PREAMBLE = r'''\documentclass[a4paper]{article}
\usepackage{geometry}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{times}
\usepackage[hidelinks]{hyperref}
\usepackage{fancyhdr}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\urlstyle{rm}
\setlength\parindent{0pt}
\geometry{top=1in,inner=1in,outer=1in,bottom=1in,headheight=3em,headsep=2em,footskip=3em}
\pagestyle{fancyplain}
\lhead{\fancyplain{}{Assignment 4}}
\rhead{\fancyplain{}{ECE 759 HPC}}
\cfoot{\thepage}
\title{\textsc{Assignment 4}}
\author{Sammarth Kapse \\ 9088922829 \\
\url{https://github.com/sammarth-kapse/repo759/tree/main/HW04}}
\date{}
\begin{document}
\maketitle
'''


def plot(policy, medians, height='5.7cm'):
    ymax = max(medians[(p, t)] for p in POLICIES for t in range(1, 9))*1.08
    coordinates = '\n'.join('(%d, %.12g)' % (t, medians[(policy, t)]) for t in range(1, 9))
    return (
        '\\begin{center}\n\\begin{tikzpicture}\n'
        '\\begin{axis}[width=0.94\\linewidth,height='+height+',\n'
        'title={'+policy.capitalize()+' scheduling},xlabel={Threads ($t$)},ylabel={Time (ms)},\n'
        'xmin=1,xmax=8,ymin=0,ymax=%.12g,xtick={1,2,3,4,5,6,7,8},\n' % ymax
        + 'grid=major,major grid style={gray!25},scaled y ticks=false,\n'
        'tick label style={font=\\small},label style={font=\\small},\n'
        '/pgf/number format/1000 sep={}]\n'
        '\\addplot[blue,thick,mark=*,mark size=2pt] coordinates {\n'+coordinates+'\n};\n'
        '\\end{axis}\n\\end{tikzpicture}\n\\end{center}\n'
    )


def sources(info, validation, medians):
    table = ['\\begin{center}\\begin{tabular}{lrrrr}\\hline',
             'Policy & 1 thread (ms) & 8 threads (ms) & Best $t$ & Best speedup \\\\ \\hline']
    for p in POLICIES:
        best = min(range(1, 9), key=lambda t: medians[(p, t)])
        table.append('%s & %s & %s & %d & $%s\\times$ \\\\' %
                     (p.capitalize(), number(medians[(p, 1)]), number(medians[(p, 8)]), best,
                      number(medians[(p, 1)]/medians[(p, best)])))
    table.extend(['\\hline\\end{tabular}\\end{center}',
                  '{\\small Speedups use the same policy at one thread as the baseline.}\n'])
    best_policy, best_t = min(((p, t) for p in POLICIES for t in range(1, 9)), key=medians.get)
    report = PREAMBLE + r'''
\section*{Task 1}
I ran the supplied \texttt{nbody.py} locally without changing its source. It uses
$N=100$, $t_{\mathrm{end}}=10$, $\Delta t=0.01$, and seed 17. The saved final
particle/trail plot is submitted separately as \texttt{task1.png}.

\section*{Task 2}
The serial C++ program completes the supplied skeleton. For each particle it computes
\[
 \mathbf a_i=G\sum_{j\ne i}m_j
 \frac{\mathbf r_j-\mathbf r_i}
 {(\|\mathbf r_j-\mathbf r_i\|^2+\epsilon^2)^{3/2}}.
\]
The constants are $G=1$, $\epsilon=0.1$, $\Delta t=0.01$, and board limits $[-4,4]$.
Velocities are converted to the center-of-mass frame. Each timestep applies a half kick,
drifts and clamps positions, recomputes acceleration, and applies the second half kick.
The timestep count follows the supplied C++ skeleton.

\section*{Task 3}
OpenMP parallel loops distribute particles among threads. Every force-loop iteration
owns one output acceleration and keeps its summation over other particles serial.
The implicit barriers finish each stage before the next stage uses its results.
\texttt{schedule(runtime)} and \texttt{OMP\_SCHEDULE} select static, dynamic, or guided
without changing the required command-line interface. Both C++ versions use seed 17;
NumPy and C++ generators produce different initial samples, so their plots need not coincide.

'''
    report += ('\\textbf{Validation:} On Euler, %d force checks, %d full-trajectory checks, and %d invalid-input '
               'checks passed. The largest position difference from an independent Python reference was '
               '$%s$. Both required compile commands worked without loading additional modules. '
               'Per-step CSV output is disabled in benchmark builds.\n'
               % (validation['acceleration_checks'], validation['trajectory_checks'],
                  validation['invalid_argument_checks'], scientific(validation['max_position_error'])))
    setup = (
        'All experiments used Euler node \\texttt{%s}, Slurm job %s, and exactly eight requested CPUs. '
        'The workload was $N=%d$, $t_{\\mathrm{end}}=%s$, $\\Delta t=0.01$ (%d timesteps). '
        'Each point is the median of %d fresh-process runs. Timings include C++ initialization, '
        'the initial acceleration, and the simulation loop; CSV I/O and printing are excluded.\n'
        % (escape(info['hostname']), escape(info['slurm_job_id']), info['particles'],
           number(info['end_time']), info['timesteps'], info['repeats']))
    report += '\\newpage\n\\section*{Task 4}\n'+setup+'\n'+'\n'.join(table)+'\n'
    report += ('\nThe serial Task 2 baseline was %s ms. The fastest measured OpenMP configuration was '
               '%s scheduling with %d threads (%s ms). Static scheduling assigns fixed chunks with little '
               'dispatch overhead. Dynamic scheduling hands out chunks as threads become free; guided '
               'scheduling uses progressively smaller chunks. The force loop has almost equal work per '
               'particle, so frequent dynamic dispatch can add overhead without fixing much imbalance. '
               'Repeated parallel regions and barriers, and shared hardware, can also limit scaling. '
               'These are possible explanations; the measurements do not isolate each cost.\n'
               % (number(medians[('serial', 1)]), best_policy, best_t,
                  number(medians[(best_policy, best_t)])))
    report += ('\nThe three required linear--linear plots are included in \\texttt{task4.pdf}, '
               'and are reproduced on the following page with identical axis limits.\n'
               '\\newpage\n\\section*{Task 4: Scheduling plots}\n')
    report += ''.join(plot(p, medians, '5.8cm') for p in POLICIES)
    report += '\\end{document}\n'
    plot_doc = r'''\documentclass[a4paper]{article}
\usepackage[margin=0.65in]{geometry}
\usepackage{times}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\pagestyle{empty}
\begin{document}
\begin{center}{\Large\bfseries Task 4: OpenMP scheduling}\\[5pt]
'''
    plot_doc += ('$N=%d$, $t_{\\mathrm{end}}=%s$, $\\Delta t=0.01$; median of %d runs on Euler.\\\\\n'
                 'All three panels use linear axes and identical limits.\\end{center}\n'
                 % (info['particles'], number(info['end_time']), info['repeats']))
    plot_doc += ''.join(plot(p, medians, '6.5cm') for p in POLICIES)
    plot_doc += '\\end{document}\n'
    return {'assignment4': report, 'task4': plot_doc}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--pdflatex', default='pdflatex')
    args = parser.parse_args()
    try:
        info, validation, medians = read_data(args.input_dir)
        generated = sources(info, validation, medians)
        with tempfile.TemporaryDirectory(prefix='hw04-report-') as directory:
            tmp = Path(directory)
            for name, source in generated.items():
                (tmp/(name+'.tex')).write_text(source)
                if args.compile:
                    result = subprocess.run([args.pdflatex, '-interaction=nonstopmode', '-halt-on-error',
                                             '-no-shell-escape', name+'.tex'], cwd=tmp, text=True,
                                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    if result.returncode:
                        raise ValueError('LaTeX compilation failed:\n'+result.stdout[-6000:])
            args.output_dir.mkdir(parents=True, exist_ok=True)
            for name in generated:
                for extension in ('.tex', '.pdf') if args.compile else ('.tex',):
                    shutil.copy2(tmp/(name+extension), args.output_dir/(name+extension))
        print('Validated %d measurements, 24 scheduling configurations, and Euler correctness checks.'
              % (25*info['repeats']))
        print('Generated', ', '.join(generated), 'in', args.output_dir.resolve())
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, str(error)+'\n')


if __name__ == '__main__':
    main()
