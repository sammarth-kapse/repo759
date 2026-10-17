#!/usr/bin/env python3
"""Build the HW03 LaTeX report and plots from complete Euler benchmark data.

Usage: python3 build_report.py RESULTS --output-dir OUTPUTS [--compile]
Only the Python standard library is required. --compile additionally needs
pdflatex with pgfplots and the standalone document class installed.
"""

import argparse
import csv
import json
import math
import re
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path


INTEGER_FIELDS = ["task", "n", "threads", "threshold", "trial"]
RESULT_FIELDS = ["first", "last", "time_ms"]
FIELDS = ["phase"] + INTEGER_FIELDS + RESULT_FIELDS
THREADS = list(range(1, 21))
THRESHOLDS = [2**power for power in range(1, 11)]
FILENAMES = ["assignment3", "task1", "task2", "task3_ts", "task3_t"]


def tex_escape(value):
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def number(value):
    """Readable prose without rounding short measured durations to zero."""
    return format(value, ".5g")


def read_benchmarks(input_dir):
    metadata = json.loads((input_dir / "benchmark.json").read_text())
    if not isinstance(metadata, dict):
        raise ValueError("benchmark.json must contain an object")
    if metadata.get("smoke") is not False:
        raise ValueError("Refusing smoke or unmarked data; metadata smoke must be false")
    job = str(metadata.get("slurm_job_id", ""))
    if not re.fullmatch(r"[1-9][0-9]*", job):
        raise ValueError("A valid Slurm job ID is required")
    if str(metadata.get("slurm_cpus_per_task", "")) != "20":
        raise ValueError("Benchmark metadata must record 20 Slurm CPUs per task")
    if not isinstance(metadata.get("hostname"), str) or not metadata["hostname"].strip():
        raise ValueError("Benchmark metadata must identify the compute hostname")
    repeats = metadata.get("repeats")
    if type(repeats) is not int or repeats < 1:
        raise ValueError("Metadata repeats must be a positive integer")

    with (input_dir / "benchmark.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError("CSV columns must match collect_benchmarks.py: " + ",".join(FIELDS))
        rows = []
        for line, raw in enumerate(reader, 2):
            if None in raw or any(raw[field] is None for field in FIELDS):
                raise ValueError("Malformed CSV row at line %d" % line)
            try:
                row = {key: int(raw[key]) for key in INTEGER_FIELDS}
                row["phase"] = raw["phase"]
                row.update({key: float(raw[key]) for key in RESULT_FIELDS})
            except (TypeError, ValueError) as error:
                raise ValueError("Invalid numeric value at CSV line %d" % line) from error
            if any(not math.isfinite(row[key]) for key in RESULT_FIELDS):
                raise ValueError("Nonfinite number at CSV line %d" % line)
            if row["time_ms"] <= 0:
                raise ValueError("Nonpositive timing at CSV line %d" % line)
            if row["task"] == 3 and not (
                row["first"].is_integer() and row["last"].is_integer()
                and -1000 <= row["first"] <= row["last"] <= 1000
            ):
                raise ValueError("Invalid merge-sort endpoints at CSV line %d" % line)
            rows.append(row)

    if len(rows) != 70 * repeats:
        raise ValueError("Expected %d benchmark rows; found %d" % (70 * repeats, len(rows)))

    # The collector writes four ordered sweeps. The best cutoff at t=8 occurs
    # twice, so reading phases separately preserves independent measurements.
    offset = 0

    def sweep(phase, task, n, configurations):
        nonlocal offset
        medians = {}
        for x, threads, threshold in configurations:
            times = []
            for trial in range(1, repeats + 1):
                row = rows[offset]
                expected = (phase, task, n, threads, threshold, trial)
                actual = tuple(row[key] for key in ["phase"] + INTEGER_FIELDS)
                if actual != expected:
                    raise ValueError("Unexpected configuration at CSV line %d: expected %r; got %r"
                                     % (offset + 2, expected, actual))
                times.append(row["time_ms"])
                offset += 1
            medians[x] = statistics.median(times)
        return medians

    curves = {
        "task1": sweep("threads", 1, 1024, [(t, t, 0) for t in THREADS]),
        "task2": sweep("threads", 2, 1024, [(t, t, 0) for t in THREADS]),
        "task3_ts": sweep("threshold", 3, 1_000_000, [(ts, 8, ts) for ts in THRESHOLDS]),
    }
    best = min(THRESHOLDS, key=curves["task3_ts"].get)
    if type(metadata.get("best_threshold")) is not int or metadata["best_threshold"] != best:
        raise ValueError("Metadata best_threshold disagrees with the measured minimum median")
    recorded = metadata.get("threshold_median_ms")
    if not isinstance(recorded, dict) or set(recorded) != {str(ts) for ts in THRESHOLDS}:
        raise ValueError("Metadata threshold_median_ms must contain all ten measured cutoffs")
    for ts in THRESHOLDS:
        value = recorded[str(ts)]
        if type(value) not in (int, float) or not math.isfinite(value) or not math.isclose(
            value, curves["task3_ts"][ts], rel_tol=1e-10, abs_tol=1e-8
        ):
            raise ValueError("Metadata median disagrees with CSV for cutoff %d" % ts)
    curves["task3_t"] = sweep("threads", 3, 1_000_000, [(t, t, best) for t in THREADS])
    return metadata, curves, best


def plot(curve, xlabel, log_x=False, height="6.5cm", width=r"0.96\linewidth"):
    coordinates = "\n".join("(%d, %.12g)" % (x, curve[x]) for x in sorted(curve))
    x_options = (
        "xmode=log,log basis x=2,xtick={2,4,8,16,32,64,128,256,512,1024},"
        "xticklabels={2,4,8,16,32,64,128,256,512,1024},"
        "x tick label style={font=\\scriptsize},"
        if log_x else "xmin=1,xmax=20,xtick={1,2,4,6,8,10,12,14,16,18,20},"
    )
    return (
        "\\begin{tikzpicture}\n"
        "\\begin{axis}[width=" + width + ",height=" + height + ",\n"
        "xlabel={" + xlabel + "},ylabel={Time (ms)},\n"
        + x_options + "\n"
        "ymin=0,grid=both,major grid style={gray!25},minor grid style={gray!10},\n"
        "tick label style={font=\\small},label style={font=\\small},\n"
        "scaled y ticks=false,/pgf/number format/1000 sep={}]\n"
        "\\addplot[blue,thick,mark=*,mark size=1.8pt] coordinates {\n"
        + coordinates + "\n};\n\\end{axis}\n\\end{tikzpicture}\n"
    )


def centered_plot(curve, xlabel, caption, log_x=False, height="6.5cm"):
    return ("\\begin{center}\n" + plot(curve, xlabel, log_x, height)
            + "\n{\\small " + caption + "}\n\\end{center}\n")


def report(metadata, curves, best):
    task1, task2 = curves["task1"], curves["task2"]
    task3_ts, task3_t = curves["task3_ts"], curves["task3_t"]
    fastest1 = min(THREADS, key=task1.get)
    fastest2 = min(THREADS, key=task2.get)
    fastest3 = min(THREADS, key=task3_t.get)
    preamble = r"""\documentclass[a4paper]{article}
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
\lhead{\fancyplain{}{Assignment 3}}
\rhead{\fancyplain{}{ECE 759 HPC}}
\cfoot{\thepage}
\title{\textsc{Assignment 3}}
\author{Sammarth Kapse \\ 9088922829 \\
\url{https://github.com/sammarth-kapse/repo759/tree/main/HW03}}
\date{}
\begin{document}
\maketitle
"""
    provenance = (
        "All measurements were collected on Euler node \\texttt{%s} in Slurm job %s "
        "with 20 CPUs requested. Each point is the median of %d runs; timings "
        "cover the kernel function only.\n"
        % (tex_escape(metadata["hostname"]), tex_escape(metadata["slurm_job_id"]), metadata["repeats"])
    )
    section1 = (
        "\\section*{Task 1}\n"
        "Matrix multiplication retains the $i$--$k$--$j$ loop order from HW02. "
        "An OpenMP parallel loop distributes rows of $C$ among threads. Each "
        "thread initializes and computes its own rows, so no reduction or lock is needed.\n\n"
        "For $n=1024$, the median time changed from %s ms at one thread to "
        "%s ms at 20 threads, a speedup of $%s\\times$. The fastest measured "
        "configuration used %d threads (%s ms; $%s\\times$ speedup).\n"
        % (number(task1[1]), number(task1[20]), number(task1[1] / task1[20]),
           fastest1, number(task1[fastest1]), number(task1[1] / task1[fastest1]))
        + centered_plot(task1, r"Threads ($t$)", r"Matrix multiplication: $n=1024$.")
    )
    if task2[20] > task2[fastest2] * 1.05:
        tail_observation = (
            "At 20 threads the time was %s ms, %s\\%% above the minimum; "
            "using all 20 threads did not improve the measured minimum."
            % (number(task2[20]), number(100 * (task2[20] / task2[fastest2] - 1)))
        )
    else:
        tail_observation = (
            "At 20 threads the time was %s ms, within %s\\%% of the measured minimum."
            % (number(task2[20]), number(100 * (task2[20] / task2[fastest2] - 1)))
        )
    # Quantify the high-thread range instead of assuming the curve plateaus.
    high = [task2[t] for t in range(10, 21)]
    spread = 100 * (max(high) / min(high) - 1)
    if spread <= 15:
        saturation = (
            "The 10--20 thread timings span only %s\\%% relative to their minimum, "
            "which is evidence of saturation in that range." % number(spread)
        )
    elif task2[20] >= task2[10]:
        saturation = (
            "Increasing from 10 to 20 threads changed the time from %s to %s ms, "
            "so the additional threads did not reduce runtime in that comparison."
            % (number(task2[10]), number(task2[20]))
        )
    else:
        saturation = (
            "Increasing from 10 to 20 threads reduced time from %s to %s ms "
            "($%s\\times$); these measurements do not establish a flat plateau "
            "throughout that range."
            % (number(task2[10]), number(task2[20]), number(task2[10] / task2[20]))
        )
    section2 = (
        "\\newpage\n\\section*{Task 2}\n"
        "An OpenMP parallel loop distributes output rows among threads. The input "
        "image and mask are read-only, and each output pixel has a private accumulator. "
        "The HW02 boundary convention is preserved: ones on edge strips and zeros "
        "where both coordinates are outside the image.\n\n"
        "For $n=1024$ and a $3\\times3$ mask, the time at one thread was %s ms. "
        "The fastest median was %s ms at %d threads ($%s\\times$ speedup). %s %s\n\n"
        "A plausible explanation for diminishing returns is that each pixel requires "
        "little work: OpenMP team and scheduling costs become substantial, while "
        "threads share caches and memory bandwidth. These are performance "
        "interpretations; the timing experiment does not measure those costs separately.\n"
        % (number(task2[1]), number(task2[fastest2]), fastest2,
           number(task2[1] / task2[fastest2]), tail_observation, saturation)
        + centered_plot(task2, r"Threads ($t$)", r"Convolution: $n=1024$, mask size $3\times3$.", height="8cm")
    )
    section3 = (
        "\\newpage\n\\section*{Task 3}\n"
        "Merge sort creates tasks for the two halves, waits for both, and then merges "
        "their sorted results. Below the threshold, iterative heap sorting avoids "
        "recursive calls and task creation.\n\n"
        "For $n=10^6$ and eight threads, the best tested threshold was $ts=%d$, "
        "with a median time of %s ms. The smallest threshold, $ts=2$, took %s ms. "
        "Small thresholds create many tiny tasks, increasing scheduling overhead. "
        "Large thresholds leave more work in serial leaf sorts; the sweep measures "
        "this tradeoff for the tested implementation and node.\n"
        % (best, number(task3_ts[best]), number(task3_ts[2]))
        + centered_plot(task3_ts, r"Threshold ($ts$)",
                        r"Threshold sweep: $n=10^6$, $t=8$; logarithmic horizontal axis.",
                        log_x=True, height="5.2cm")
        + ("At $ts=%d$, the fastest thread count was %d (%s ms). One thread took "
           "%s ms and 20 threads took %s ms ($%s\\times$ speedup). Serial merging "
           "near the top of the recursion tree and task overhead limit ideal scaling.\n"
           % (best, fastest3, number(task3_t[fastest3]), number(task3_t[1]),
              number(task3_t[20]), number(task3_t[1] / task3_t[20])))
        + centered_plot(task3_t, r"Threads ($t$)",
                        r"Thread sweep: $n=10^6$, selected threshold $ts=%d$." % best,
                        height="5.2cm")
        + "\\end{document}\n"
    )
    return preamble + provenance + section1 + section2 + section3


def build_sources(metadata, curves, best):
    sources = {"assignment3": report(metadata, curves, best)}
    for name in FILENAMES[1:]:
        sources[name] = (
            "\\documentclass[border=6pt]{standalone}\n"
            "\\usepackage{times}\n\\usepackage{pgfplots}\n"
            "\\pgfplotsset{compat=1.18}\n\\begin{document}\n"
            + plot(curves[name], r"Threshold ($ts$)" if name == "task3_ts" else r"Threads ($t$)",
                   log_x=name == "task3_ts", height="7.2cm", width="13cm")
            + "\\end{document}\n"
        )
    return sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", nargs="?", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compile", action="store_true", help="Also export all five PDFs with pdflatex")
    parser.add_argument("--pdflatex", default="pdflatex", help="pdflatex executable for --compile")
    args = parser.parse_args()
    try:
        metadata, curves, best = read_benchmarks(args.input_dir)
        sources = build_sources(metadata, curves, best)
        executable = shutil.which(args.pdflatex) if args.compile else None
        if args.compile and not executable:
            raise ValueError("pdflatex was not found; sources can be generated without --compile")
        # Validate and, if requested, compile everything before publishing files.
        with tempfile.TemporaryDirectory(prefix="hw03-report-") as temporary:
            staging = Path(temporary)
            for name, source in sources.items():
                (staging / (name + ".tex")).write_text(source)
            if args.compile:
                for name in FILENAMES:
                    result = subprocess.run(
                        [executable, "-interaction=nonstopmode", "-halt-on-error", "-no-shell-escape", name + ".tex"],
                        cwd=staging, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    )
                    if result.returncode or not (staging / (name + ".pdf")).is_file():
                        raise ValueError("LaTeX failed for %s:\n%s" % (name, result.stdout[-6000:]))
            args.output_dir.mkdir(parents=True, exist_ok=True)
            extensions = [".tex", ".pdf"] if args.compile else [".tex"]
            for name in FILENAMES:
                for extension in extensions:
                    shutil.copy2(staging / (name + extension), args.output_dir / (name + extension))
        print("Validated %d benchmark runs; selected threshold %d." % (70 * metadata["repeats"], best))
        print("Wrote five standalone LaTeX sources" + (" and five PDFs" if args.compile else "")
              + " to " + str(args.output_dir.resolve()))
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(1, "Report generation stopped: %s\n" % error)


if __name__ == "__main__":
    main()
