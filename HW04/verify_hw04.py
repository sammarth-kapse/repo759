#!/usr/bin/env python3
"""Compile and check HW04 against independent force and trajectory references."""
import argparse
import ast
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess

HARNESS = r'''
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
#include <omp.h>
void serial_getAcc(const double[][3], const double[], double[][3], int);
void getAcc(const double[][3], const double[], double[][3], int);
using Vec3 = double[3];
int main(int argc, char** argv) {
    if (argc == 2) {
        const int n = std::stoi(argv[1]);
        std::mt19937 gen(17);
        std::uniform_real_distribution<double> uniform(0.0,1.0);
        std::normal_distribution<double> normal(0.0,1.0);
        std::cout << std::setprecision(17) << '[';
        for(int i=0;i<n;++i) {
            const double m=uniform(gen);
            double p[3],v[3];
            for(int k=0;k<3;++k) p[k]=normal(gen);
            for(int k=0;k<3;++k) v[k]=normal(gen);
            if(i) std::cout << ',';
            std::cout << '[' << m;
            for(double x:p) std::cout << ',' << x;
            for(double x:v) std::cout << ',' << x;
            std::cout << ']';
        }
        std::cout << "]\n";
        return 0;
    }
    int cases=0;
    for(int n:{1,2,3,7,17,65}) {
        auto p=new double[n][3];
        auto a=new double[n][3];
        std::vector<double> m(n);
        std::vector<std::array<long double,3>> expected(n);
        for(int i=0;i<n;++i) {
            m[i]=0.25+static_cast<double>(i%7)/3.0;
            p[i][0]=std::sin(i*0.7);
            p[i][1]=std::cos(i*0.4);
            p[i][2]=(i%5)*0.25;
        }
        if(n>=3) for(int k=0;k<3;++k) p[2][k]=p[1][k];
        for(int i=0;i<n;++i) for(int j=0;j<n;++j) if(i!=j) {
            long double d[3],r2=0.01L;
            for(int k=0;k<3;++k) {
                d[k]=static_cast<long double>(p[j][k])-p[i][k];
                r2+=d[k]*d[k];
            }
            const long double weight=m[j]*std::pow(r2,-1.5L);
            for(int k=0;k<3;++k) expected[i][k]+=weight*d[k];
        }
        auto check=[&](bool parallel) {
            for(int i=0;i<n;++i) for(int k=0;k<3;++k) a[i][k]=987.0;
            if(parallel) getAcc(p,m.data(),a,n);
            else serial_getAcc(p,m.data(),a,n);
            for(int i=0;i<n;++i) for(int k=0;k<3;++k) {
                const long double error=std::abs(a[i][k]-expected[i][k]);
                if(!std::isfinite(a[i][k]) ||
                   error>2e-11L*(1.0L+std::abs(expected[i][k]))) {
                    throw std::runtime_error("Acceleration differs from independent oracle");
                }
            }
            for(int k=0;k<3;++k) {
                long double force=0.0L,scale=1.0L;
                for(int i=0;i<n;++i) {force+=m[i]*a[i][k];scale+=std::abs(m[i]*a[i][k]);}
                if(std::abs(force)>1e-11L*scale) throw std::runtime_error("Net internal force is nonzero");
            }
            ++cases;
        };
        check(false);
        for(omp_sched_t schedule:{omp_sched_static,omp_sched_dynamic,omp_sched_guided}) {
            omp_set_schedule(schedule,schedule==omp_sched_static ? 0 : 1);
            for(int threads:{1,2,8}) {
                omp_set_dynamic(0);
                omp_set_num_threads(threads);
                check(true);
            }
        }
        delete[] p;delete[] a;
    }
    std::cout << cases << " acceleration checks passed\n";
}
'''


def execute(command, **kwargs):
    return subprocess.check_output([str(x) for x in command], text=True, **kwargs)


def acceleration(pos, mass):
    # Independent direct Python calculation, following the supplied nbody.py.
    result = [[0.0] * 3 for _ in pos]
    for i in range(len(pos)):
        for j in range(len(pos)):
            if i == j:
                continue
            delta = [pos[j][k] - pos[i][k] for k in range(3)]
            inv_r3 = (sum(x*x for x in delta) + 0.1**2) ** -1.5
            for k in range(3):
                result[i][k] += (delta[k] * inv_r3) * mass[j]
    return result


def reference(initial, end_time):
    mass = [row[0] for row in initial]
    pos = [row[1:4] for row in initial]
    vel = [row[4:7] for row in initial]
    total = sum(mass)
    cm = [sum(mass[i]*vel[i][k] for i in range(len(pos)))/total for k in range(3)]
    for v in vel:
        for k in range(3):
            v[k] -= cm[k]
    acc = acceleration(pos, mass)
    history = []
    for _ in range(int(end_time / 0.01)):
        for i in range(len(pos)):
            for k in range(3):
                vel[i][k] += acc[i][k] * 0.01 / 2.0
                pos[i][k] = min(4.0, max(-4.0, pos[i][k] + vel[i][k] * 0.01))
        acc = acceleration(pos, mass)
        for i in range(len(pos)):
            for k in range(3):
                vel[i][k] += acc[i][k] * 0.01 / 2.0
        history.append([row[:] for row in pos])
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', default='g++')
    parser.add_argument('--source-dir', type=Path, default=Path('.'))
    parser.add_argument('--work-dir', type=Path, required=True)
    args = parser.parse_args()
    source = args.source_dir.resolve()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    flags = ['-Wall', '-O3', '-std=c++17']
    for task in (2, 3):
        extra = ['-fopenmp'] if task == 3 else []
        for trace in (False, True):
            output = work / ('task%d%s' % (task, '_trace' if trace else ''))
            execute([args.compiler, source/('task%d.cpp' % task)] + flags + extra +
                    (['-DNBODY_SAVE_POSITIONS'] if trace else []) + ['-o', output])
    (work/'harness.cpp').write_text(HARNESS)
    execute([args.compiler, source/'task2.cpp'] + flags + ['-c', '-Dmain=serial_main',
            '-DgetAcc=serial_getAcc', '-DsavePositionsToCSV=serial_savePositionsToCSV',
            '-o', work/'serial.o'])
    execute([args.compiler, source/'task3.cpp'] + flags + ['-fopenmp', '-c',
            '-Dmain=parallel_main', '-o', work/'parallel.o'])
    execute([args.compiler, work/'harness.cpp', work/'serial.o', work/'parallel.o'] +
            flags + ['-fopenmp', '-o', work/'harness'])
    force_output = execute([work/'harness'])
    print(force_output.strip(), flush=True)
    trajectory_checks = 0
    max_error = 0.0
    for n, end_time in [(1, 0.05), (2, 0.03), (9, 0.17), (33, 0.07), (65, 0.25)]:
        initial = json.loads(execute([work/'harness', n]))
        expected = reference(initial, end_time)
        for task, schedule, threads in [(2, 'serial', 1)] + [
            (3, schedule, threads) for schedule in ('static', 'dynamic', 'guided')
            for threads in (1, 2, 8)
        ]:
            env = os.environ.copy()
            env['OMP_SCHEDULE'] = schedule if task == 3 else 'static'
            command = [work/('task%d_trace' % task), n, end_time]
            if task == 3:
                command.append(threads)
            output = execute(command, env=env, cwd=work)
            if not re.fullmatch(r'time: [0-9]+\.[0-9]+ms\n', output):
                raise RuntimeError('Unexpected program output: ' + repr(output))
            lines = (work/'positions.csv').read_text().splitlines()
            actual = [ast.literal_eval(line.split(',', 1)[1]) for line in lines]
            if len(actual) != len(expected):
                raise RuntimeError('Wrong timestep count')
            for step, truth in zip(actual, expected):
                for got, want in zip(step, truth):
                    for x, y in zip(got, want):
                        error = abs(x-y)
                        max_error = max(max_error, error)
                        if not math.isfinite(x) or not -4 <= x <= 4 or error > 2e-9*(1+abs(y)):
                            raise RuntimeError('Trajectory differs from independent Python reference')
            trajectory_checks += 1
    invalid_checks = 0
    for task, invalid_args in [
        (2, []), (2, ['0', '1']), (2, ['-2', '1']), (2, ['2x', '1']),
        (2, ['2', 'nan']), (2, ['2', 'inf']), (2, ['2', '-1']),
        (3, ['2', '1', '0']), (3, ['2', '1', '9']), (3, ['2', '1', 'x'])
    ]:
        result = subprocess.run([str(work/('task%d' % task))] + invalid_args,
                                cwd=work, text=True, capture_output=True)
        if result.returncode == 0:
            raise RuntimeError('Invalid arguments were accepted')
        invalid_checks += 1
    (work/'positions.csv').unlink(missing_ok=True)
    for task in (2, 3):
        command = [work/('task%d' % task), '9', '0.03'] + (['2'] if task == 3 else [])
        execute(command, cwd=work)
        if (work/'positions.csv').exists():
            raise RuntimeError('Normal build unexpectedly wrote positions.csv')
    results = {
        'status': 'passed', 'hostname': socket.getfqdn(),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'compiler': execute([args.compiler, '--version']).splitlines()[0],
        'acceleration_checks': int(force_output.split()[0]),
        'trajectory_checks': trajectory_checks, 'invalid_argument_checks': invalid_checks,
        'max_position_error': max_error, 'normal_build_csv_disabled': True,
    }
    (work/'validation.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
