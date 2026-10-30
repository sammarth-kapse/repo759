#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>

#include <omp.h>

// Constants and array layout follow the supplied HW04 skeleton.
const double G = 1.0;
const double softening = 0.1;
const double dt = 0.01;
const double board_size = 4.0;

void getAcc(const double pos[][3], const double mass[], double acc[][3], int N) {
    // Each thread owns acc[i]; the j loop and per-particle sum stay serial.
#pragma omp parallel for schedule(runtime)
    for (int i = 0; i < N; ++i) {
        double ax = 0.0;
        double ay = 0.0;
        double az = 0.0;
        for (int j = 0; j < N; ++j) {
            if (i == j) {
                continue;
            }
            const double dx = pos[j][0] - pos[i][0];
            const double dy = pos[j][1] - pos[i][1];
            const double dz = pos[j][2] - pos[i][2];
            const double r2 = dx * dx + dy * dy + dz * dz + softening * softening;
            const double inv_r3 = 1.0 / (r2 * std::sqrt(r2));
            ax += G * (dx * inv_r3) * mass[j];
            ay += G * (dy * inv_r3) * mass[j];
            az += G * (dz * inv_r3) * mass[j];
        }
        acc[i][0] = ax;
        acc[i][1] = ay;
        acc[i][2] = az;
    }
}

// Optional debugging only; no CSV file is opened in a normal benchmark build.
void savePositionsToCSV(const double pos[][3], int N, int step,
                        const std::string& filename) {
    std::ofstream file(filename, std::ios_base::app);
    if (!file) {
        throw std::runtime_error("Unable to open positions CSV");
    }
    file << std::setprecision(17) << step << ",[";
    for (int i = 0; i < N; ++i) {
        if (i != 0) {
            file << ',';
        }
        file << '[' << pos[i][0] << ',' << pos[i][1] << ',' << pos[i][2] << ']';
    }
    file << "]\n";
}

namespace {
int positive_integer(const char* text) {
    const std::string value(text);
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument("Particle and thread counts must be positive integers");
    }
    const unsigned long long parsed = std::stoull(value);
    if (parsed == 0 || parsed > static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("Integer argument is outside the supported range");
    }
    return static_cast<int>(parsed);
}

double simulation_end(const char* text) {
    std::size_t consumed = 0;
    const double value = std::stod(text, &consumed);
    if (consumed != std::string(text).size() || !std::isfinite(value) || value <= 0.0 ||
        value / dt > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("Simulation end time must be finite and positive");
    }
    return value;
}
}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <number_of_particles> <simulation_end_time> <num_threads>\n";
        return 1;
    }
    try {
        const int N = positive_integer(argv[1]);
        const double tEnd = simulation_end(argv[2]);
        const int threads = positive_integer(argv[3]);
        if (threads > 8) {
            throw std::invalid_argument("Thread count must be in [1, 8]");
        }
        omp_set_dynamic(0);
        omp_set_num_threads(threads);
        // OMP_SCHEDULE selects static, dynamic, or guided without changing the CLI.
        if (std::getenv("OMP_SCHEDULE") == nullptr) {
            omp_set_schedule(omp_sched_static, 0);
        }
        // Retain the timestep count specified by the supplied C++ skeleton.
        const int Nt = static_cast<int>(tEnd / dt);

#ifdef NBODY_SAVE_POSITIONS
        const std::string filename = "positions.csv";
        std::ofstream(filename, std::ios::trunc).close();
#endif

        const auto start = std::chrono::steady_clock::now();
        using Vec3 = double[3];
        auto mass = std::make_unique<double[]>(N);
        auto pos = std::make_unique<Vec3[]>(N);
        auto vel = std::make_unique<Vec3[]>(N);
        auto acc = std::make_unique<Vec3[]>(N);

        // A fixed seed gives every schedule/thread count the same C++ input.
        std::mt19937 generator(17);
        std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
        std::normal_distribution<double> normal_dist(0.0, 1.0);
        for (int i = 0; i < N; ++i) {
            mass[i] = uniform_dist(generator);
            for (int k = 0; k < 3; ++k) {
                pos[i][k] = normal_dist(generator);
            }
            for (int k = 0; k < 3; ++k) {
                vel[i][k] = normal_dist(generator);
            }
        }

        double velCM[3] = {0.0, 0.0, 0.0};
        double totalMass = 0.0;
        for (int i = 0; i < N; ++i) {
            totalMass += mass[i];
            for (int k = 0; k < 3; ++k) {
                velCM[k] += vel[i][k] * mass[i];
            }
        }
        for (int k = 0; k < 3; ++k) {
            velCM[k] /= totalMass;
        }
        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < 3; ++k) {
                vel[i][k] -= velCM[k];
            }
        }

        getAcc(pos.get(), mass.get(), acc.get(), N);
        for (int step = 0; step < Nt; ++step) {
            // First half kick uses acceleration at the old positions.
#pragma omp parallel for schedule(runtime)
            for (int i = 0; i < N; ++i) {
                for (int k = 0; k < 3; ++k) {
                    vel[i][k] += acc[i][k] * dt / 2.0;
                }
            }
            // Drift using the half-step velocity.
#pragma omp parallel for schedule(runtime)
            for (int i = 0; i < N; ++i) {
                for (int k = 0; k < 3; ++k) {
                    pos[i][k] += vel[i][k] * dt;
                }
            }
            // Clamp all three coordinates before computing new accelerations.
#pragma omp parallel for schedule(runtime)
            for (int i = 0; i < N; ++i) {
                for (int k = 0; k < 3; ++k) {
                    pos[i][k] = std::clamp(pos[i][k], -board_size, board_size);
                }
            }
            getAcc(pos.get(), mass.get(), acc.get(), N);
            // Second half kick uses acceleration at the new positions.
#pragma omp parallel for schedule(runtime)
            for (int i = 0; i < N; ++i) {
                for (int k = 0; k < 3; ++k) {
                    vel[i][k] += acc[i][k] * dt / 2.0;
                }
            }
#ifdef NBODY_SAVE_POSITIONS
            savePositionsToCSV(pos.get(), N, step, filename);
#endif
        }
        const auto end = std::chrono::steady_clock::now();
        const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "time: " << std::fixed << std::setprecision(6) << elapsed_ms << "ms\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
