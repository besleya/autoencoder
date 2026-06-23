#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <omp.h>
#include <singlet/singlet.h>

namespace fs = std::filesystem;

struct BenchResult {
    std::string path;
    uint64_t file_size;
    uint32_t n_cells;
    uint64_t nnz;
    uint64_t decoded_bytes;
    double cold_ms;
    double warm_ms;
};

// Drop OS page cache for a file
void drop_cache(const std::string& path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return;
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    close(fd);
}

// Get file size
uint64_t get_file_size(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return 0;
    return st.st_size;
}

// Benchmark a single file
bool benchmark_file(const std::string& path, BenchResult& result) {
    try {
        result.path = path;
        result.file_size = get_file_size(path);

        // Cold read
        drop_cache(path);
        auto t1 = std::chrono::high_resolution_clock::now();
        singlet::pz::ReadResult r = singlet::pz::read_1pz(path);
        auto t2 = std::chrono::high_resolution_clock::now();
        result.cold_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

        // Warm read (cache populated)
        auto t3 = std::chrono::high_resolution_clock::now();
        singlet::pz::ReadResult r2 = singlet::pz::read_1pz(path);
        auto t4 = std::chrono::high_resolution_clock::now();
        result.warm_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();

        result.n_cells = r.n;
        result.nnz = r.nnz;
        result.decoded_bytes = (uint64_t)(r.n + 1) * 4 + (uint64_t)r.nnz * 8;

        return true;
    } catch (const std::exception& e) {
        std::cerr << "# ERROR: " << path << ": " << e.what() << std::endl;
        return false;
    }
}

// Get list of .1pz files from directory
std::vector<std::string> glob_1pz_files(const std::string& dir) {
    std::vector<std::string> files;
    try {
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".1pz") {
                files.push_back(entry.path().string());
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error reading directory " << dir << ": " << e.what() << std::endl;
        return files;
    }
    std::sort(files.begin(), files.end());
    return files;
}

// Read file as list of paths
std::vector<std::string> read_file_list(const std::string& file_path) {
    std::vector<std::string> files;
    std::ifstream f(file_path);
    if (!f.is_open()) {
        std::cerr << "Cannot open file list: " << file_path << std::endl;
        return files;
    }
    std::string line;
    while (std::getline(f, line)) {
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        if (!line.empty() && line[0] != '#') {
            files.push_back(line);
        }
    }
    return files;
}

// Calculate percentile
double percentile(const std::vector<double>& data, double p) {
    if (data.empty()) return 0.0;
    std::vector<double> sorted = data;
    std::sort(sorted.begin(), sorted.end());
    size_t idx = (size_t)std::ceil((p / 100.0) * sorted.size()) - 1;
    if (idx >= sorted.size()) idx = sorted.size() - 1;
    return sorted[idx];
}

int main(int argc, char* argv[]) {
    // Print build hint
    std::cerr << "# g++ -O2 -fopenmp -std=c++17 bench_read_1pz.cpp -o bench_read_1pz -lsinglet_pz -I<singlet_include>" << std::endl;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <dir_or_file_list> [n_threads]" << std::endl;
        return 1;
    }

    std::string input_path = argv[1];
    int n_threads = 1;
    if (argc > 2) {
        n_threads = std::atoi(argv[2]);
        if (n_threads < 1) n_threads = 1;
    }

    // Get list of files
    std::vector<std::string> files;
    fs::path p(input_path);

    if (fs::is_directory(p)) {
        files = glob_1pz_files(input_path);
    } else if (fs::is_regular_file(p)) {
        files = read_file_list(input_path);
    } else {
        std::cerr << "Path is neither a directory nor a regular file: " << input_path << std::endl;
        return 1;
    }

    if (files.empty()) {
        std::cerr << "No .1pz files found" << std::endl;
        return 1;
    }

    std::vector<BenchResult> results;

    // Benchmark each file
    for (const auto& file : files) {
        BenchResult result;
        if (benchmark_file(file, result)) {
            results.push_back(result);
        }
    }

    // Output CSV header and data
    std::cout << "path,file_size,n_cells,nnz,decoded_bytes,cold_ms,warm_ms" << std::endl;
    for (const auto& r : results) {
        std::cout << r.path << "," << r.file_size << "," << r.n_cells << ","
                  << r.nnz << "," << r.decoded_bytes << ","
                  << r.cold_ms << "," << r.warm_ms << std::endl;
    }

    // Summary statistics
    if (!results.empty()) {
        std::vector<double> cold_times, warm_times;
        uint64_t total_decoded = 0;
        uint64_t total_file_size = 0;

        for (const auto& r : results) {
            cold_times.push_back(r.cold_ms);
            warm_times.push_back(r.warm_ms);
            total_decoded += r.decoded_bytes;
            total_file_size += r.file_size;
        }

        double mean_cold = std::accumulate(cold_times.begin(), cold_times.end(), 0.0) / cold_times.size();
        double mean_warm = std::accumulate(warm_times.begin(), warm_times.end(), 0.0) / warm_times.size();
        double median_cold = percentile(cold_times, 50);
        double median_warm = percentile(warm_times, 50);
        double p95_cold = percentile(cold_times, 95);
        double p95_warm = percentile(warm_times, 95);
        double mean_decoded = total_decoded / (double)results.size();

        std::cout << "# SUMMARY: files=" << results.size()
                  << " mean_cold_ms=" << mean_cold
                  << " median_cold_ms=" << median_cold
                  << " p95_cold_ms=" << p95_cold
                  << " mean_warm_ms=" << mean_warm
                  << " median_warm_ms=" << median_warm
                  << " p95_warm_ms=" << p95_warm
                  << " mean_decoded_bytes=" << mean_decoded << std::endl;

        // Parallel sweep if n_threads > 1
        if (n_threads > 1) {
            omp_set_num_threads(n_threads);

            auto t_start = std::chrono::high_resolution_clock::now();

            #pragma omp parallel for
            for (size_t i = 0; i < files.size(); ++i) {
                try {
                    singlet::pz::read_1pz(files[i]);
                } catch (const std::exception& e) {
                    // Already reported in per-file benchmarks
                }
            }

            auto t_end = std::chrono::high_resolution_clock::now();
            double parallel_wall_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

            double throughput_mb_s = (total_file_size / (1024.0 * 1024.0)) / (parallel_wall_ms / 1000.0);
            double throughput_files_s = (results.size() * 1000.0) / parallel_wall_ms;

            std::cout << "# SUMMARY (parallel n_threads=" << n_threads
                      << "): wall_clock_ms=" << parallel_wall_ms
                      << " throughput_mb_s=" << throughput_mb_s
                      << " throughput_files_s=" << throughput_files_s << std::endl;
        }
    }

    return 0;
}
