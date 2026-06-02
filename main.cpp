// SPDX-License-Identifier: MIT
#include <singlet/singlet.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <numeric>



void process_file(const std::string& path) {
    try {
        auto result = singlet::pz::read_1pz(path);
        
        auto* data_ptr = result.data.data();

        bool is_gpu = false;

        double sum = 0.0;
        sum = std::accumulate(result.data.begin(), result.data.end(), 0.0);

        double mean = (result.m > 0 && result.n > 0) ? sum / (static_cast<double>(result.m) * result.n) : 0.0;
        
        std::cout << "File: " << path << (is_gpu ? " [GPU]" : " [CPU]") << "\n";
        std::cout << "Shape: " << result.m << " x " << result.n << "\n";
        std::cout << "Mean value: " << mean << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error processing " << path << ": " << e.what() << std::endl;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path-to-.1pz> [additional-paths...]" << std::endl;
        return 1;
    }

    for (int i = 1; i < argc; ++i) {
        process_file(argv[i]);
    }

    return 0;
}
