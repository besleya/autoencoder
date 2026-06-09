#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <singlet/pileup/pz_reader.h>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: version <path.1pz>\n";
        return 1;
    }

    try {
        std::string path = argv[1];
        std::ifstream fin(path, std::ios::binary | std::ios::ate);
        if (!fin) {
            throw std::runtime_error("cannot open " + path);
        }

        std::streamsize sz = fin.tellg();
        if (sz < static_cast<std::streamsize>(sizeof(singlet::pz::PZHeader))) {
            throw std::runtime_error("file too short");
        }

        fin.seekg(0, std::ios::beg);
        singlet::pz::PZHeader hdr;
        fin.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
        fin.close();

        std::cout << static_cast<int>(hdr.version) << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
