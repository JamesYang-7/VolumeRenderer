/**
 * This program reads from http://graphics.stanford.edu/data/voldata/ dataset and writes to a binary file.
 * header: 3 uint16_t values representing the dimensions of the volume data
 * data: 16-bit unsigned integers representing the volume data
 */

#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>


int main() {
    const std::string outFilePath = "MRbrain.bin";

    std::ofstream outFile(outFilePath, std::ios::binary);
    if (!outFile) {
        throw std::runtime_error("Failed to open output file: " + outFilePath);
    }
    int x = 256, y = 256, z = 109;
    uint16_t header[3] = {static_cast<uint16_t>(x), static_cast<uint16_t>(y), static_cast<uint16_t>(z)};
    for (int i = 0; i < 3; ++i) {
        header[i] = (header[i] << 8) | (header[i] >> 8); // swap endian
    }

    // write header
    if (!outFile.write(reinterpret_cast<const char*>(header), sizeof(header))) {
        throw std::runtime_error("Failed to write to output file: " + outFilePath);
    }

    for (int i = 1; i <= z; ++i) {
        std::string filePath = "../../../data/MRbrain/MRbrain." + std::to_string(i);
        std::ifstream file(filePath, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open file: " + filePath);
        }

        file.seekg(0, std::ios::end);
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        if (size % sizeof(uint16_t) != 0) {
            throw std::runtime_error("File size is not a multiple of 16-bit integers");
        }
        std::vector<uint16_t> buffer(size / sizeof(uint16_t));
        if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
            throw std::runtime_error("Failed to read file: " + filePath);
        }
        if (!outFile.write(reinterpret_cast<const char*>(buffer.data()), size)) {
            throw std::runtime_error("Failed to write to output file: " + outFilePath);
        }
    }

    printf("Successfully wrote to %s\n", outFilePath.c_str());
}