#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>

class DataLoader {
public:
    DataLoader(const std::string& filePath) {
        loadData(filePath);
    }

    const std::vector<uint16_t>& getData() const {
        return data;
    }

private:
    std::vector<uint16_t> data;

    void loadData(const std::string& filePath) {
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

        data.resize(size / sizeof(uint16_t));
        if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
            throw std::runtime_error("Failed to read data from file: " + filePath);
        }
    }
};

#endif // DATA_LOADER_H