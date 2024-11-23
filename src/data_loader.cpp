#include "vol_renderer/data_loader.h"

uint16_t swapEndian(uint16_t value) {
    return (value << 8) | (value >> 8);
}

void DataLoader::normalize() {
    data.resize(raw_data.size());
    float min = std::numeric_limits<float>::max();
    float max = std::numeric_limits<float>::min();
    for (float value : raw_data) {
        min = std::min(min, value);
        max = std::max(max, value);
    }
    for (size_t i = 0; i < raw_data.size(); ++i) {
        data[i] = (raw_data[i] - min) / (max - min);
    }
    printf("max: %f, min: %f\n", max, min);
}

void DataLoader::loadData(const std::string& filePath) {
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

    raw_data.resize(size / sizeof(uint16_t));
    if (!file.read(reinterpret_cast<char*>(raw_data.data()), size)) {
        throw std::runtime_error("Failed to read data from file: " + filePath);
    }

    for (auto& value : raw_data) {
        value = swapEndian(value);
    }
    normalize();
}