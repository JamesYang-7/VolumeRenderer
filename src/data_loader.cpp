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
}

void DataLoader::loadData(const std::string& filePath, bool changeEndian) {
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

    size_t data_size = size / sizeof(uint16_t) - 3;
    uint16_t size_x, size_y, size_z;
    file.read(reinterpret_cast<char*>(&size_x), sizeof(uint16_t));
    file.read(reinterpret_cast<char*>(&size_y), sizeof(uint16_t));
    file.read(reinterpret_cast<char*>(&size_z), sizeof(uint16_t));

    raw_data.resize(data_size);
    if (!file.read(reinterpret_cast<char*>(raw_data.data()), size - 3 * sizeof(uint16_t))) {
        throw std::runtime_error("Failed to read data from file: " + filePath);
    }
    
    if (changeEndian) {
        size_x = swapEndian(size_x);
        size_y = swapEndian(size_y);
        size_z = swapEndian(size_z);
        for (auto& value : raw_data) {
            value = swapEndian(value);
        }
    }
    m_size.x = static_cast<uint32_t>(size_x);
    m_size.y = static_cast<uint32_t>(size_y);
    m_size.z = static_cast<uint32_t>(size_z);
    
    normalize();
}