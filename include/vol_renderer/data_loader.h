#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <cstdint>

class DataLoader {
public:
    DataLoader(const std::string& filePath) {
        loadData(filePath);
    }

    const std::vector<float>& getData() const {
        return data;
    }

    const std::vector<uint16_t>& getRawData() const {
        return raw_data;
    }

    void normalize();

private:
    std::vector<uint16_t> raw_data;
    std::vector<float> data;

    void loadData(const std::string& filePath);
};
