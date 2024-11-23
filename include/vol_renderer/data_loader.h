#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <glm/glm.hpp>

class DataLoader {
public:
    DataLoader(const std::string& filePath, bool changeEndian = false) {
        loadData(filePath, changeEndian);
    }

    const float* getData() const {
        return data.data();
    }

    const uint16_t* getRawData() const {
        return raw_data.data();
    }

    void normalize();

    glm::uvec3 getSize() const {
        return m_size;
    }

    uint32_t getNum() const {
        return m_size.x * m_size.y * m_size.z;
    }

private:
    std::vector<uint16_t> raw_data;
    std::vector<float> data;
    glm::uvec3 m_size;

    void loadData(const std::string& filePath, bool changeEndian = false);
};
