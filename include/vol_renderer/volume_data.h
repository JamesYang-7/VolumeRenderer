#pragma once

#include <vector>
#include <glm/glm.hpp>
#include <iostream>
#include <stdexcept>

struct Voxel {
    int v[8];
};

template <typename T>
class VolumeData {
public:
    VolumeData(const std::vector<T>& data, glm::ivec3 data_shape, glm::vec3 voxel_ratio = glm::vec3(1.0f));

    T at(const glm::vec3& pos) const;

    void print();

private:
    const T* data;
    std::vector<Voxel> voxels;
    glm::ivec3 m_data_shape;
    glm::ivec3 m_shape;
    glm::vec3 m_voxel_size;
    glm::vec3 m_voxel_ratio;
};

template <typename T>
T trilinear_interpolation(const T* data, const Voxel& voxel, const glm::vec3& pos) {
    float x = pos.x;
    float y = pos.y;
    float z = pos.z;
    const int* v = voxel.v;
    
    T c00 = data[v[0]] * (1 - x) + data[v[1]] * x;
    T c10 = data[v[2]] * (1 - x) + data[v[3]] * x;
    T c01 = data[v[4]] * (1 - x) + data[v[5]] * x;
    T c11 = data[v[6]] * (1 - x) + data[v[7]] * x;
    T c0 = c00 * (1 - y) + c10 * y;
    T c1 = c01 * (1 - y) + c11 * y;
    return c0 * (1 - z) + c1 * z;
}