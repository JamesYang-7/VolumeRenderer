#pragma once

#include <vector>
#include <glm/glm.hpp>
#include <iostream>

struct Voxel {
    uint32_t v[8];
};

template <typename T>
class VolumeData {
public:
    VolumeData(const T* data, glm::uvec3 data_shape, glm::vec3 voxel_ratio = glm::vec3(1.0f));
    __host__ __device__ VolumeData(const T* data, Voxel* voxels, glm::uvec3 data_shape, glm::vec3 voxel_ratio = glm::vec3(1.0f));

    __host__ __device__ T at(const glm::vec3& pos) const;

    __host__ __device__ glm::uvec3 getShape() const {
        return m_shape;
    }

    __host__ __device__ uint32_t getNum() const {
        return m_shape.x * m_shape.y * m_shape.z;
    }

    __host__ __device__ const T* getData() const {
        return data;
    }

    __host__ __device__ const Voxel* getVoxels() const {
        return voxels;
    }

    __host__ __device__ void setData(const T* data) {
        this->data = data;
    }

    __host__ __device__ void setVoxels(Voxel* voxels) {
        this->voxels = voxels;
    }

    void print();

private:
    const T* data;
    Voxel* voxels;
    glm::uvec3 m_data_shape;
    glm::uvec3 m_shape;
    glm::vec3 m_voxel_size;
    glm::vec3 m_voxel_ratio;
};

template <typename T>
__host__ __device__ T trilinear_interpolation(const T* data, const Voxel& voxel, const glm::vec3& pos) {
    float x = pos.x;
    float y = pos.y;
    float z = pos.z;
    const uint32_t* v = voxel.v;
    
    T c00 = data[v[0]] * (1 - x) + data[v[1]] * x;
    T c10 = data[v[2]] * (1 - x) + data[v[3]] * x;
    T c01 = data[v[4]] * (1 - x) + data[v[5]] * x;
    T c11 = data[v[6]] * (1 - x) + data[v[7]] * x;
    T c0 = c00 * (1 - y) + c10 * y;
    T c1 = c01 * (1 - y) + c11 * y;
    return c0 * (1 - z) + c1 * z;
}