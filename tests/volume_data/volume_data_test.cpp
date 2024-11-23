#include "vol_renderer/volume_data.h"
#include <glm/glm.hpp>
#include <vector>
#include <iostream>
#include <stdio.h>

int main() {
    Voxel voxel;
    int x = 2;
    int y = 2;
    int z = 2;
    int num = x * y * z;

    for (int i = 0; i < 8; ++i) {
        voxel.v[i] = i;
    }

    // create a voxel, assume the value is the position
    std::vector<glm::vec3> voxel_data_vec3(num);
    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < y; ++j) {
            for (int k = 0; k < z; ++k) {
                voxel_data_vec3[i * 4 + j * 2 + k] = glm::vec3(k, j, i);
            }
        }
    }

    printf("Voxel interpolation test:\n");
    // interpolate in the voxel
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) {
                glm::vec3 p = glm::vec3(0.25f + k * 0.5f, 0.25f + j * 0.5f, 0.25f + i * 0.5f);
                glm::vec3 value = trilinear_interpolation(voxel_data_vec3.data(), voxel, p);
                printf("Value at (%.2f, %.2f, %.2f): (%.2f, %.2f, %.2f)\n", p.x, p.y, p.z, value.x, value.y, value.z);
            }
        }
    }

    x = y = z = 3;
    num = x * y * z;
    std::vector<glm::vec3> volume_data_array(num);
    for (int i = 0; i < z; ++i) {
        for (int j = 0; j < y; ++j) {
            for (int k = 0; k < x; ++k) {
                volume_data_array[i * x * y + j * x + k] = glm::vec3((float)k / (x - 1), (float)j / (y - 1), (float)i / (z - 1));
            }
        }
    }
    VolumeData<glm::vec3> volume_data(volume_data_array, glm::ivec3(x, y, z));
    std::cout << std::endl << "Volume Data Test:" << std::endl;
    volume_data.print();
    printf("Volume data created\n");
    float voxel_size = 1.0f / (x - 1);
    float half_voxel_size = voxel_size / 2.0f;
    for (int i = 0; i < z - 1; ++i) {
        for (int j = 0; j < y - 1; ++j) {
            for (int k = 0; k < x - 1; ++k) {
                glm::vec3 p = glm::vec3(half_voxel_size + k * voxel_size, half_voxel_size+ j * voxel_size, half_voxel_size + i * voxel_size);
                glm::vec3 value = volume_data.at(p);
                printf("Value at (%.2f, %.2f, %.2f): (%.2f, %.2f, %.2f)\n", p.x, p.y, p.z, value.x, value.y, value.z);
            }
        }
    }
}