#pragma once

#include <glm/glm.hpp>

// class TransferFunction {
// public:
//     TransferFunction() {}

//     __host__ __device__ virtual static glm::vec4 getColor(float value) = 0;
// };

// class GrayScaleTransferFunction : public TransferFunction {
// public:
//     GrayScaleTransferFunction() : TransferFunction() {}

//     __host__ __device__ static glm::vec4 getColor(float value) {
//         return glm::vec4(value, value, value, value);
//     }
// };

__host__ __device__ glm::vec4 getColor(float value) {
    return glm::vec4(value, value, value, value);
}