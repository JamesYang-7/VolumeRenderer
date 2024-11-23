#pragma once

#include <glm/glm.hpp>

class TransferFunction {
public:
    __host__ __device__ TransferFunction() {}

    __host__ __device__ virtual glm::vec4 getColor(float value) const = 0;
};

class GrayScaleTransferFunction : public TransferFunction {
public:
    __host__ __device__ GrayScaleTransferFunction() : TransferFunction() {}

    __host__ __device__ glm::vec4 getColor(float value) const {
        return glm::vec4(value, value, value, value);
    }
};

class SkullTransferFunction : public TransferFunction {
public:
    __host__ __device__ SkullTransferFunction() : TransferFunction() {}

    __host__ __device__ glm::vec4 getColor(float value) const {
        if (value < 0.03) {
            return glm::vec4(0.0, 0.0, 0.0, 0.0);
        } else if (value < 1.0) {
            return glm::vec4(1.0, 1.0, 1.0, 0.8);
        } else {
            return glm::vec4(0.0, 0.0, 0.0, 0.0);
        }
    }
};

__host__ __device__ glm::vec4 getColor(float value) {
    return glm::vec4(value, value, value, value);
}
