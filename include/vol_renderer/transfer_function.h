#pragma once

#include <glm/glm.hpp>

enum class TransferFunctionType {
    GRAYSCALE,
    TF1,
    TF2
};

__host__ __device__ glm::vec4 getColor(float value, TransferFunctionType type) {
    glm::vec4 color(0.0f);
    if (type == TransferFunctionType::GRAYSCALE) {
        color = glm::vec4(value, value, value, value);
    }else if (type == TransferFunctionType::TF1) {
        color = glm::vec4(0, 0, value, value);
    } else if (type == TransferFunctionType::TF2){
        color = glm::vec4(0, value, 0, value);
    }
    return color;
}
