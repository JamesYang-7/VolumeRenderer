#pragma once

#include <glm/glm.hpp>

enum class TransferFunctionType {
    GRAYSCALE,
    SKULL,
    SKIN,
    NERVE,
};

__host__ __device__ glm::vec4 getColor(float value, TransferFunctionType type) {
    glm::vec4 color(0.0f);
    if (type == TransferFunctionType::GRAYSCALE) {
        color = glm::vec4(value, value, value, value);
    }else if (type == TransferFunctionType::SKULL) {
        if (value < 0.5) {
            color = glm::vec4(0.0, 0.0, 0.0, 0.0);
        } else {
            color = glm::vec4(0.5f, 0.5f, 0.5f, value * 0.5f);
        }
    } else if (type == TransferFunctionType::SKIN){
        if (value > 0.1 && value < 0.3) {
            color = 2.0f * glm::vec4(value, value, value, value / 2.0f);
        } else {
            color = glm::vec4(0.0, 0.0, 0.0, 0.0);
        }
    } else if (type == TransferFunctionType::NERVE){
        if (value > 0.30 && value < 0.35) {
            float gray = value * 2.0f;
            color = glm::vec4(gray, gray, gray, value);
        } else {
            color = glm::vec4(0.0, 0.0, 0.0, 0.0);
        }
    }
    return color;
}
