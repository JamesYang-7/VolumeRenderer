#pragma once

#include <glm/glm.hpp>

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;

    __host__ __device__ Ray() : origin(glm::vec3(0.0f)), direction(glm::vec3(0.0f)) {}
    __host__ __device__ Ray(const glm::vec3& o, const glm::vec3& d) : origin(o) {
        direction = glm::normalize(d);
    }

    __host__ __device__ glm::vec3 at(float t) const {
        return origin + t * direction;
    }
};