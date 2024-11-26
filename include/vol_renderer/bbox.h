#pragma once

#include <glm/glm.hpp>
#include "vol_renderer/ray.h"

struct AABB {
    glm::vec3 min;
    glm::vec3 max;

    __host__ __device__ AABB() : min(glm::vec3(0.0f)), max(glm::vec3(0.0f)) {}
    __host__ __device__ AABB(const glm::vec3& min_t, const glm::vec3& max_t) : min(min_t), max(max_t) {}

    __host__ __device__ glm::vec3 center() const;

    __host__ __device__ glm::vec3 size() const;

    __host__ __device__ float volume() const;

    __host__ __device__ bool contains(const glm::vec3& point) const;

    __host__ __device__ bool ray_intersect(const Ray* ray, float* t_in, float* t_out) const;

    __host__ __device__ glm::vec3 getLocalPos(const glm::vec3& pos) const;
};