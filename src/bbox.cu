#include "vol_renderer/bbox.h"
#include <tiny-cuda-nn/common_host.h>

__host__ __device__ glm::vec3 AABB::center() const {
    return (min + max) * 0.5f;
}

__host__ __device__ glm::vec3 AABB::size() const {
    return max - min;
}

__host__ __device__ float AABB::volume() const {
    glm::vec3 s = size();
    return s.x * s.y * s.z;
}

__host__ __device__ bool AABB::contains(const glm::vec3& point) const {
    return point.x >= min.x && point.x <= max.x &&
            point.y >= min.y && point.y <= max.y &&
            point.z >= min.z && point.z <= max.z;
}

__host__ __device__ bool AABB::ray_intersect(const Ray* ray, float* t_in, float* t_out) const {
    auto pos = ray->origin;
    auto dir = ray->direction;
    float tmin = -std::numeric_limits<float>::max();
    float tmax = std::numeric_limits<float>::max();
    auto m_min = min;
    auto m_max = max;
    if (dir.x == 0.0f) {
        if (pos.x < min.x || pos.x > max.x) {
            return false;
        }
    } else {
        float txmin = (min.x - pos.x) / dir.x;
        float txmax = (max.x - pos.x) / dir.x;
        
        if (txmin > txmax) tcnn::host_device_swap(txmin, txmax);
        if (tmin > txmax || txmin > tmax) {
            return false;
        }
        tmin = txmin > tmin ? txmin : tmin;
        tmax = txmax < tmax ? txmax : tmax;
    }
    if (dir.y == 0.0f) {
        if (pos.y < min.y || pos.y > max.y) {
            return false;
        }
    } else {
        float tymin = (min.y - pos.y) / dir.y;
        float tymax = (max.y - pos.y) / dir.y;
        if (tymin > tymax) tcnn::host_device_swap(tymin, tymax);
        if (tmin > tymax || tymin > tmax) {
            return false;
        }
        tmin = tymin > tmin ? tymin : tmin;
        tmax = tymax < tmax ? tymax : tmax;
    }
    if (dir.z == 0.0f) {
        if (pos.z < min.z || pos.z > max.z) {
            return false;
        }
    } else {
        float tzmin = (min.z - pos.z) / dir.z;
        float tzmax = (max.z - pos.z) / dir.z;
        if (tzmin > tzmax) tcnn::host_device_swap(tzmin, tzmax);
        if (tmin > tzmax || tzmin > tmax) {
            return false;
        }
        tmin = tzmin > tmin ? tzmin : tmin;
        tmax = tzmax < tmax ? tzmax : tmax;
    }
    *t_in = tmin;
    *t_out = tmax;
    return true;
}

__host__ __device__ glm::vec3 AABB::getLocalPos(const glm::vec3& pos) const {
    return (pos - min) / size();
}