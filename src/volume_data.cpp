#include "vol_renderer/volume_data.h"

template <typename T>
VolumeData<T>::VolumeData(const std::vector<T>& data, glm::ivec3 data_shape, glm::vec3 voxel_ratio)
    :
    m_data_shape(data_shape),
    m_voxel_ratio(voxel_ratio)
{
    this->data = data.data();
    m_shape = m_data_shape - glm::ivec3(1);
    m_voxel_size = glm::vec3(1.0f) / glm::vec3(m_shape);
    int vx = m_shape.x;
    int vy = m_shape.y;
    int vz = m_shape.z;
    int dx = m_data_shape.x;
    int dy = m_data_shape.y;
    int dz = m_data_shape.z;
    voxels.resize(vx * vy * vz);
    for (int zi = 0; zi < vz; ++zi) {
        for (int yi = 0; yi < vy; ++yi) {
            for (int xi = 0; xi < vx; ++xi) {
                Voxel& voxel = voxels[zi * vx * vy + yi * vx + xi];
                int start = xi + yi * dx + zi * dx * dy;
                voxel.v[0] = start;
                voxel.v[1] = start + 1;
                voxel.v[2] = start + dx;
                voxel.v[3] = start + dx + 1;
                start += dx * dy;
                voxel.v[4] = start;
                voxel.v[5] = start + 1;
                voxel.v[6] = start + dx;
                voxel.v[7] = start + dx + 1;
            }
        }
    }
}

template <typename T>
T VolumeData<T>::at(const glm::vec3& pos) const {
    // assume pos is in [0, 1]
    if (pos.x < 0 || pos.x > 1 || pos.y < 0 || pos.y > 1 || pos.z < 0 || pos.z > 1) {
        throw new std::out_of_range("VolumeData: Position is out of bounds");
    }
    // locate voxel
    int x = pos.x / m_voxel_size.x;
    int y = pos.y / m_voxel_size.y;
    int z = pos.z / m_voxel_size.z;
    x = std::min(x, m_shape.x - 1);
    y = std::min(y, m_shape.y - 1);
    z = std::min(z, m_shape.z - 1);
    int voxel_index = z * m_shape.x * m_shape.y + y * m_shape.x + x;
    // get local position in the voxel
    glm::vec3 local_pos = glm::vec3(
        (pos.x - x * m_voxel_size.x) / m_voxel_size.x,
        (pos.y - y * m_voxel_size.y) / m_voxel_size.y,
        (pos.z - z * m_voxel_size.z) / m_voxel_size.z
    ); 
    return trilinear_interpolation<T>(data, voxels[voxel_index], local_pos);
}

template <typename T>
void VolumeData<T>::print() {
    std::cout << "Volume Data Members:" << std::endl;
    std::cout << "Data Shape: (" << m_data_shape.x << ", " << m_data_shape.y << ", " << m_data_shape.z << ")" << std::endl;
    std::cout << "Shape: (" << m_shape.x << ", " << m_shape.y << ", " << m_shape.z << ")" << std::endl;
    std::cout << "Voxel Size: (" << m_voxel_size.x << ", " << m_voxel_size.y << ", " << m_voxel_size.z << ")" << std::endl;
    std::cout << "Voxel Ratio: (" << m_voxel_ratio.x << ", " << m_voxel_ratio.y << ", " << m_voxel_ratio.z << ")" << std::endl;
}

template class VolumeData<float>;
template class VolumeData<glm::vec3>;