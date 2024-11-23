#include "vol_renderer/data_loader.h"
#include <stdio.h>
#include <algorithm>
#include <glm/glm.hpp>

int main() {
    DataLoader loader("../../../data/MRbrain.bin", true);
    glm::uvec3 size = loader.getSize();
    printf("Data shape: (%u, %u, %u)\n", size.x, size.y, size.z);
    printf("Data size: %u\n", size.x * size.y * size.z);
    uint16_t min = *std::min_element(loader.getRawData(), loader.getRawData() + size.x * size.y * size.z);
    uint16_t max = *std::max_element(loader.getRawData(), loader.getRawData() + size.x * size.y * size.z);
    printf("Raw data min: %u\n", min);
    printf("Raw data max: %u\n", max);
    return 0;
}