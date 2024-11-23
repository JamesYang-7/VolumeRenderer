#include "vol_renderer/data_loader.h"
#include <stdio.h>

int main() {
    DataLoader loader("F:\\Code\\volume_renderer\\data\\MRbrain\\MRbrain.1");
    const std::vector<uint16_t>& data = loader.getData();
    printf("Data size: %d\n", data.size());
    return 0;
}