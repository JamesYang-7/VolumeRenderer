#include "vol_renderer/data_loader.h"
#include <stdio.h>

int main() {
    DataLoader loader("F:\\Code\\volume_renderer\\data\\MRbrain\\MRbrain.1");
    const std::vector<float>& data = loader.getData();
    printf("Data size: %zu\n", data.size());
    std::ofstream outFile("output.csv");
    if (outFile.is_open()) {
        for (size_t i = 0; i < data.size(); ++i) {
            outFile << data[i];
            if ((i + 1) % 256 == 0) {
                outFile << "\n";
            } else {
                outFile << ",";
            }
        }
        outFile.close();
        printf("Data written to output.csv\n");
    } else {
        printf("Failed to open output.csv\n");
    }
    return 0;
}