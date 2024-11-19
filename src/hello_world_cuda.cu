#include <stdio.h>

__global__ void hello_world_cuda() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello world from thread %d!\n", idx);
}

int main() {
  hello_world_cuda<<<8, 1>>>();
  cudaDeviceSynchronize();
  return 0;
}