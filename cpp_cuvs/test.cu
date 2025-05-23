// test.cu
#include <stdio.h>
__global__ void hello() { printf("Hello, CUDA!\n"); }
int main() { hello<<<1,1>>>(); cudaDeviceSynchronize(); }