#include <cuda.h>
#include "cuda_runtime.h"
#include <iostream>
using namespace std;

#define cudaCheckErrors(msg) \
do { \
    cudaError_t __err = cudaGetLastError(); \
    if (__err != cudaSuccess) { \
        fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg, cudaGetErrorString(__err), __FILE__, __LINE__); \
        fprintf(stderr, "*** FAILED - ABORTING\n"); \
        exit(1); \
    } \
} while (0)

__global__ void kernel(int *output, const int *input) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    output[id] = input[id];
}

void launch(int gridSizeX, int blockSizeX, int gridSizeY = 1, int blockSizeY = 1, int gridSizeZ = 1, int blockSizeZ = 1) {
    dim3 blockSize(blockSizeX, blockSizeY, blockSizeZ);
    dim3 gridSize(gridSizeX, gridSizeY, gridSizeZ);
    kernel<<<gridSize, blockSize>>>(outputPtr, inputPtr);
}

int main() {
    int N = 1024;
    int *outputPtr, *inputPtr;
    int hostInput[N]; // Assuming hostInput is defined and initialized

    cudaMalloc(&outputPtr, sizeof(int) * N);
    cudaMalloc(&inputPtr, sizeof(int) * N);
    cudaMemcpy(inputPtr, hostInput, sizeof(int) * N, cudaMemcpyHostToDevice);

    launch(4, 1024);
    cudaCheckErrors("kernel launch failed");

    launch(4, 32, 4, 32);
    cudaCheckErrors("kernel launch failed");

    launch(4, 16, 4, 16, 4, 4);
    cudaCheckErrors("kernel launch failed");

    cudaFree(outputPtr);
    cudaFree(inputPtr);
}
