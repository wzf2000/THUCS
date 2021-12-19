// DO NOT MODIFY THIS FILE

// Brute Force APSP Implementation:

#include "apsp.h"

namespace {

__global__ void kernel(int n, int k, int *graph) {
    auto i = blockIdx.y * blockDim.y + threadIdx.y;
    auto j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < n) {
        graph[i * n + j] = min(graph[i * n + j], graph[i * n + k] + graph[k * n + j]);
    }
}

}

void apspRef(int n, /* device */ int *graph) {
    for (int k = 0; k < n; k++) {
        dim3 thr(32, 32);
        dim3 blk((n - 1) / 32 + 1, (n - 1) / 32 + 1);
        kernel<<<blk, thr>>>(n, k, graph);
    }
}

