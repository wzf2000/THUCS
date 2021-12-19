// PLEASE MODIFY THIS FILE TO IMPLEMENT YOUR SOLUTION

// Brute Force APSP Implementation:

#include "apsp.h"
#include <algorithm>

#define b 32

namespace {

__global__ void kernel(int n, int k, int *graph) {
    auto i = blockIdx.y * blockDim.y + threadIdx.y;
    auto j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < n) {
        graph[i * n + j] = min(graph[i * n + j], graph[i * n + k] + graph[k * n + j]);
    }
}

__global__ void selfKernel(int n, int p, int *graph) {
    __shared__ int sharedGraph[b][b];
    const auto i = threadIdx.y;
    const auto j = threadIdx.x;
    const auto realI = b * p + i;
    const auto realJ = b * p + j;
    const auto id = realI * n + realJ;
    if (realI < n && realJ < n) {
        sharedGraph[i][j] = graph[id];
    } else {
        sharedGraph[i][j] = 1000000000;
    }
    __syncthreads();
    int newDis;
    for (int k = 0; k < b; ++k) {
        newDis = sharedGraph[i][k] + sharedGraph[k][j];
        __syncthreads();
        if (newDis < sharedGraph[i][j]) {
            sharedGraph[i][j] = newDis;
        }
        __syncthreads();
    }
    if (realI < n && realJ < n) {
        graph[id] = sharedGraph[i][j];
    }
}

__global__ void crossKernel(int n, int p, int *graph) {
    if (blockIdx.x == p) return;
    __shared__ int sharedGraph[b][b];
    const auto i = threadIdx.y;
    const auto j = threadIdx.x;
    auto realI1 = b * p + i;
    auto realJ1 = b * p + j;
    const auto id1 = realI1 * n + realJ1;
    if (realI1 < n && realJ1 < n) {
        sharedGraph[i][j] = graph[id1];
    } else {
        sharedGraph[i][j] = 1000000000;
    }
    __syncthreads();
    auto realI2 = realI1;
    auto realJ2 = realJ1;
    if (blockIdx.y == 0) {
        realI2 = b * blockIdx.x + i;
    } else {
        realJ2 = b * blockIdx.x + j;
    }
    __shared__ int sharedNewGraph[b][b];
    const auto id2 = realI2 * n + realJ2;
    int minDis;
    if (realI2 < n && realJ2 < n) {
        minDis = sharedNewGraph[i][j] = graph[id2];
    } else {
        minDis = sharedNewGraph[i][j] = 1000000000;
    }
    __syncthreads();
    int newDis;
    if (blockIdx.y == 0) {
        for (int k = 0; k < b; ++k) {
            newDis = sharedNewGraph[i][k] + sharedGraph[k][j];
            if (newDis < minDis) {
                minDis = newDis;
            }
            __syncthreads();
            sharedNewGraph[i][j] = minDis;
            __syncthreads();
        }
    } else {
        for (int k = 0; k < b; ++k) {
            newDis = sharedGraph[i][k] + sharedNewGraph[k][j];
            if (newDis < minDis) {
                minDis = newDis;
            }
            __syncthreads();
            sharedNewGraph[i][j] = minDis;
            __syncthreads();
        }

    }
    if (realI2 < n && realJ2 < n) {
        graph[id2] = sharedNewGraph[i][j];
    }
}

__global__ void globalKernel(int n, int p, int *graph) {
    if (blockIdx.x == p || blockIdx.y == p) return;
    __shared__ int sharedRowGraph[b][b], sharedColGraph[b][b];
    const auto i = threadIdx.y;
    const auto j = threadIdx.x;
    auto realI1 = b * p + i;
    auto realJ1 = b * p + j;
    const auto id1 = realI1 * n + realJ1;
    auto realI2 = b * blockIdx.y + i;
    auto realJ2 = b * blockIdx.x + j;
    const auto id2 = realI2 * n + realJ2;
    if (realI1 < n && realJ2 < n) {
        sharedRowGraph[i][j] = graph[realI1 * n + realJ2];
    } else {
        sharedRowGraph[i][j] = 1000000000;
    }
    if (realI2 < n && realJ1 < n) {
        sharedColGraph[i][j] = graph[realI2 * n + realJ1];
    } else {
        sharedColGraph[i][j] = 1000000000;
    }
    __syncthreads();
    if (realI2 < n && realJ2 < n) {
        int minDis = graph[id2], newDis;
        for (int k = 0; k < b; ++k) {
            newDis = sharedColGraph[i][k] + sharedRowGraph[k][j];
            if (newDis < minDis) {
                minDis = newDis;
            }
        }
        graph[id2] = minDis;
    }
}

}

void apsp(int n, /* device */ int *graph) {
    int N = (n - 1) / b + 1;
    dim3 thr(b, b);
    dim3 blk1(1, 1);
    dim3 blk2(N, 2);
    dim3 blk3(N, N);
    for (int p = 0; p < N; ++p) {
        selfKernel<<<blk1, thr>>>(n, p, graph);
        crossKernel<<<blk2, thr>>>(n, p, graph);
        globalKernel<<<blk3, thr>>>(n, p, graph);
    }
}

