// DO NOT MODIFY THIS FILE
// YOU CAN INCLUDE THIS HEADER TO YOUR SOLUTION

#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>

#define CHK_CUDA_ERR(call) { \
    auto err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(-1); \
    } \
}

#endif // CUDA_UTILS_H
