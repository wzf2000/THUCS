#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>

#include "worker.h"

static void mergeSort(float *a, size_t len1, float *b, size_t len2, float *c) {
    size_t i = 0, j = 0, k = 0;
    while (i < len1 && j < len2) {
        if (a[i] < b[j]) c[k++] = a[i++];
        else c[k++] = b[j++];
    }
    while (i < len1) c[k++] = a[i++];
    while (j < len2) c[k++] = b[j++];
    for (i = j = k = 0; i < len1; ++i, ++k)
        a[i] = c[k];
    for (; j < len2; ++j, ++k)
        b[j] = c[k];
}

void Worker::sort() {
    /** Your code ... */
    // you can use variables in class Worker: n, nprocs, rank, block_len, data
#define CHECK_BEFORE 10
#define IF_START_SORT 11
#define SEND_LEN 12
#define SEND_DATA 13
#define SEND_SORTED 14
#define SEND_FLAG 15
    std::sort(data, data + block_len);
    bool swapped[2] = { true, true };
    bool isEven = true;
    char flag, last;
    size_t nextBlockLen = 0;
    if (rank) {
        MPI_Send(&block_len, 1, MPI_UNSIGNED_LONG, rank - 1, SEND_LEN, MPI_COMM_WORLD);
    }
    if (!last_rank) {
        MPI_Recv(&nextBlockLen, 1, MPI_UNSIGNED_LONG, rank + 1, SEND_LEN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    float *nextData = new float[nextBlockLen], *mergeData = new float[block_len + nextBlockLen];
    MPI_Request request;
    while (swapped[0] || swapped[1]) {
        bool isFirst = isEven ^ (rank & 1);
        if (isFirst) {
            if (!last_rank) {
                // First
                float secondMin;
                MPI_Recv(&secondMin, 1, MPI_FLOAT, rank + 1, CHECK_BEFORE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (secondMin >= data[block_len - 1]) {
                    // No need to sort
                    flag = 0;
                } else {
                    flag = 1;
                }
                MPI_Send(&flag, 1, MPI_CHAR, rank + 1, IF_START_SORT, MPI_COMM_WORLD);
            }
        } else {
            if (rank) {
                // Second
                MPI_Send(data, 1, MPI_FLOAT, rank - 1, CHECK_BEFORE, MPI_COMM_WORLD);
                MPI_Recv(&flag, 1, MPI_CHAR, rank - 1, IF_START_SORT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        if (flag) {
            if (isFirst && !last_rank) {
                // Receive and sort
                MPI_Recv(nextData, nextBlockLen, MPI_FLOAT, rank + 1, SEND_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                mergeSort(data, block_len, nextData, nextBlockLen, mergeData);
                MPI_Isend(nextData, nextBlockLen, MPI_FLOAT, rank + 1, SEND_DATA, MPI_COMM_WORLD, &request);
            } else if (!isFirst && rank) {
                // Send and sort
                MPI_Send(data, block_len, MPI_FLOAT, rank - 1, SEND_DATA, MPI_COMM_WORLD);
                MPI_Irecv(data, block_len, MPI_FLOAT, rank - 1, SEND_DATA, MPI_COMM_WORLD, &request);
            }
        }
        if (rank == nprocs / 2) {
            if (rank) {
                MPI_Recv(&last, 1, MPI_CHAR, rank - 1, SEND_SORTED, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                flag |= last;
            }
            if (!last_rank) {
                MPI_Recv(&last, 1, MPI_CHAR, rank + 1, SEND_SORTED, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                flag |= last;
            }
            if (rank) {
                MPI_Send(&flag, 1, MPI_CHAR, rank - 1, SEND_FLAG, MPI_COMM_WORLD);
            }
            if (!last_rank) {
                MPI_Send(&flag, 1, MPI_CHAR, rank + 1, SEND_FLAG, MPI_COMM_WORLD);
            }
        } else if (rank < nprocs / 2) {
            if (rank) {
                MPI_Recv(&last, 1, MPI_CHAR, rank - 1, SEND_SORTED, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                flag |= last;
            }
            MPI_Send(&flag, 1, MPI_CHAR, rank + 1, SEND_SORTED, MPI_COMM_WORLD);
            MPI_Recv(&flag, 1, MPI_CHAR, rank + 1, SEND_FLAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (rank) {
                MPI_Send(&flag, 1, MPI_CHAR, rank - 1, SEND_FLAG, MPI_COMM_WORLD);
            }
        }
        else {
            if (!last_rank) {
                MPI_Recv(&last, 1, MPI_CHAR, rank + 1, SEND_SORTED, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                flag |= last;
            }
            MPI_Send(&flag, 1, MPI_CHAR, rank - 1, SEND_SORTED, MPI_COMM_WORLD);
            MPI_Recv(&flag, 1, MPI_CHAR, rank - 1, SEND_FLAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (!last_rank) {
                MPI_Send(&flag, 1, MPI_CHAR, rank + 1, SEND_FLAG, MPI_COMM_WORLD);
            }
        }
        if ((isFirst && !last_rank) || (!isFirst && rank)) {
            MPI_Wait(&request, MPI_STATUS_IGNORE);
        }
        swapped[isEven] = flag;
        isEven ^= 1;
    }
    delete [] nextData;
    delete [] mergeData;
#undef CHECK_BEFORE
#undef IF_START_SORT
#undef SEND_LEN
#undef SEND_DATA
#undef SEND_SORTED
#undef SEND_FLAG
}
