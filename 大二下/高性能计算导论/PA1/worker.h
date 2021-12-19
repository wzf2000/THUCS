#ifndef WORKER_H
#define WORKER_H

#ifndef DEBUG
#define NDEBUG
#endif

#include <mpi.h>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>

#ifndef NDEBUG
#define assertSuccess(err)                                                    \
    {                                                                         \
        if (err != MPI_SUCCESS) {                                             \
            char errStr[100];                                                 \
            int strLen;                                                       \
            MPI_Error_string(err, errStr, &strLen);                           \
            printf("Err 0x%X in line %d : %s\n", int(err), __LINE__, errStr); \
            abort();                                                          \
        }                                                                     \
    }
#else
#define assertSuccess(err)
#endif
#define CHKERR(func)             \
    {                            \
        int _errCode = (func);   \
        assertSuccess(_errCode); \
    }

template <typename T, typename P>
inline T ceiling(T x, P y) {
    return (x + y - 1) / y;
}

class Worker {
    // you may use the following variables
   private:
    int nprocs, rank;
    size_t n, block_len;
    float *data;
    bool last_rank, out_of_range;

    // you need to implement this function
   public:
    void sort();

    // don't touch the following variables & functions
   private:
    size_t IO_offset;
    MPI_File in_file, out_file;

   public:
    Worker(size_t _n, int _nprocs, int _rank) : nprocs(_nprocs), rank(_rank), n(_n) {
        size_t block_size = ceiling(n, nprocs);
        IO_offset = block_size * rank;
        out_of_range = IO_offset >= n;
        last_rank = IO_offset + block_size >= n;
        if (!out_of_range) {
            block_len = std::min(block_size, n - IO_offset);
            data = new float[block_len];
        } else {
            block_len = 0;
            data = nullptr;
        }
    }

    ~Worker() {
        if (!out_of_range) {
            delete[] data;
        }
    }

    void input(const char *input_name);
    int check();
};

#endif  // WORKER_H
