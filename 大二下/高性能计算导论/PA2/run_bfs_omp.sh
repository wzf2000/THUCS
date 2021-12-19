#!/bin/bash

# run with 1 thread with binding, feel free to change it!

export OMP_NUM_THREADS=28
export OMP_PLACES=cores

srun -n 1 ./bfs_omp $*

