#!/bin/bash

# run on 1 machine * 2 process * 14 threads with process binding, feel free to change it!

export OMP_NUM_THREADS=14
# export OMP_PROC_BIND=false
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

srun -N 4 -n 4 --cpu-bind=sockets ./bfs_omp_mpi $*

