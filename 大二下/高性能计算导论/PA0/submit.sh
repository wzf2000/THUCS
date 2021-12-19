#!/bin/bash
set -e

source /home/spack/spack/share/spack/setup-env.sh

spack load openmpi

make -j 4

# run openmp_pow
OMP_NUM_THREADS=1  srun -N 1 ./openmp_pow 112000 100000 0
OMP_NUM_THREADS=7  srun -N 1 ./openmp_pow 112000 100000 0
OMP_NUM_THREADS=14 srun -N 1 ./openmp_pow 112000 100000 0
OMP_NUM_THREADS=28 srun -N 1 ./openmp_pow 112000 100000 0

# run mpi_pow
srun -N 1 -n 1   --cpu-bind sockets ./mpi_pow 112000 100000 0
srun -N 1 -n 7   --cpu-bind sockets ./mpi_pow 112000 100000 0
srun -N 1 -n 14  --cpu-bind sockets ./mpi_pow 112000 100000 0
srun -N 1 -n 28  --cpu-bind sockets ./mpi_pow 112000 100000 0
srun -N 2 -n 56  --cpu-bind sockets ./mpi_pow 112000 100000 0
srun -N 4 -n 112 --cpu-bind sockets ./mpi_pow 112000 100000 0

echo All done!

