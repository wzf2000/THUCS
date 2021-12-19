#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <sys/time.h>

#include "worker.h"

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int nprocs, rank;
  CHKERR(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));
  CHKERR(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  if (argc != 3) {
    if (!rank)
      printf("Usage: ./odd_even_sort <number_count> <input_file>\n");
    MPI_Finalize();
    return 1;
  }
  const int n = atoi(argv[1]);
  const char *input_name = argv[2];

  if (n < nprocs)
  // if (nprocs == 1 || n < nprocs)
  {
    MPI_Finalize();
    return 0;
  }

  Worker *worker = new Worker(n, nprocs, rank);
  /** Read input data from the input file */
  worker->input(input_name);

  /** Sort the list (input data) */
  timeval start, end;
  unsigned long time;
  MPI_Barrier(MPI_COMM_WORLD);
  gettimeofday(&start, NULL);

  // run your code
  worker->sort();

  MPI_Barrier(MPI_COMM_WORLD);
  gettimeofday(&end, NULL);
  time = 1000000.0 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;

  /** Check the sorted list */
  int ret = worker->check();
  if (ret > 0) {
    printf("Rank %d: pass\n", rank);
  } else {
    printf("Rank %d: failed\n", rank);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    printf("Execution time of function sort is %lf ms.\n", time / 1000.0);
  }

#ifndef NDEBUG
  printf("Process %d: finalize\n", rank);
#endif
  MPI_Finalize();
  return 0;
}
