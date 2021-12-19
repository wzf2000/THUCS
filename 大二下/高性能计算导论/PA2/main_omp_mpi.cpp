#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <sstream>
#include <sys/time.h>
#include <vector>

#include "bfs_common.h"
#include "graph.h"

#define USE_BINARY_GRAPH 1

void bfs_omp_mpi(Graph graph, solution *sol);

int main(int argc, char **argv) {

  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  std::string graph_filename;

  if (argc < 2) {
    if (rank == 0)
      std::cerr << "Usage: ./bfs_omp_mpi <path/to/graph/file> \n";
    MPI_Finalize();
    exit(1);
  }

  graph_filename = argv[1];

  Graph g;

  if (rank == 0) {
    printf("----------------------------------------------------------\n");
    printf("Max system threads = %d\n", omp_get_max_threads());
    printf("----------------------------------------------------------\n");

    printf("Loading graph...\n");
  }
  if (USE_BINARY_GRAPH) {
    g = load_graph_binary(graph_filename.c_str());
  } else {
    if (rank == 0) {
      g = load_graph(argv[1]);
      printf("storing binary form of graph!\n");
      store_graph_binary(graph_filename.append(".graph").c_str(), g);
    }
    MPI_Finalize();
    exit(1);
  }
  if (rank == 0) {
    printf("\n");
    printf("Graph stats:\n");
    printf("  Edges: %d\n", g->num_edges);
    printf("  Nodes: %d\n", g->num_nodes);
  }

  solution sol1;
  sol1.distances = (int *)malloc(sizeof(int) * g->num_nodes);
  solution sol2;
  sol2.distances = (int *)malloc(sizeof(int) * g->num_nodes);
  int correct = 1;
  bfs_omp_mpi(g, &sol1);
  if (rank == 0) {
    bfs_serial(g, &sol2);
    for (int j = 0; j < g->num_nodes; j++) {
      if (sol1.distances[j] != sol2.distances[j]) {
        printf("*** Results disagree at %d: %d, %d\n", j, sol1.distances[j],
               sol2.distances[j]);
        correct = 0;
        break;
      }
    }
  }
  MPI_Bcast(&correct, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (correct != 1) {
    MPI_Finalize();
    exit(1);
  }

  int repeat = 10;
  unsigned long total_time = 0.0;
  for (int i = 0; i < repeat; ++i) {
    timeval start, end;
    MPI_Barrier(MPI_COMM_WORLD);
    gettimeofday(&start, NULL);
    bfs_omp_mpi(g, &sol1);
    MPI_Barrier(MPI_COMM_WORLD);
    gettimeofday(&end, NULL);
    total_time +=
        1000000.0 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
  }
  if (rank == 0)
    printf("Average execution time of function bfs_omp_mpi is %lf ms.\n",
           total_time / 1000.0 / repeat);

  MPI_Finalize();
  return 0;
}
