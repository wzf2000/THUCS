#include "bfs_common.h"
#include "graph.h"
#include <cstdio>
#include <sys/time.h>
#include <mpi.h>
#include <omp.h>

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void my_vertex_set_clear(vertex_set *list) { list->count = 0; }

void my_vertex_set_init(vertex_set *list, int count) {
  list->max_vertices = count;
  list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
  my_vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void my_top_down_step(Graph g, vertex_set *frontier, vertex_set *new_frontier,
                      int *distances, int rank, int beta) {
  int distance = distances[frontier->vertices[0]];
  
  if (rank == 0) {
    for (int i = 0; i < frontier->count; i++) {

      int node = frontier->vertices[i];

      int start_edge = g->outgoing_starts[node];
      int end_edge = (node == g->num_nodes - 1) ? g->num_edges
                                                : g->outgoing_starts[node + 1];

      // attempt to add all neighbors to the new frontier
      for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
        int outgoing = g->outgoing_edges[neighbor];
        if (distances[outgoing] == NOT_VISITED_MARKER) {
          distances[outgoing] = distances[node] + 1;
          int index = new_frontier->count++;
          new_frontier->vertices[index] = outgoing;
        }
      }
    }
  }
  MPI_Bcast(&new_frontier->count, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (new_frontier->count <= 1. * num_nodes(g) / beta) {
    MPI_Bcast(new_frontier->vertices, new_frontier->count, MPI_INT, 0, MPI_COMM_WORLD);
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < new_frontier->count; ++i)
      distances[new_frontier->vertices[i]] = distance + 1;
  } else {
    MPI_Bcast(new_frontier->vertices, new_frontier->count, MPI_INT, 0, MPI_COMM_WORLD);
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < new_frontier->count; ++i)
      distances[new_frontier->vertices[i]] = distance + 1;
    // MPI_Bcast(distances, num_nodes(g), MPI_INT, 0, MPI_COMM_WORLD);
  }
}

void my_bottom_up_step(Graph g, vertex_set *new_frontier, vertex_set *tmp_frontier,
                       int *distances, int num_threads, vertex_set *list,
                       int rank, int nprocs, int *count,
                       int *disp, double beta, int distance) {
  // timeval start, end;
  // gettimeofday(&start, NULL);

  int blockSize = num_nodes(g) / nprocs;
  if (num_nodes(g) % nprocs) ++blockSize;
  int l = std::min(num_nodes(g), blockSize * rank), r = std::min(num_nodes(g) - 1, blockSize * (rank + 1));
  #pragma omp parallel for schedule(guided)
  for (int i = 0; i < num_threads; ++i) {
    my_vertex_set_clear(list + i);
  }
  #pragma omp parallel for schedule(guided)
  for (int i = l; i < r; ++i) {
    if (distances[i] !=  NOT_VISITED_MARKER) continue;

    for (int *v = g->incoming_edges + g->incoming_starts[i]; v < g->incoming_edges + g->incoming_starts[i + 1]; ++v) {
      if (distances[*v] == distance) {
        int id = omp_get_thread_num();
        distances[i] = distance + 1;
        list[id].vertices[list[id].count++] = i;
        break;
      }
    }
  }
  if ((num_nodes(g) - 1) / blockSize == rank && distances[num_nodes(g) - 1] == NOT_VISITED_MARKER) {
    int i = num_nodes(g) - 1;
    for (int v = g->incoming_starts[i]; v < g->num_edges; ++v) {
      if (distances[g->incoming_edges[v]] == distance) {
        distances[i] = distance + 1;
        list[0].vertices[list[0].count++] = i;
        break;
      }
    }
  }
  int sum = 0;
  #pragma omp parallel for schedule(guided) reduction(+:sum)
  for (int i = 0; i < num_threads; ++i)
    sum += list[i].count;
  // gettimeofday(&end, NULL);
  // printf("Full Rank %d: %lf ms.\n", 
  //         rank, (1000000.0 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec) / 1000.0);
  MPI_Allreduce(&sum, &sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (sum > 1. * num_nodes(g) / beta) {
    new_frontier->count = sum;
    MPI_Allgather(distances + l, blockSize, MPI_INT, distances, blockSize, MPI_INT, MPI_COMM_WORLD);
    return;
  }
  my_vertex_set_clear(tmp_frontier);
  for (int i = 0; i < num_threads; ++i) {
    for (int j = 0; j < list[i].count; ++j) {
      tmp_frontier->vertices[tmp_frontier->count++] = list[i].vertices[j];
    }
  }
  MPI_Allgather(&tmp_frontier->count, 1, MPI_INT, count, 1, MPI_INT, MPI_COMM_WORLD);
  for (int i = 0; i < nprocs; ++i) {
    if (i) disp[i] = disp[i - 1] + count[i - 1];
    else disp[i] = 0;
  }
  MPI_Allgatherv(tmp_frontier->vertices, tmp_frontier->count, MPI_INT, new_frontier->vertices, count, disp, MPI_INT, MPI_COMM_WORLD);
  new_frontier->count = disp[nprocs - 1] + count[nprocs - 1];
  #pragma omp parallel for schedule(guided)
  for (int i = 0; i < new_frontier->count; ++i)
    distances[new_frontier->vertices[i]] = distance + 1;
}

void bfs_omp_mpi(Graph graph, solution *sol) {
  /** Your code ... */
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  int blockSize = num_nodes(graph) / nprocs;
  if (num_nodes(graph) % nprocs) ++blockSize;

  int *count = new int[nprocs];
  int *disp = new int[nprocs];
  int num_threads;
  #pragma omp parallel
  {
    #pragma omp master
    num_threads = omp_get_num_threads();
  }
  double beta = 120;
  vertex_set list1;
  vertex_set list2;
  vertex_set list3;
  vertex_set *list = new vertex_set[num_threads];
  my_vertex_set_init(&list1, graph->num_nodes);
  my_vertex_set_init(&list2, graph->num_nodes);
  my_vertex_set_init(&list3, blockSize);
  #pragma omp parallel for schedule(guided)
  for (int i = 0; i < num_threads; ++i)
    my_vertex_set_init(list + i, graph->num_nodes);

  vertex_set *frontier = &list1;
  vertex_set *new_frontier = &list2;
  vertex_set *tmp_frontier = &list3;

  int *my_distances = new int[nprocs * blockSize];
  // initialize all nodes to NOT_VISITED
  #pragma omp parallel for schedule(guided)
  for (int i = 0; i < graph->num_nodes; i++)
    my_distances[i] = NOT_VISITED_MARKER;

  // setup frontier with the root node
  frontier->vertices[frontier->count++] = ROOT_NODE_ID;
  my_distances[ROOT_NODE_ID] = 0;

  int distance = 0;

  // printf("Rank %d Start!!!!!\n", rank);

  while (frontier->count != 0) {
    my_vertex_set_clear(new_frontier);
    
    // timeval start, end;
    // gettimeofday(&start, NULL);

    if (frontier->count > 1. * num_nodes(graph) / beta)
      // printf("Bottom up "),
      my_bottom_up_step(graph, new_frontier, tmp_frontier,  my_distances, num_threads, list, rank, nprocs, count, disp, beta, distance);
    else
      // printf("Top down "),
      my_top_down_step(graph, frontier, new_frontier, my_distances, rank, beta);
    
    // gettimeofday(&end, NULL);
    // printf("Rank %d: %lf ms. %d\n", 
    //         rank, (1000000.0 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec) / 1000.0, new_frontier->count);

    // swap pointers
    vertex_set *tmp = frontier;
    frontier = new_frontier;
    new_frontier = tmp;
    ++distance;
  }
  #pragma omp parallel for schedule(guided)
  for (int i = 0; i < graph->num_nodes; i++)
    sol->distances[i] = my_distances[i];
  // printf("Rank %d End!!!!!\n", rank);
}
