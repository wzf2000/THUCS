#include "bfs_common.h"
#include "graph.h"
#include <cstdio>
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
                      int *distances) {

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

void my_bottom_up_step(Graph g, vertex_set *new_frontier,
                       int *distances, int num_threads, vertex_set *list,
                       double beta, int distance) {
  #pragma omp parallel for schedule(guided)
  for (int i = 0; i < num_threads; ++i) {
    my_vertex_set_clear(list + i);
  }
  #pragma omp parallel for schedule(guided)
  for (int i = 0; i < num_nodes(g); ++i) {
    if (distances[i] !=  NOT_VISITED_MARKER) continue;
    int end = (i == g->num_nodes - 1) ? g->num_edges : g->incoming_starts[i + 1];
    for (int v = g->incoming_starts[i]; v < end; ++v) {
      if (distances[g->incoming_edges[v]] == distance) {
        int id = omp_get_thread_num();
        distances[i] = distance + 1;
        list[id].vertices[list[id].count++] = i;
        break;
      }
    }
  }
  int sum = 0;
  #pragma omp parallel for schedule(guided) reduction(+:sum)
  for (int i = 0; i < num_threads; ++i)
    sum += list[i].count;
  if (sum > 1. * num_nodes(g) / beta) {
    new_frontier->count = sum;
    return;
  }
  for (int i = 0; i < num_threads; ++i) {
    for (int j = 0; j < list[i].count; ++j) {
      new_frontier->vertices[new_frontier->count++] = list[i].vertices[j];
    }
  }
}

void bfs_omp(Graph graph, solution *sol) {
  /** Your code ... */
  int num_threads;
  #pragma omp parallel
  {
    #pragma omp master
    num_threads = omp_get_num_threads();
  }
  double beta = 120;
  vertex_set list1;
  vertex_set list2;
  vertex_set *list = new vertex_set[num_threads];
  my_vertex_set_init(&list1, graph->num_nodes);
  my_vertex_set_init(&list2, graph->num_nodes);
  #pragma omp parallel for schedule(guided)
  for (int i = 0; i < num_threads; ++i)
    my_vertex_set_init(list + i, graph->num_nodes);

  vertex_set *frontier = &list1;
  vertex_set *new_frontier = &list2;

  // initialize all nodes to NOT_VISITED
  #pragma omp parallel for schedule(guided)
  for (int i = 0; i < graph->num_nodes; i++)
    sol->distances[i] = NOT_VISITED_MARKER;

  // setup frontier with the root node
  frontier->vertices[frontier->count++] = ROOT_NODE_ID;
  sol->distances[ROOT_NODE_ID] = 0;

  int distance = 0;

  while (frontier->count != 0) {
    my_vertex_set_clear(new_frontier);
    if (frontier->count > 1. * num_nodes(graph) / beta)
      my_bottom_up_step(graph, new_frontier, sol->distances, num_threads, list, beta, distance);
    else
      my_top_down_step(graph, frontier, new_frontier, sol->distances);

    // swap pointers
    vertex_set *tmp = frontier;
    frontier = new_frontier;
    new_frontier = tmp;
    ++distance;
  }
}
