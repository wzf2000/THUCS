#ifndef __BFS_COMMON_H__
#define __BFS_COMMON_H__

#include "graph.h"

struct solution {
  int *distances;
};

struct vertex_set {
  // # of vertices in the set
  int count;
  // max size of buffer vertices
  int max_vertices;
  // array of vertex ids in set
  int *vertices;
};

void bfs_serial(Graph graph, solution *sol);
#endif
