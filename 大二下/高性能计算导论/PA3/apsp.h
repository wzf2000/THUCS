// DO NOT MODIFY THIS FILE

#ifndef APSP_H
#define APSP_H

/**
 * All-Pairs Shortest Path (Student implementation)
 *
 * Results are updated in place
 * 
 * @param n : #nodes in the graph
 * @param graph : Graph in adjacency matrix of size n * n.
 *                graph[i * n + j] = distance from i to j.
 *                IN DEVICE MEMORY
 */
void apsp(int n, /* device */ int *graph);

/**
 * All-Pairs Shortest Path (Reference for validation)
 */
void apspRef(int n, /* device */ int *graph);

#endif // APSP_H
