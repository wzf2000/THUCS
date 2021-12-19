// #define GRAINSIZE 128

// static omp_lock_t lock;
// std::recursive_mutex mut;

// struct Pennant {
//   int data;
//   Pennant *left, *right;
//   int size;
//   Pennant(int _) : data(_), left(nullptr), right(nullptr), size(1) {}
// };

// Pennant *pennant_union(Pennant *x, Pennant *y) {
//   y->right = x->left;
//   x->left = y;
//   x->size += y->size;
//   return x;
// }

// Pennant *pennant_split(Pennant *x) {
//   Pennant *y = x->left;
//   x->left = y->right;
//   y->right = nullptr;
//   y->size = (x->size /= 2);
//   return y;
// }

// class Bag {
//   Pennant **S;
//   int r = 0;
//   bool isEmpty = true;
  
//   auto FA(Pennant *x, Pennant *y, Pennant *z) -> std::tuple<Pennant*, Pennant*> {
//     if (x == nullptr && y == nullptr && z == nullptr) return std::make_tuple(nullptr, nullptr);
//     if (x != nullptr && y == nullptr && z == nullptr) return std::make_tuple(x, nullptr);
//     if (x == nullptr && y != nullptr && z == nullptr) return std::make_tuple(y, nullptr);
//     if (x == nullptr && y == nullptr && z != nullptr) return std::make_tuple(z, nullptr);
//     if (x != nullptr && y != nullptr && z == nullptr) return std::make_tuple(nullptr, pennant_union(x, y));
//     if (x != nullptr && y == nullptr && z != nullptr) return std::make_tuple(nullptr, pennant_union(x, z));
//     if (x == nullptr && y != nullptr && z != nullptr) return std::make_tuple(nullptr, pennant_union(y, z));
//     return std::make_tuple(x, pennant_union(y, z));
//   }
//   void remove(Pennant *x) {
//     if (!x) return;
//     remove(x->left);
//     remove(x->right);
//     delete x;
//   }
// public:
//   Bag(int limit) {
//     int tmp = 1;
//     while (tmp < limit) {
//       tmp <<= 1;
//       ++r;
//     }
//     S = new Pennant*[r];
//     for (int i = 0; i < r; ++i)
//       S[i] = nullptr;
//   }
//   Bag &operator=(Bag &rhs) {
//     isEmpty = rhs.isEmpty;
//     for (int i = 0; i < r; ++i)
//       remove(S[i]);
//     delete [] S;
//     S = rhs.S;
//     r = rhs.r;
//     rhs.S = nullptr;
//     return *this;
//   }
//   void insert(Pennant *x) {
//     int k = 0;
//     while (S[k]) {
//       x = pennant_union(S[k], x);
//       S[k++] = nullptr;
//     }
//     S[k] = x;
//     isEmpty = false;
//   }
//   void bag_union(const Bag &other) {
//     Pennant *y = nullptr;
//     for (int i = 0; i < r; ++i)
//       std::tie(S[i], y) = FA(S[i], other.S[i], y);
//     isEmpty &= other.isEmpty;
//   }
//   int depth() {
//     return r;
//   }
//   bool empty() {
//     return isEmpty;
//   }
//   Pennant *operator[](int index) {
//     return S[index];
//   }
//   ~Bag() {
//     if (!S) return;
//     for (int i = 0; i < r; ++i)
//       remove(S[i]);
//     delete [] S;
//   }
// };

// void dfs(Graph g, Pennant *x, Bag *out_bag, int d, int *distances) {
//   const Vertex *start = outgoing_begin(g, x->data);
//   const Vertex *end = outgoing_end(g, x->data);
//   //#pragma omp parallel for schedule(guided)
//   for (const Vertex *v = start; v < end; ++v) {
//     if (distances[*v] == NOT_VISITED_MARKER) {
//       //omp_set_lock(&lock);
//       distances[*v] = d + 1;
//       mut.lock();
//       out_bag->insert(new Pennant(*v));
//       mut.unlock();
//       //omp_unset_lock(&lock);
//     }
//   }
//   if (x->left) dfs(g, x->left, out_bag, d, distances);
//   if (x->right) dfs(g, x->right, out_bag, d, distances);
// }

// void process_pennant(Graph g, Pennant *in_pennant, Bag *out_bag, int d, int *distances) {
//   if (in_pennant->size < GRAINSIZE) {
//     dfs(g, in_pennant, out_bag, d, distances);
//     return;
//   }
//   Pennant *new_pennant = pennant_split(in_pennant);
//   //#pragma omp parallel num_threads(2)
//   {
//     //#pragma omp sections
//     {
//       //#pragma omp section
//       //{
//         auto fu = std::async(std::launch::async, process_pennant, g, new_pennant, out_bag, d, distances);
//         //process_pennant(g, new_pennant, out_bag, d, distances);
//       //}
//       //#pragma omp section
//       //{
//         process_pennant(g, in_pennant, out_bag, d, distances);
//         fu.wait();
//       //}
//     }
//   }
// }

// void process_layer(Graph g, Bag *in_bag, Bag *out_bag, int d, int *distances) {
//   for (int i = 0; i < in_bag->depth(); ++i)
//     if ((*in_bag)[i] != nullptr)
//       process_pennant(g, (*in_bag)[i], out_bag, d, distances);
// }

// void bfs_omp_mpi(Graph graph, solution *sol) {
//   int rank, nprocs;
//   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
//   omp_init_lock(&lock);
//   // initialize all nodes to NOT_VISITED
//   if (rank == 0) {
//     int num_threads = 14;
//     omp_set_num_threads(num_threads);
//     #pragma omp parallel for schedule(guided)
//     for (int i = 0; i < graph->num_nodes; i++)
//       sol->distances[i] = NOT_VISITED_MARKER;
//     Bag *last = new Bag(num_nodes(graph));
//     sol->distances[ROOT_NODE_ID] = 0;
//     last->insert(new Pennant(ROOT_NODE_ID));
//     int d = 0;
//     while (!last->empty()) {
//       Bag *now = new Bag(num_nodes(graph));
//       process_layer(graph, last, now, d, sol->distances);
//       ++d;
//       delete last;
//       last = now;
//     }
//   }
//   MPI_Bcast(sol->distances, num_nodes(graph), MPI_INT, 0, MPI_COMM_WORLD);
//   omp_destroy_lock(&lock);
// }

// int get_all(vertex_set *frontier) {
//   int ret;
//   MPI_Allreduce(&ret, &frontier->count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
//   return ret;
// }

// void bfs_omp_mpi(Graph graph, solution *sol) {
//   /** Your code ... */
//   int n = num_nodes(graph);
//   int rank, nprocs;
//   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
//   int num_threads = 14;
//   omp_set_num_threads(num_threads);
//   // int cnt = 0;
//   // double beta = 120;
//   vertex_set list1;
//   vertex_set list2;
//   vertex_set list3;
//   my_vertex_set_init(&list1, n);
//   my_vertex_set_init(&list2, n);
//   my_vertex_set_init(&list3, n);

//   int *owner = new int [n];
//   int size = n / nprocs + (n % nprocs > 0);
//   #pragma omp parallel for schedule(guided)
//   for (int i = 0; i < n; ++i) {
//     owner[i] = i / size;
//   }

//   vertex_set *frontier = &list1;
//   vertex_set *local_frontier = &list2;
//   vertex_set *remote_frontier = &list3;

//   // initialize all nodes to NOT_VISITED
//   #pragma omp parallel for schedule(guided)
//   for (int i = 0; i < n; i++)
//     sol->distances[i] = NOT_VISITED_MARKER;
//   int l = std::min(n, size * rank), r = std::min(n, size * (rank + 1));

//   if (rank == owner[0]){
//     // setup frontier with the root node
//     frontier->vertices[frontier->count++] = ROOT_NODE_ID;
//     sol->distances[ROOT_NODE_ID] = 0;
//   }
//   int *sendTo = new int [nprocs];
//   memset(sendTo, 0, sizeof(int) * nprocs);
//   int **ifSend = new int* [nprocs];
//   #pragma omp parallel for schedule(guided)
//   for (int i = 0; i < nprocs; ++i) {
//     ifSend[i] = new int [n];
//     #pragma omp parallel for schedule(guided)
//     for (int j = 0; j < n; ++j)
//       ifSend[i][j] = 0;
//   }
//   int *recvBuf = new int [n];
//   int *visit = new int [n];
//   #pragma omp parallel for schedule(guided)
//   for (int i = 0; i < n; ++i) {
//     visit[i] = 0;
//   }
//   int distance = 1;
//   while (get_all(frontier) > 0) {
//     printf("Rank %d.\n", rank);
//     my_vertex_set_clear(local_frontier);
//     my_vertex_set_clear(remote_frontier);
//     for (int i = 0; i < frontier->count; ++i) {
//       if (owner[frontier->vertices[i]] == rank) local_frontier->vertices[local_frontier->count++] = frontier->vertices[i];
//       else remote_frontier->vertices[remote_frontier->count++] = frontier->vertices[i];
//     }
//     my_vertex_set_clear(frontier);
//     for (int i = 0; i < remote_frontier->count; ++i) {
//       sendTo[owner[remote_frontier->vertices[i]]] = 1;
//       ifSend[owner[remote_frontier->vertices[i]]][remote_frontier->vertices[i]] = sol->distances[remote_frontier->vertices[i]];
//     }
//     MPI_Request request;
//     for (int i = 0; i < nprocs; ++i) {
//       if (i == rank || !sendTo[i]) continue;
//       MPI_Isend(ifSend[i], n, MPI_INT, i, 1, MPI_COMM_WORLD, &request);
//       sendTo[i] = 0;
//     }
//     for (int i = 0; i < local_frontier->count; ++i) {
//       int v = local_frontier->vertices[i];
//       if (visit[v]) continue;
//       visit[v] = 1;
//       const Vertex *start = outgoing_begin(graph, v);
//       const Vertex *end = outgoing_end(graph, v);
//       #pragma omp parallel for schedule(guided)
//       for (const Vertex *u = start; u < end; ++u) {
//         if (sol->distances[*u] == NOT_VISITED_MARKER) continue;
//         #pragma omp critical
//         {
//           frontier->vertices[frontier->count++] = *u;
//           sol->distances[*u] = sol->distances[v] + 1;
//         }
//       }
//       MPI_Status status;
//       MPI_Recv(recvBuf, n, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
//       #pragma omp parallel for schedule(guided)
//       for (int i = l; i < r; ++i) {
//         if (!recvBuf[i] || visit[i]) continue;
//         if (sol->distances[i] != NOT_VISITED_MARKER)
//           continue;
//         sol->distances[i] = recvBuf[i];
//         visit[i] = 1;
//         const Vertex *start = outgoing_begin(graph, i);
//         const Vertex *end = outgoing_end(graph, i);
//         for (const Vertex *u = start; u < end; ++u) {
//           if (sol->distances[*u] == NOT_VISITED_MARKER) continue;
//           #pragma omp critical
//           {
//             frontier->vertices[frontier->count++] = *u;
//             sol->distances[*u] = distance;
//           }
//         }
//       }
//     }
//     ++distance;
//   }
//   int *tmp_distances = new int [nprocs * size];
//   int *now_distances = new int [size];
//   #pragma omp parallel for schedule(guided)
//   for (int i = l; i < r; ++i)
//     now_distances[i - l] = sol->distances[i];
//   MPI_Allgather(now_distances, size, MPI_INT, tmp_distances, size, MPI_INT, MPI_COMM_WORLD);
//   #pragma omp parallel for schedule(guided)
//   for (int i = 0; i < n; ++i)
//     sol->distances[i] = tmp_distances[i];
// }
