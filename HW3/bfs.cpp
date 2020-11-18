#include "bfs.h"

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

int test_flag = 0;

class Vertex_list {
 public:
  unsigned capacity;
  int *vertices;
  std::atomic<unsigned> length;
  Vertex_list(size_t _capacity = 10) {
    capacity = _capacity;
    vertices = new int[capacity];
    length = 0;
  }
  ~Vertex_list() { delete[] vertices; }
  void clear_all() { length = 0; }
  int &operator[](int idx) { return vertices[idx]; }
  const int &operator[](int idx) const { return vertices[idx]; }
};

void top_dn(Graph g, Vertex_list *front, Vertex_list *front_next,
            int *distances) {
#pragma omp parallel for schedule(guided)
  for (int i = 0; i < front->length; i++) {
    int node = (*front)[i];
    int start = g->outgoing_starts[node];
    int end = (node == g->num_nodes - 1) ? g->num_edges
                                         : g->outgoing_starts[node + 1];

    for (int j = start; j < end; j++) {
      int outgoing = g->outgoing_edges[j];
      if (distances[outgoing] == NOT_VISITED_MARKER &&
          __sync_bool_compare_and_swap(&(distances[outgoing]),
                                       NOT_VISITED_MARKER,
                                       distances[node] + 1)) {
        (*front_next)[front_next->length++] = outgoing;
      }
    }
  }
}

void bfs_top_down(Graph graph, solution *sol) {
  Vertex_list l1(graph->num_nodes), l2(graph->num_nodes);
  Vertex_list *front = &l1, *front_next = &l2;

#pragma omp parallel for
  for (int i = 0; i < graph->num_nodes; i++)
    sol->distances[i] = NOT_VISITED_MARKER;

  (*front)[front->length++] = ROOT_NODE_ID;
  sol->distances[ROOT_NODE_ID] = 0;

  while (front->length > 0) {
    front_next->clear_all();
    // auto start = std::chrono::steady_clock::now();
    top_dn(graph, front, front_next, sol->distances);
    // auto end = std::chrono::steady_clock::now();
    // std::chrono::duration<double> elapsed_seconds = end-start;
    // std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
    std::swap(front, front_next);
  }
}

void btm_up(Graph g, Vertex_list *novis, Vertex_list *novis_next,
            Vertex_list &front_next, int *distances,
            std::atomic<int> &novis_count) {
#pragma omp parallel for schedule(guided)
  for (int i = 0; i < novis->length; i++) {
    if (distances[(*novis)[i]] == NOT_VISITED_MARKER) {
      int node = (*novis)[i];
      int start = g->incoming_starts[node];
      int end = (node == g->num_nodes - 1) ? g->num_edges
                                           : g->incoming_starts[node + 1];

      // bool reached = false;
      for (int j = start; j < end; j++) {
        int incoming = g->incoming_edges[j];
        if (distances[incoming] != NOT_VISITED_MARKER) {
          front_next[front_next.length++] = node;
          novis_count--;
          // reached = true;
          break;
        }
      }
      // if (!reached) (*novis_next)[novis_next->length++] = node;
    }
  }
}

void garbage_collect(Vertex_list *novis, Vertex_list *novis_next,
                     int *distances) {
  novis_next->clear_all();
  int cnt = 0;
  // #pragma omp parallel for
  for (int i = 0; i < novis->length; i++) {
    if (distances[(*novis)[i]] == NOT_VISITED_MARKER) {
      (*novis_next)[cnt++] = (*novis)[i];
    }
  }
  novis_next->length = cnt;
}

void bfs_bottom_up(Graph graph, solution *sol) {
  Vertex_list l1(graph->num_nodes), l2(graph->num_nodes);
  Vertex_list *novis = &l1, *novis_next = &l2, front_next(graph->num_nodes);

#pragma omp parallel for
  for (int i = 0; i < graph->num_nodes; i++)
    sol->distances[i] = NOT_VISITED_MARKER;

#pragma omp parallel for
  for (int i = 0; i < graph->num_nodes - 1; i++) (*novis)[i] = i + 1;
  novis->length = graph->num_nodes - 1;
  sol->distances[ROOT_NODE_ID] = 0;
  int depth_count = 1;
  std::atomic<int> novis_count(graph->num_nodes - 1);
  while (novis_count > 0) {
    front_next.clear_all();
    // auto start = std::chrono::steady_clock::now();
    btm_up(graph, novis, novis_next, front_next, sol->distances, novis_count);
    // auto end = std::chrono::steady_clock::now();
    // std::chrono::duration<double> elapsed_seconds = end-start;
    // std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
    if (front_next.length == 0) return;

#pragma omp parallel for
    for (int i = 0; i < front_next.length; i++) {
      sol->distances[front_next[i]] = depth_count;
    }

    if (novis_count < novis->length / 2) {
      garbage_collect(novis, novis_next, sol->distances);
      std::swap(novis, novis_next);
    }
    depth_count++;
  }
}
void bfs_hybrid(Graph graph, solution *sol) {
  Vertex_list l1(graph->num_nodes), l2(graph->num_nodes), l3(graph->num_nodes),
      l4(graph->num_nodes);
  Vertex_list *front = &l1, *front_next = &l2;
  Vertex_list *novis = &l3, *novis_next = &l4;

#pragma omp parallel for
  for (int i = 0; i < graph->num_nodes; i++)
    sol->distances[i] = NOT_VISITED_MARKER;
#pragma omp parallel for
  for (int i = 0; i < graph->num_nodes - 1; i++) (*novis)[i] = i + 1;
  novis->length = graph->num_nodes - 1;

  (*front)[front->length++] = ROOT_NODE_ID;
  sol->distances[ROOT_NODE_ID] = 0;
  int depth_count = 1;
  std::atomic<int> novis_count(graph->num_nodes - 1);

  while (front->length > 0) {
    front_next->clear_all();
    if (novis->length > front->length) {
      top_dn(graph, front, front_next, sol->distances);
      novis_count -= front_next->length;
    } else {
      btm_up(graph, novis, novis_next, *front_next, sol->distances,
             novis_count);
#pragma omp parallel for
      for (int i = 0; i < front_next->length; i++) {
        sol->distances[(*front_next)[i]] = depth_count;
      }
    }
    if (novis_count < novis->length / 4) {
      garbage_collect(novis, novis_next, sol->distances);
      std::swap(novis, novis_next);
    }
    depth_count++;
    std::swap(front, front_next);
  }
}