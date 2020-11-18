#include "page_rank.h"

#include <omp.h>
#include <stdlib.h>

#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is
// num_nodes(g)) damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence) {
  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;
  for (int i = 0; i < numNodes; ++i) {
    solution[i] = equal_prob;
  }
  double *sol_new = new double[numNodes];
  double *sol_out = new double[numNodes];
  int *out_num = new int[numNodes];
  int *in_num = new int[numNodes];
  std::vector<int> no_out_ver;
#pragma omp parallel for schedule(guided)
  for (int i = 0; i < numNodes; ++i) {
    out_num[i] = outgoing_size(g, i);
    if (out_num[i] == 0) {
#pragma omp critical
      no_out_ver.push_back(i);
    }
    in_num[i] = incoming_size(g, i);
  }
  double global_diff = convergence;
  double add_up = (1.0 - damping) / numNodes;
  while (global_diff >= convergence) {
#pragma omp parallel for
    for (int i = 0; i < numNodes; ++i) {
      if (out_num[i] != 0) sol_out[i] = solution[i] / out_num[i];
    }
#pragma omp parallel for schedule(guided)
    for (int i = 0; i < numNodes; ++i) {
      sol_new[i] = 0;
      for (int j = 0; j < in_num[i]; ++j) {
        sol_new[i] += sol_out[g->incoming_edges[g->incoming_starts[i] + j]];
      }
      sol_new[i] = sol_new[i] * damping + add_up;
    }
    double s = 0;
#pragma omp parallel for reduction(+ : s)
    for (int i = 0; i < no_out_ver.size(); ++i) {
      s += damping * solution[no_out_ver[i]] / numNodes;
    }
#pragma omp parallel for
    for (int i = 0; i < numNodes; ++i) {
      sol_new[i] += s;
    }

    global_diff = 0;
#pragma omp parallel for reduction(+ : global_diff)
    for (int i = 0; i < numNodes; ++i) {
      global_diff += abs(solution[i] - sol_new[i]);
    }
    for (int i = 0; i < numNodes; ++i) {
      solution[i] = sol_new[i];
    }
  }
  delete[] sol_new;
  delete[] sol_out;
  delete[] out_num;
  delete[] in_num;
  /*
     For PP students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi])
     }; converged = (global_diff < convergence)
     }

   */
}
