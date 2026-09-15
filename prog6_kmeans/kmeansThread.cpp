#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <thread>

#include "CycleTimer.h"

using namespace std;

typedef struct {
  // Control work assignments
  int start, end;

  // Shared by all functions
  double *data;
  double *clusterCentroids;
  int *clusterAssignments;
  double *currCost;
  int M, N, K;
} WorkerArgs;


/**
 * Checks if the algorithm has converged.
 * 
 * @param prevCost Pointer to the K dimensional array containing cluster costs 
 *    from the previous iteration.
 * @param currCost Pointer to the K dimensional array containing cluster costs 
 *    from the current iteration.
 * @param epsilon Predefined hyperparameter which is used to determine when
 *    the algorithm has converged.
 * @param K The number of clusters.
 * 
 * NOTE: DO NOT MODIFY THIS FUNCTION!!!
 */
static bool stoppingConditionMet(double *prevCost, double *currCost,
                                 double epsilon, int K) {
  for (int k = 0; k < K; k++) {
    if (abs(prevCost[k] - currCost[k]) > epsilon)
      return false;
  }
  return true;
}

/**
 * Computes L2 distance between two points of dimension nDim.
 * 
 * @param x Pointer to the beginning of the array representing the first
 *     data point.
 * @param y Poitner to the beginning of the array representing the second
 *     data point.
 * @param nDim The dimensionality (number of elements) in each data point
 *     (must be the same for x and y).
 */
double dist(double *x, double *y, int nDim) {
  double accum = 0.0;
  for (int i = 0; i < nDim; i++) {
    accum += pow((x[i] - y[i]), 2);
  }
  return sqrt(accum);
}

/**
 * Assigns each data point to its "closest" cluster centroid.
 */
void computeAssignments(WorkerArgs *const args) {

  // Assign each datapoint in this worker's range to its closest centroid
  for (int m = args->start; m < args->end; m++) {
    double minDist = 1e30;
    int bestAssignment = -1;

    for (int k = 0; k < args->K; k++) {
      double d = dist(&args->data[m * args->N],
                      &args->clusterCentroids[k * args->N], args->N);

      if (d < minDist) {
        minDist = d;
        bestAssignment = k;
      }
    }

    args->clusterAssignments[m] = bestAssignment;
  }
}

/**
 * Given the cluster assignments, computes the new centroid locations for
 * each cluster.
 */
void computeCentroids(WorkerArgs *const args) {
  int *counts = new int[args->K];

  // Zero things out
  for (int k = 0; k < args->K; k++) {
    counts[k] = 0;
    for (int n = 0; n < args->N; n++) {
      args->clusterCentroids[k * args->N + n] = 0.0;
    }
  }


  // Sum up contributions from assigned examples
  for (int m = 0; m < args->M; m++) {
    int k = args->clusterAssignments[m];
    for (int n = 0; n < args->N; n++) {
      args->clusterCentroids[k * args->N + n] +=
          args->data[m * args->N + n];
    }
    counts[k]++;
  }

  // Compute means
  for (int k = 0; k < args->K; k++) {
    counts[k] = max(counts[k], 1); // prevent divide by 0
    for (int n = 0; n < args->N; n++) {
      args->clusterCentroids[k * args->N + n] /= counts[k];
    }
  }

  delete[] counts;
}

/**
 * Computes the per-cluster cost. Used to check if the algorithm has converged.
 */
void computeCost(WorkerArgs *const args) {

  // Each worker computes private per-cluster costs
  for (int k = 0; k < args->K; k++) {
    args->currCost[k] = 0.0;
  }

  for (int m = args->start; m < args->end; m++) {
    int k = args->clusterAssignments[m];
    args->currCost[k] +=
        dist(&args->data[m * args->N],
             &args->clusterCentroids[k * args->N], args->N);
  }
}

/**
 * Computes the K-Means algorithm, using std::thread to parallelize the work.
 *
 * @param data Pointer to an array of length M*N representing the M different N 
 *     dimensional data points clustered. The data is layed out in a "data point
 *     major" format, so that data[i*N] is the start of the i'th data point in 
 *     the array. The N values of the i'th datapoint are the N values in the 
 *     range data[i*N] to data[(i+1) * N].
 * @param clusterCentroids Pointer to an array of length K*N representing the K 
 *     different N dimensional cluster centroids. The data is laid out in
 *     the same way as explained above for data.
 * @param clusterAssignments Pointer to an array of length M representing the
 *     cluster assignments of each data point, where clusterAssignments[i] = j
 *     indicates that data point i is closest to cluster centroid j.
 * @param M The number of data points to cluster.
 * @param N The dimensionality of the data points.
 * @param K The number of cluster centroids.
 * @param epsilon The algorithm is said to have converged when
 *     |currCost[i] - prevCost[i]| < epsilon for all i where i = 0, 1, ..., K-1
 */
void kMeansThread(double *data, double *clusterCentroids, int *clusterAssignments,
               int M, int N, int K, double epsilon) {

  // Used to track convergence
  double *prevCost = new double[K];
  double *currCost = new double[K];

  // The WorkerArgs array is used to pass inputs to and return output from
  // functions.
  WorkerArgs args;
  args.data = data;
  args.clusterCentroids = clusterCentroids;
  args.clusterAssignments = clusterAssignments;
  args.currCost = currCost;
  args.M = M;
  args.N = N;
  args.K = K;

  // Initialize arrays to track cost
  for (int k = 0; k < K; k++) {
    prevCost[k] = 1e30;
    currCost[k] = 0.0;
  }

  double assignmentTime = 0.0;
  double centroidTime = 0.0;
  double costTime = 0.0;

  /* Main K-Means Algorithm Loop */
  int iter = 0;
  while (!stoppingConditionMet(prevCost, currCost, epsilon, K)) {
    // Update cost arrays (for checking convergence criteria)
    for (int k = 0; k < K; k++) {
      prevCost[k] = currCost[k];
    }

    double start = CycleTimer::currentSeconds();

    const int numThreads = 12;
    thread workers[numThreads];
    WorkerArgs workerArgs[numThreads];

    for (int t = 0; t < numThreads; t++) {
      workerArgs[t] = args;
      workerArgs[t].start = t * M / numThreads;
      workerArgs[t].end = (t + 1) * M / numThreads;

      workers[t] = thread(computeAssignments, &workerArgs[t]);
    }

    for (int t = 0; t < numThreads; t++) {
      workers[t].join();
    }

    assignmentTime += CycleTimer::currentSeconds() - start;

    // Setup args struct for centroid and cost computation
    args.start = 0;
    args.end = K;

    start = CycleTimer::currentSeconds();
    computeCentroids(&args);
    centroidTime += CycleTimer::currentSeconds() - start;
    start = CycleTimer::currentSeconds();

    thread costWorkers[numThreads];
    WorkerArgs costWorkerArgs[numThreads];
    double *localCosts = new double[numThreads * K];

    for (int t = 0; t < numThreads; t++) {
      costWorkerArgs[t] = args;
      costWorkerArgs[t].start = t * M / numThreads;
      costWorkerArgs[t].end = (t + 1) * M / numThreads;
      costWorkerArgs[t].currCost = &localCosts[t * K];

      costWorkers[t] = thread(computeCost, &costWorkerArgs[t]);
    }

    for (int t = 0; t < numThreads; t++) {
      costWorkers[t].join();
    }

    for (int k = 0; k < K; k++) {
      currCost[k] = 0.0;
      for (int t = 0; t < numThreads; t++) {
        currCost[k] += localCosts[t * K + k];
      }
    }

    delete[] localCosts;

    costTime += CycleTimer::currentSeconds() - start;
    iter++;
  }

  double profiledTime = assignmentTime + centroidTime + costTime;

  printf("[Iterations]: %d\n", iter);
  printf("[Assignments]: %.3f ms (%.2f%%)\n",
         assignmentTime * 1000,
         100.0 * assignmentTime / profiledTime);

  printf("[Centroids]: %.3f ms (%.2f%%)\n",
         centroidTime * 1000,
         100.0 * centroidTime / profiledTime);

  printf("[Cost]: %.3f ms (%.2f%%)\n",
         costTime * 1000,
         100.0 * costTime / profiledTime);

  delete[] currCost;
  delete[] prevCost;
}
