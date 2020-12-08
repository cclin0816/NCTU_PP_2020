#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

long long cal_pi(long long toss, int rank) {
  long long hit = 0;
  unsigned seed = time(0) ^ (rank << rank);
  for (int i = 0; i < toss; i++) {
    float x, y;
    x = ((float)rand_r(&seed)) / RAND_MAX;
    y = ((float)rand_r(&seed)) / RAND_MAX;
    if (x * x + y * y <= 1.0) {
      hit++;
    }
  }
  return hit;
}

int main(int argc, char **argv) {
  // --- DON'T TOUCH ---
  MPI_Init(&argc, &argv);
  double start_time = MPI_Wtime();
  double pi_result;
  long long int tosses = atoi(argv[1]);
  int world_rank, world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  long long *gather;
  long long hit =
      cal_pi(tosses / world_size + (tosses % world_size <= world_rank ? 0 : 1),
             world_rank);

  if (world_rank == 0) {
    MPI_Request *requests =
        (MPI_Request *)malloc((world_size - 1) * sizeof(MPI_Request));
    MPI_Status *status =
        (MPI_Status *)malloc((world_size - 1) * sizeof(MPI_Status));
    gather = (long long *)malloc(world_size * sizeof(long long));
    gather[0] = hit;
    for (int i = 1; i < world_size; i++) {
      MPI_Irecv(&(gather[i]), 1, MPI_LONG_LONG_INT, i, 0, MPI_COMM_WORLD,
                &(requests[i - 1]));
    }
    MPI_Waitall(world_size - 1, requests, status);

    free(requests);
    free(status);
  } else {
    MPI_Send(&hit, 1, MPI_LONG_LONG_INT, 0, 0, MPI_COMM_WORLD);
  }

  if (world_rank == 0) {
    long long total_hit = 0;
    for (int i = 0; i < world_size; i++) {
      total_hit += gather[i];
    }
    pi_result = 4.0 / tosses * total_hit;
    // --- DON'T TOUCH ---
    double end_time = MPI_Wtime();
    printf("%lf\n", pi_result);
    printf("MPI running time: %lf Seconds\n", end_time - start_time);
    // ---
    free(gather);
  }

  MPI_Finalize();
  return 0;
}
