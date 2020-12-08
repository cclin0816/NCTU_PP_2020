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

  long long hit =
      cal_pi(tosses / world_size + (tosses % world_size <= world_rank ? 0 : 1),
             world_rank);

  unsigned round = 1;
  // unsigned mask = 0xfffffffe;

  long long recv_hit;
  MPI_Status status;
  while (round < world_size) {
    if ((world_rank & round) != 0) {
      MPI_Send(&hit, 1, MPI_LONG_LONG_INT, world_rank - round, 0,
               MPI_COMM_WORLD);
      break;
    } else if (world_rank + round < world_size) {
      MPI_Recv(&recv_hit, 1, MPI_LONG_LONG_INT, world_rank + round, 0,
               MPI_COMM_WORLD, &status);
      hit += recv_hit;
    }
    round = round << 1;
  }

  if (world_rank == 0) {
    pi_result = 4.0 / tosses * hit;
    // --- DON'T TOUCH ---
    double end_time = MPI_Wtime();
    printf("%lf\n", pi_result);
    printf("MPI running time: %lf Seconds\n", end_time - start_time);
    // ---
  }

  MPI_Finalize();
  return 0;
}
