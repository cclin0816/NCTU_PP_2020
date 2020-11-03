#include <pthread.h>

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <random>

using namespace std;

atomic<long long> total_hits(0);

typedef struct {
  long long tosses;
  uint32_t seed;
} Arg;

void *cal_pi(void *arg) {
  mt19937 generator(((Arg *)arg)->seed);
  uniform_real_distribution<float> distribution(0.0, 1.0);
  long long hits = 0;
  for (long long i = 0; i < ((Arg *)arg)->tosses; i++) {
    float x = distribution(generator);
    float y = distribution(generator);
    if (x * x + y * y < 1.0) hits++;
  }
  total_hits += hits;
  pthread_exit(nullptr);
}

int main(int argc, char **argv) {
  int cores = atoi(argv[1]);
  long long tosses = atoll(argv[2]);
  pthread_t *threads = new pthread_t[cores];
  Arg *args = new Arg[cores];
  random_device rd;
  for (int i = 0; i < cores; i++) {
    args[i].seed = rd();
    args[i].tosses = tosses / cores + (tosses % cores > i ? 1 : 0);
  }
  for (int i = 0; i < cores; i++)
    pthread_create(&threads[i], nullptr, cal_pi, (void *)&args[i]);
  void *status;
  for (int i = 0; i < cores; i++) pthread_join(threads[i], &status);
  cout << 4 * total_hits / (double)tosses << endl;
}