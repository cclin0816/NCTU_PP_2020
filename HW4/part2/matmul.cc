#include <mpi.h>

#include <cstdio>

inline int cal_row_len(int n, int rank, int size) {
  return n / size + (n % size > rank ? 1 : 0);
}

void do_matmul(const int n, const int m, const int l, const int *a_mat,
               const int *b_mat, int *c_mat) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      for (int k = 0; k < l; k++) {
        c_mat[i * l + k] += a_mat[i * m + j] * b_mat[j * l + k];
      }
    }
  }
}

void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr, int **a_mat_ptr,
                        int **b_mat_ptr) {
  int world_rank, world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (world_rank == 0) {
    scanf("%d", n_ptr);
    scanf("%d", m_ptr);
    scanf("%d", l_ptr);
  }
  MPI_Request nml_req[3];
  MPI_Status nml_stat[3];

  MPI_Ibcast(n_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD, &(nml_req[0]));
  MPI_Ibcast(m_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD, &(nml_req[1]));
  MPI_Ibcast(l_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD, &(nml_req[2]));
  MPI_Waitall(3, nml_req, nml_stat);

  int row_len = cal_row_len(*n_ptr, world_rank, world_size);

  int a_size;
  int b_size = *m_ptr * *l_ptr;
  if (world_rank == 0) {
    a_size = *n_ptr * *m_ptr;
  } else {
    a_size = row_len * *m_ptr;
  }

  *a_mat_ptr = new int[a_size];
  *b_mat_ptr = new int[b_size];

  MPI_Request *req;
  MPI_Status *stat;
  int count;

  if (world_rank == 0) {
    req = new MPI_Request[world_size];
    stat = new MPI_Status[world_size];
    count = world_size;

    for (int i = 0; i < a_size; i++) {
      scanf("%d", &((*a_mat_ptr)[i]));
    }

    int sum_row = row_len;
    for (int i = 1; i < world_size; i++) {
      int t_row_len = cal_row_len(*n_ptr, i, world_size);
      MPI_Isend(&((*a_mat_ptr)[sum_row * *m_ptr]), t_row_len * *m_ptr, MPI_INT,
                i, 0, MPI_COMM_WORLD, &(req[i]));
      sum_row += t_row_len;
    }

    for (int i = 0; i < b_size; i++) {
      scanf("%d", &((*b_mat_ptr)[i]));
    }
  } else {
    req = new MPI_Request[2];
    stat = new MPI_Status[2];
    count = 2;
    MPI_Irecv((*a_mat_ptr), row_len * *m_ptr, MPI_INT, 0, 0, MPI_COMM_WORLD,
              &(req[1]));
  }
  MPI_Ibcast(*b_mat_ptr, b_size, MPI_INT, 0, MPI_COMM_WORLD, &(req[0]));
  MPI_Waitall(count, req, stat);
  delete[] req;
  delete[] stat;
}

void matrix_multiply(const int n, const int m, const int l, const int *a_mat,
                     const int *b_mat) {
  int world_rank, world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int row_len = cal_row_len(n, world_rank, world_size);
  int c_size;
  if (world_rank == 0) {
    c_size = n * l;
  } else {
    c_size = row_len * l;
  }
  int *c_mat = new int[c_size];
  for (int i = 0; i < row_len * l; i++) {
    c_mat[i] = 0;
  }

  do_matmul(row_len, m, l, a_mat, b_mat, c_mat);

  if (world_rank == 0) {
    int count = world_size - 1;
    MPI_Request *req = new MPI_Request[count];
    MPI_Status *stat = new MPI_Status[count];

    int sum_row = row_len;

    for (int i = 1; i < world_size; i++) {
      int t_row_len = cal_row_len(n, i, world_size);
      MPI_Irecv(&(c_mat[sum_row * l]), t_row_len * l, MPI_INT, i, 1,
                MPI_COMM_WORLD, &(req[i - 1]));
      sum_row += t_row_len;
    }
    MPI_Waitall(count, req, stat);
    delete[] req;
    delete[] stat;
  } else {
    MPI_Send(c_mat, c_size, MPI_INT, 0, 1, MPI_COMM_WORLD);
  }

  if (world_rank == 0) {
    for (int i = 0; i < c_size; i++) {
      printf("%d", c_mat[i]);
      if((i + 1) % l == 0) {
        printf("\n");
      }
      else {
        printf(" ");
      }
    }
    fflush(stdout);
  }
  delete[] c_mat;
}

void destruct_matrices(int *a_mat, int *b_mat) {
  delete[] a_mat;
  delete[] b_mat;
}