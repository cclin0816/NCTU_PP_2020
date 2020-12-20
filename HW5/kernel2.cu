#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BlockSize 16

__global__ void mandelKernel(int *cudaMem, float lowerX, float lowerY,
                             float stepX, float stepY, int maxIteration,
                             int widthX, int widthY, int pitch) {
  // To avoid error caused by the floating number, use the following pseudo code
  //
  int xid = threadIdx.x + blockIdx.x * BlockSize;
  int yid = threadIdx.y + blockIdx.y * BlockSize;
  if (xid >= widthX || yid >= widthY) return;
  float x = lowerX + xid * stepX;
  float y = lowerY + yid * stepY;
  float z_re = x, z_im = y;
  int i;
  for (i = 0; i < maxIteration; ++i) {
    if (z_re * z_re + z_im * z_im > 4.f) break;

    float new_re = z_re * z_re - z_im * z_im;
    float new_im = 2.f * z_re * z_im;
    z_re = x + new_re;
    z_im = y + new_im;
  }
  cudaMem[xid + yid * pitch] = i;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE(float upperX, float upperY, float lowerX, float lowerY, int *img,
            int resX, int resY, int maxIterations) {
  float stepX = (upperX - lowerX) / resX;
  float stepY = (upperY - lowerY) / resY;
  int *Mem;
  cudaHostAlloc((void **)&Mem, resX * resY * sizeof(int), cudaHostAllocDefault);
  int *cudaMem;
  size_t pitch;
  cudaMallocPitch((void **)&cudaMem, &pitch, resX * sizeof(int), resY);
  fflush(stdout);
  dim3 dimBlock(BlockSize, BlockSize);
  dim3 dimGrid((resX / BlockSize) + (resX % BlockSize == 0 ? 0 : 1),
               (resY / BlockSize) + (resY % BlockSize == 0 ? 0 : 1));
  mandelKernel<<<dimGrid, dimBlock>>>(cudaMem, lowerX, lowerY, stepX, stepY,
                                      maxIterations, resX, resY, pitch / sizeof(int));
  cudaDeviceSynchronize();
  cudaMemcpy2D((void *)Mem, resX * sizeof(int), (void *)cudaMem, pitch,
               resX * sizeof(int), resY, cudaMemcpyDeviceToHost);
  memcpy((void *)img, (void *)Mem, resX * resY * sizeof(int));
  cudaFree(cudaMem);
  cudaFreeHost(Mem);
}
