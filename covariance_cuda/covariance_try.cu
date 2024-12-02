#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>
/* Include benchmark-specific header. */
/* Default data type is double, default size is 4000. */
#include "covariance.h"
#define BLOCK_DIM 32


__global__ void kernel_covariance(DATA_TYPE float_n, DATA_TYPE* __restrict__ data, DATA_TYPE* __restrict__ mean, DATA_TYPE* __restrict__ symmat)
{   
    // Compute the row (i) and column (j) indices for this thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.y;
    int index = j * blockDim.x * gridDim.x + i; //indice globale
    // Ensure we are within bounds
    if (index < M * N and tid < M) {
        data[index] = ((DATA_TYPE)i * j) / M;
        symmat[index] = 0.0;
        // valutare utilizzo __device__ e inline, per richiamre funzioni con parti di codice pulizia codice

        /* Determine mean of column vectors of input data matrix */
        mean[tid] = 0.0;
        __syncthreads();
        
        atomicAdd(mean[tid], data[index]);
        __syncthreads();

        if (atomicCAS(&flags[tid], 0, 1) == 0) {
            // Esegui la divisione una sola volta
            mean[tid] /= float_n;
        }        

        /*
        for (j < _PB_M; j++)
          if(threadIdx.x == j)
              lock.lock();
              mean[j] /= float_n;
              lock.unlock();
        */
        __syncthreads();
       
        atomicSub(data[index], mean[tid]);
        //__syncthreads();

        __shared__ DATA_TYPE temp = 0.0;

        if(tid+1 < M)
          temp += data[i][tid] * data[i][tid+1];


    }

  /*int i, j, j1, j2;

  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++)
      data[i j] = ((DATA_TYPE) i*j) / M;

  
    for (j = 0; j < M; j++)
      {
        mean[j] = 0.0;
	for (i = 0; i < N; i++)
	  mean[j] += data[i j];
	mean[j] /= float_n;
      }
      
    
    for (i = 0; i < N; i++)
      for (j = 0; j < M; j++)
	      data[i j] -= mean[j];
      
    
    /*for (j1 = 0; j1 < M; j1++)
        for (j2 = j1; j2 < M; j2++)
	      {
          symmat[j1 j2] = 0.0;
	        for (i = 0; i < N; i++)
	          symmat[j1][j2] += data[i][j1] * data[i][j2];
	        symmat[j2][j1] = symmat[j1][j2];
        }*/
}

int main(int argc, char** argv)
{
  /* Retrieve problem size. */

  double wt;
  struct timespec rt[2];
  /* Variable declaration/allocation. */
  DATA_TYPE float_n = 1.2;
  DATA_TYPE *h_symmat;
  DATA_TYPE *d_mean, *d_data, *d_symmat;
  int* d_flags;

  cudaMalloc(&d_flags, size * sizeof(int));
  cudaMemset(d_flags, 0, size * sizeof(int));
  cudaMallocHost((void**)&h_symmat,sizeof(DATA_TYPE) * M * M); 
  cudaMalloc((void**)&d_data, sizeof(DATA_TYPE) * M * N);
  cudaMalloc((void**)&d_mean, sizeof(DATA_TYPE) * M);
  cudaMalloc((void**)&d_symmat, sizeof(DATA_TYPE) * M * M);
  
  /* Start timer. */
  clock_gettime(CLOCK_REALTIME, rt + 0); // non va dopo?
  dim3 dimBlock(BLOCK_DIM, BLOCK_DIM); 
  dim3 dimGrid(((N+BLOCK_DIM-1)/BLOCK_DIM)/2, ((N+BLOCK_DIM-1)/BLOCK_DIM)/2);
  /* Run kernel. */
  kernel_covariance<<<dimGrid,dimBlock>>>(float_n, d_data, d_mean, d_symmat);  
  cudaMemcpy(h_symmat, d_symmat, sizeof(DATA_TYPE) * M * M, cudaMemcpyDeviceToHost);
  /* Stop and print timer. */
  clock_gettime(CLOCK_REALTIME, rt + 1);
  wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
  printf("GEMM (Host) : %9.3f sec %9.1f GFLOPS\n", wt, 2.0 * N * N * N / (1.0e9 * wt));


  /* Be clean. */
  cudaFree(d_data);
  cudaFree(d_symmat);
  cudaFree(d_mean);
  cudaFreeHost(h_symmat);

  return 0;
}