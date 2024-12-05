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


__device__ double atomicAddDouble(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

__global__ void kernel_calc_mean(DATA_TYPE* data, DATA_TYPE* mean)
{   
    // Compute the row (i) and column (j) indices for this thread
    int tidy = threadIdx.y; //indice della colonna
    int tidx = threadIdx.x;
    int j = blockIdx.x * blockDim.x + tidx; //fino a M
    int i = blockIdx.y * blockDim.y + tidy;
    // Ensure we are within bounds
    if (i < M and j < N) {
        data[i * M + j] = ((DATA_TYPE)i * j) / M;
        __syncthreads();
        atomicAddDouble(&mean[j], data[i * M + j]);
        __syncthreads();

    }
}

__global__ void kernel_covariance(DATA_TYPE* data, DATA_TYPE* mean, int* flags, DATA_TYPE float_n)
  {   
    // Compute the row (i) and column (j) indices for this thread
    int tidy = threadIdx.y; //indice della colonna
    int tidx = threadIdx.x;
    int j = blockIdx.x * blockDim.x + tidx; //fino a M
    int i = blockIdx.y * blockDim.y + tidy;
    // Ensure we are within bounds
    if (i < M and j < N) {
       if(atomicCAS(&flags[j],0,1) == 0) //si può testare rimuovendo il calcolo ed eseguendo direttamente sia sottrazione che divisione
          mean[j] /= float_n;
        __syncthreads();
       
        atomicAddDouble(&data[i * M + j], -(mean[j]));
        

        //__shared__ DATA_TYPE temp;

        /*int j1 = blockIdx.x * blockDim.x + threadIdx.x;
        int j2 = blockIdx.y * blockDim.y + threadIdx.y;

        // Ensure indices are within bounds
        if (j1 < M && j2 < M && j1 <= j2) {  // Per calcolare solo la metà alta, in modo da non computare tutto che tanto con la simmetrica non conviene
          float sum = 0.0f;
          for (int i = 0; i < N; i++) {
            sum += data[i * M + j1] * data[i * M + j2];
          }
          symmat[j1 * M + j2] = sum;
          symmat[j2 * M + j1] = sum; 
        }*/

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
	        for (i   = 0; i < N; i++)
	          symmat[j1][j2] += data[i][j1] * data[i][j2];
	        symmat[j2][j1] = symmat[j1][j2];
        }
    */
}

static
void print_array(DATA_TYPE* h_symmat)
{
  int i;
  FILE *ftpr;
  ftpr = fopen("file2.txt", "w");
  for (i = 0; i < M * M; i++) {
      fprintf (ftpr, DATA_PRINTF_MODIFIER, h_symmat[i]);
      if (i % 20 == 0) fprintf (ftpr, "\n");
    }
  fprintf (ftpr, "\n");
  fclose(ftpr);
}

int main(int argc, char** argv)
{
  /* Retrieve problem size. */

  double wt;
  struct timespec rt[2];
  /* Variable declaration/allocation. */
  DATA_TYPE float_n = 1.2;
  DATA_TYPE *h_symmat, *h_mean, *h_data;
  DATA_TYPE *d_mean, *d_data, *d_symmat;
  int* d_flags;

  cudaMalloc(&d_flags, M * sizeof(int));
  cudaMemset(d_flags, 0, M * sizeof(int)); //setto a 0 tutte le celle
  cudaMallocHost((void**)&h_symmat,sizeof(DATA_TYPE) * M * M); 
  cudaMallocHost((void**)&h_mean,sizeof(DATA_TYPE) * M); 
  cudaMallocHost((void**)&h_data,sizeof(DATA_TYPE) * M * N); 
  cudaMalloc((void**)&d_data, sizeof(DATA_TYPE) * M * N);
  cudaMalloc((void**)&d_mean, sizeof(DATA_TYPE) * M);
  cudaMalloc((void**)&d_symmat, sizeof(DATA_TYPE) * M * M);
  cudaMemset(d_symmat, 0, M * M * sizeof(DATA_TYPE));
  cudaMemset(d_mean, 0, M * sizeof(DATA_TYPE));
  /* Start timer. */
  clock_gettime(CLOCK_REALTIME, rt + 0); // non va dopo?
  dim3 dimBlock(BLOCK_DIM, BLOCK_DIM); 
  dim3 dimGrid(((N+BLOCK_DIM-1)/BLOCK_DIM), ((N+BLOCK_DIM-1)/BLOCK_DIM));
  /* Run kernel. */
  kernel_calc_mean<<<dimGrid,dimBlock>>>(d_data, d_mean); //calcolo il vettore mean in un kernel a parte così da poter risolvere i problemi di sincronizzazione  
  cudaMemcpy(h_mean, d_mean, sizeof(DATA_TYPE) * M, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_data, d_data, sizeof(DATA_TYPE) * M * M, cudaMemcpyDeviceToHost);
  kernel_covariance<<<dimGrid,dimBlock>>>(d_data, d_mean, d_flags, float_n);
  cudaMemcpy(h_data, d_data, sizeof(DATA_TYPE) * M * M, cudaMemcpyDeviceToHost);
  /* Stop and print timer. */
  clock_gettime(CLOCK_REALTIME, rt + 1);
  wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
  printf("GEMM (device) : %9.3f sec %9.1f GFLOPS\n", wt, 2.0 * N * N * N / (1.0e9 * wt));
  print_array(h_data);

  /* Be clean. */
  cudaFree(d_data);
  cudaFree(d_symmat);
  cudaFree(d_mean);
  cudaFreeHost(h_symmat);

  return 0;
}