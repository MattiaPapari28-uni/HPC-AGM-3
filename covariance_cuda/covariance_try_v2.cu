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
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,__double_as_longlong(val + __longlong_as_double(assumed)));
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
    }
}

__global__ void kernel_covariance(DATA_TYPE* data, DATA_TYPE* mean, int* flags, DATA_TYPE float_n, DATA_TYPE* symmat)
  {   
    int tidy = threadIdx.y;
    int tidx = threadIdx.x;
    int j = blockIdx.x * blockDim.x + tidx;
    int i = blockIdx.y * blockDim.y + tidy;
    if (i < M and j < N) {
      if(atomicCAS(&flags[j],0,1) == 0) 
          mean[j] /= float_n;
      __syncthreads();
       
      atomicAddDouble(&data[i * M + j], -(mean[j]));
      DATA_TYPE sum = 0.0;
         
      if(j <= i){
        for (int k = 0; k < N; k++) {
            sum += (data[k * M + j] * data[k * M + i]);
        }

        symmat[j * M + i] = sum;
        symmat[i * M + j] = sum;  
      }
    }
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
  int* h_flags;

  cudaMallocManaged((void**)&h_flags, M * sizeof(int));
  cudaMallocManaged((void**)&h_symmat,sizeof(DATA_TYPE) * M * M); 
  cudaMallocManaged((void**)&h_mean,sizeof(DATA_TYPE) * M); 
  cudaMallocManaged((void**)&h_data,sizeof(DATA_TYPE) * M * N); 
  cudaMemset(h_symmat, 0, M * M * sizeof(DATA_TYPE));
  cudaMemset(h_mean, 0, M * sizeof(DATA_TYPE));
  cudaMemset(h_flags, 0, M * sizeof(int)); //setto a 0 tutte le celle
  /* Start timer. */
  clock_gettime(CLOCK_REALTIME, rt + 0); 
  dim3 dimBlock(BLOCK_DIM, BLOCK_DIM); 
  dim3 dimGrid(((N+BLOCK_DIM-1)/BLOCK_DIM), ((N+BLOCK_DIM-1)/BLOCK_DIM));
  /* Run kernel. */
  kernel_calc_mean<<<dimGrid,dimBlock>>>(h_data, h_mean); //calcolo il vettore mean in un kernel a parte così da poter risolvere i problemi di sincronizzazione  
  kernel_covariance<<<dimGrid,dimBlock>>>(h_data, h_mean, h_flags, float_n, h_symmat);
  cudaDeviceSynchronize();
  /* Stop and print timer. */
  clock_gettime(CLOCK_REALTIME, rt + 1);
  wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
  printf("GEMM (device) : %9.3f sec %9.1f GFLOPS\n", wt, 2.0 * N * N * N / (1.0e9 * wt));

  /* Be clean. */
  cudaFree(h_data);
  cudaFree(h_symmat);
  cudaFree(h_mean);
  cudaFree(h_flags);

  return 0;
}
