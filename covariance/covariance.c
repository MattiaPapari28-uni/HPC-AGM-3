#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 4000. */
#include "covariance.h"



/* Array initialization. 
 *
 * I think here we have a problem because we dont use m and n but the M and N (defined in covariance.h or polybench)*/
static
void init_array (int m, int n,
		 DATA_TYPE *float_n,
		 DATA_TYPE POLYBENCH_2D(data,M,N,m,n))
{
  int i, j;
  printf("%d, %d\n", m, n);
  printf("%d, %d\n", M, N);
  printf("%d, %d\n", _PB_M, _PB_N);

  *float_n = 1.2;
  //Questa parte deve essere parallelizzata ci siamo un po' persi qualcosa perchè a volte migliora altre no, forse serve qualcosa che non abbiamo trovato
  //

//#pragma omp target data map(from: data[:M][:N])
////#pragma target teams omp distribute parallel for collapse(2)
  for (i = 0; i < M; i++)
	for (j = 0; j < N; j++)
		data[i][j] = ((DATA_TYPE) i*j) / M; //DATA TYPE è un cast
   

  printf("\n\n");    

}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m,
		 DATA_TYPE POLYBENCH_2D(symmat,M,M,m,m))

{
  int i, j;
//#pragma omp teams distribute parallel for collapse(2)
  for (i = 0; i < m; i++)
    for (j = 0; j < m; j++) {
      fprintf (stderr, DATA_PRINTF_MODIFIER, symmat[i][j]);
      if ((i * m + j) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_covariance(int m, int n,
		       DATA_TYPE float_n,
		       DATA_TYPE POLYBENCH_2D(data,M,N,m,n),
		       DATA_TYPE POLYBENCH_2D(symmat,M,M,m,m),
		       DATA_TYPE POLYBENCH_1D(mean,M,m))
{
int i, j, j1, j2;
  
  /* Determine mean of column vectors of input data matrix */

/*
carichiamo in memoria device e host le variabili/array necessari all'esecuzione 
dell'algoritmo
i primi due pragma lavoreranno su due gruppi di for, 
uno per il calcolo della media
uno per il calcolo della sottrazione della media sulle colonne 
in sostanza dove c'è target si ha anche l'utilizzo di graffe
*/
#pragma omp target data map(to: _PB_N, _PB_M, float_n) map(tofrom: mean[0:_PB_M], data[:_PB_N][:_PB_M])   
{


//ritorna numero dei threads utilizzabili sulla macchina in esecuzione
int NTHREADS_GPU = omp_get_num_threads();


//andiamo a creare un numero di teams pari al _PB_M/NTHREADS_GPU e diamo un limite di threads a NTHREADS_GPU per ogni teams
#pragma omp target teams num_teams(_PB_M/NTHREADS_GPU) thread_limit(NTHREADS_GPU)
{  
//distribuisce il carico di lavoro tra i threads di tutti i teams
    #pragma omp distribute parallel for num_threads(NTHREADS_GPU) dist_schedule(static, NTHREADS_GPU) 
    for (j = 0; j < _PB_M; j++)
    {
      mean[j] = 0.0;
      for (i = 0; i < _PB_N; i++)
	  mean[j] += data[i][j]; // mean[j] = mean[j]+data[i][j]
      mean[j] /= float_n;
    }
    /* Center the colum vectors. */
//distribuisce il carico di lavoro a tutti i threads
//collapse -> trasforma i for anaidati i un singolo for permettendo di migliorare le presetanzioni
    #pragma omp distribute parallel for collapse(2)
    for (i = 0; i < _PB_N; i++)
      for (j = 0; j < _PB_M; j++)
        data[i][j] -= mean[j];
    //media sulle colonne
}
}



    /* Center the column vectors. */
/*
 * precedente versione
#pragma omp target teams map(to: mean[:_PB_M], _PB_N, _PB_M) map(tofrom:data[:_PB_N][:_PB_M])
    #pragma omp distribute parallel for collapse(2) 
    for (i = 0; i < _PB_N; i++)
      for (j = 0; j < _PB_M; j++)
	data[i][j] -= mean[j];
*/



    /* Calculate the m * m covariance matrix. */
//#pragma omp target teams map(to: _PB_M, _PB_N, data[:_PB_M][:_PB_N]) map(symmat[:_PB_M][:_PB_M])
//{

//#pragma omp distribute dist_schedule(static, 1)
    for (j1 = 0; j1 < _PB_M; j1++)
    {
      for (j2 = j1; j2 < _PB_M; j2++) // j2=0 : _PB_M > j2=1 : _PB_M
	{
	  symmat[j1][j2] = 0.0;

  //    	  #pragma omp parallel for
	  for (i = 0; i < _PB_N; i++)
	    symmat[j1][j2] += data[i][j1] * data[i][j2]; 
	  // symmat[j1][j2] = symmat[j1][j2] + data[i][j1] * data[i][j2];
	  symmat[j2][j1] = symmat[j1][j2]; //trasposta ^T
	}
    }

//}
    //calcola la covarianza 
}

int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int m = M;

  /* Variable declaration/allocation. */
  DATA_TYPE float_n;
  POLYBENCH_2D_ARRAY_DECL(data,DATA_TYPE,M,N,m,n);
  POLYBENCH_2D_ARRAY_DECL(symmat,DATA_TYPE,M,M,m,m);
  POLYBENCH_1D_ARRAY_DECL(mean,DATA_TYPE,M,m);
  //print_array(m, POLYBENCH_ARRAY(data)); 
  /* Initialize array(s). */
  init_array (m, n, &float_n, POLYBENCH_ARRAY(data));
  //print_array(m, POLYBENCH_ARRAY(data)); 
  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_covariance (m, n, float_n,
		     POLYBENCH_ARRAY(data),
		     POLYBENCH_ARRAY(symmat),
		     POLYBENCH_ARRAY(mean));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, POLYBENCH_ARRAY(symmat)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(data);
  POLYBENCH_FREE_ARRAY(symmat);
  POLYBENCH_FREE_ARRAY(mean);

  return 0;
}
