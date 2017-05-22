/******************************************************************************
* FILE: omp_getEnvInfo.c
* DESCRIPTION:
*   OpenMP Example - Get Environment Information - C/C++ Version
*   The master thread queries and prints selected environment information.
* AUTHOR: Blaise Barney  7/06
* LAST REVISED: 07/12/06
******************************************************************************/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[]) 
{
int nthreads, tid, procs, maxt, inpar, dynamic, nested;

/* Start parallel region */
#pragma omp parallel private(nthreads, tid)
  {

  /* Obtain thread number */
  tid = omp_get_thread_num();

  /* Only master thread does this */
  if (tid == 0) 
    {
    printf("Thread %d getting environment info...\n", tid);

    /* Get environment information */
    procs = omp_get_num_procs();
    nthreads = omp_get_num_threads();
    maxt = omp_get_max_threads();
    inpar = omp_in_parallel();
    dynamic = omp_get_dynamic();
    nested = omp_get_nested();

    /* Print environment information */
    printf("Number of processors = %d\n", procs);
    printf("Number of threads = %d\n", nthreads);
    printf("Max threads = %d\n", maxt);
    printf("In parallel? = %d\n", inpar);
    printf("Dynamic threads enabled? = %d\n", dynamic);
    printf("Nested parallelism enabled? = %d\n", nested);

    }

  }  /* Done */

}


/* Output:

 Thread 0 getting environment info...
 Number of processors = 8
 Number of threads = 8
 Max threads = 8
 In parallel? = 1
 Dynamic threads enabled? = 0
 Nested parallelism enabled? = 0

*/
