/******************************************************************************
* FILE: omp_section.c
* DESCRIPTION:
*   OpenMP Example - Sections Work-sharing - C Version
*   In this example, the OpenMP SECTION directive is used to assign
*   different array operations to each thread that executes a SECTION. 
* AUTHOR: Blaise Barney  5/99
* LAST REVISED: 07/16/07
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N     50

int main (int argc, char *argv[]) 
{
int i, nthreads, tid;
float a[N], b[N], c[N], d[N];

/* Some initializations */
for (i=0; i<N; i++) {
  a[i] = i * 1.5;
  b[i] = i + 22.35;
  c[i] = d[i] = 0.0;
  }

#pragma omp parallel shared(a,b,c,d,nthreads) private(i,tid)
  {
  tid = omp_get_thread_num();
  if (tid == 0)
    {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  printf("Thread %d starting...\n",tid);

  #pragma omp sections nowait
    {
    #pragma omp section
      {
      printf("Thread %d doing section 1\n",tid);
      for (i=0; i<N; i++)
        {
        c[i] = a[i] + b[i];
        printf("Thread %d: c[%d]= %f\n",tid,i,c[i]);
        }
      }

    #pragma omp section
      {
      printf("Thread %d doing section 2\n",tid);
      for (i=0; i<N; i++)
        {
        d[i] = a[i] * b[i];
        printf("Thread %d: d[%d]= %f\n",tid,i,d[i]);
        }
      }

    }  /* end of sections */

    printf("Thread %d done.\n",tid); 

  }  /* end of parallel section */

}


/* Output:
 
 Thread 6 starting...
 Thread 6 doing section 1
 Thread 6: c[0]= 22.350000
 Thread 6: c[1]= 24.850000
 Thread 6: c[2]= 27.350000
 Thread 6: c[3]= 29.850000
 Thread 6: c[4]= 32.349998
 Thread 6: c[5]= 34.849998
 Thread 6: c[6]= 37.349998
 Thread 6: c[7]= 39.849998
 Thread 6: c[8]= 42.349998
 Thread 6: c[9]= 44.849998
 Thread 6: c[10]= 47.349998
 Thread 6: c[11]= 49.849998
 Thread 6: c[12]= 52.349998
 Thread 6: c[13]= 54.849998
 Thread 6: c[14]= 57.349998
 Thread 6: c[15]= 59.849998
 Thread 6: c[16]= 62.349998
 Thread 6: c[17]= 64.849998
 Thread 6: c[18]= 67.349998
 Thread 6: c[19]= 69.849998
 Thread 6: c[20]= 72.349998
 Thread 6: c[21]= 74.849998
 Thread 6: c[22]= 77.349998
 Thread 6: c[23]= 79.849998
 Thread 6: c[24]= 82.349998
 Thread 6: c[25]= 84.849998
 Thread 6: c[26]= 87.349998
 Thread 6: c[27]= 89.849998
 Thread 6: d[28]= 2114.699951
 Thread 6: d[29]= 2233.724854
 Thread 6: d[30]= 2355.750000
 Thread 6: d[31]= 2480.774902
 Thread 6: d[32]= 2608.799805
 Thread 6: d[33]= 2739.824951
 Thread 6: d[34]= 2873.849854
 Thread 6: d[35]= 3010.875000
 Thread 6: d[36]= 3150.899902
 Thread 6: d[37]= 3293.924805
 Thread 6: d[38]= 3439.949951
 Thread 6: d[39]= 3588.974854
 Thread 6: d[40]= 3741.000000
 Thread 6: d[41]= 3896.024902
 Thread 6: d[42]= 4054.049805
 Thread 6: d[43]= 4215.074707
 Thread 6: d[44]= 4379.100098
 Thread 6: d[45]= 4546.125000
 Thread 6: d[46]= 4716.149902
 Thread 6: d[47]= 4889.174805
 Thread 6: d[48]= 5065.199707
 Thread 6: d[49]= 5244.225098
 Thread 6 done.
 Thread 7 starting...
 Thread 7 done.
 Thread 4 starting...
 Thread 4 done.
 Number of threads = 8
 Thread 0 starting...
 Thread 0 done.
 Thread 5 starting...
 Thread 5 done.
 Thread 2 starting...
 Thread 2 done.
 Thread 1 starting...
 Thread 1 done.
 Thread 3 starting...
 Thread 3 done.

*/
