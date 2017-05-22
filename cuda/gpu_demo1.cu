#include <stdio.h>
#include "book.h"

#define N (1024*1024)
#define M (1000000)

  __global__ void cudaKernel(float *buf) {
     int i = threadIdx.x + blockIdx.x * blockDim.x; //what is this line doing?
   
/* Grids made of blocks
 * Blocks made of threads
 * threadIdx - threadId: built-in variable that stores the id of each thread
 * blockIdx - blockId
 * blockDim - block dimentions
 * gridDim - grid dimensions  

 * This line is assigning value to the index.  The number of indices equals num_threads * num_blocks.
 */

     buf[i] = 1.0f * i / N;
     for(int j = 0; j < M; j++)
        buf[i] = buf[i] * buf[i] - 0.25f;
  }

  int main() {
     float data[N]; //array on host
     float *d_data; //device pointer

     //allocate memory on GPU
     int size = N * sizeof(float);
     cudaMalloc((void **) &d_data, N * sizeof(float));
    
     //invoke kernel with 4096 blocks of 256 threads
     cudaKernel<<<4096, 256>>> (d_data);

     //copy results back to host
     cudaMemcpy(data, d_data, size, cudaMemcpyDeviceToHost);

     
     cudaFree(d_data); 

     int input;
     printf("Enter an index: ");
     scanf("%d", &input);
     printf("data[%d] = %f\n", input, data[input]);
  }


/* cudaFree() frees the memory on GPU
 * printf() prints from CPU
 * that is why I can do printf() after cudaFree(), was confused earlier
 */ 
