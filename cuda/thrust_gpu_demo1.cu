#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <stdio.h>
#include "book.h"

#define N (1024*1024)
#define M (1000000)

  __global__ void cudaKernel(float *buf) {
     int i = threadIdx.x + blockIdx.x * blockDim.x; 

     buf[i] = 1.0f * i / N;
     for(int j = 0; j < M; j++)
        buf[i] = buf[i] * buf[i] - 0.25f;
  }

  int main() {

     //allocate memory on CPU
     thrust::host_vector<float> h_vec(N);
     thrust::device_vector<float> d_vec(N);
     float *raw_d = thrust::raw_pointer_cast(&d_vec[0]);

     //invoke kernel with 4096 blocks of 256 threads
     cudaKernel<<<4096, 256>>> (raw_d);

     //copy results back to host
     thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

     int input;
     printf("Enter an index: ");
     scanf("%d", &input);
     printf("data[%d] = %f\n", input, h_vec[input]);
  }

 
