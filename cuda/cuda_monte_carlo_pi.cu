#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h> // CURAND lib header file
#define TRIALS_PER_THREAD 1024 // Set the value for global variables
#define BLOCKS 256
#define THREADS 512
#define PI 3.1415926535 // Known value of pi, to calculate error


__global__ void pi_mc(float *estimate, curandState *states){
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int points_in_circle = 0;
    float x, y;
    
    // Initialize CURAND
    curand_init(tid, 0, 0, &states[tid]);
    for(int i = 0; i < TRIALS_PER_THREAD; i++){
        x = curand_uniform(&states[tid]);
        y = curand_uniform(&states[tid]);        
        // Count if x & y is in the circle
        points_in_circle += (x*x + y*y <= 1.0f);
    }
    estimate[tid] = 4.0f * points_in_circle / (float) TRIALS_PER_THREAD;
}


int main(int argc, char *argv[]){
    float host[BLOCKS * THREADS];
    float *dev;
    curandState *devStates;

    // Allocate memory on GPU
    cudaMalloc((void **) &dev, BLOCKS * THREADS * sizeof(float));
    cudaMalloc((void **) &devStates, BLOCKS * THREADS * sizeof(curandState));
    
    // Invoke the kernel
    pi_mc<<<BLOCKS, THREADS>>>(dev, devStates);
    
    // Copy from device back to host
    cudaMemcpy(host, dev, BLOCKS * THREADS * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free the memory on GPU
    cudaFree(dev);
    cudaFree(devStates);
    
    // Get the average estimate pi value among all blocks and threads, and calculate error
    float pi_gpu = 0.0;
    for(int i = 0; i < BLOCKS * THREADS; i++){
        pi_gpu += host[i];
    }
    pi_gpu /= (BLOCKS * THREADS);
    printf("Trials per thread is: %d, number of blocks is: %d, number of threads is: %d\n", 
		TRIALS_PER_THREAD, BLOCKS, THREADS);
    printf("CUDA estimate of PI = %f [error of %f]\n", pi_gpu, pi_gpu - PI);
    
    return 0;
}



