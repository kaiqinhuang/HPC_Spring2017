#include <stdio.h>
#include "book.h"


__global__ void cudaKernelSaveId(int *threadId)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    threadId[i] = i;
}

    
int main(int argc, char *argv[])
{
    int N;

    if (argc < 2) {  
    // the condition is argc<2 instead of argc<1 because the first argument is the file name
	N = 512;
    } else {
	N = atoi(argv[1]);  // need function atoi because N is int but argv is char
    }

    int thread[N];
    int *dev_thread;

    int size = N * sizeof(int);
    cudaMalloc((void **) &dev_thread, size);
    cudaKernelSaveId<<<1, N>>> (dev_thread);
    cudaMemcpy(thread, dev_thread, size, cudaMemcpyDeviceToHost);

    cudaFree(dev_thread);

    for (int i=0; i<N; i++) {
        printf("thread[%d] = %d\n", i, i);
    }
 
    return 0;
}







