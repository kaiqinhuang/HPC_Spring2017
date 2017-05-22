/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "book.h"

#define N   10

__global__ void add( int *a, int *b, int *c ) {
    int tid = blockIdx.x;
    if (tid < N)
        c[tid] = a[tid] + b[tid];
}

int main( void ) {
 
    // allocate the memory on the CPU
    thrust::host_vector<int> h_vec_a(N);
    thrust::host_vector<int> h_vec_b(N);
    thrust::host_vector<int> h_vec_c(N);

    // fill the arrays 'a' and 'b' on the CPU
    for (int i=0; i<N; i++) {
        h_vec_a[i] = -i;
        h_vec_b[i] = i * i;
    }

    // copy the arrays 'a' and 'b' to the GPU
    thrust::device_vector<int> d_vec_a = h_vec_a;
    thrust::device_vector<int> d_vec_b = h_vec_b;
    thrust::device_vector<int> d_vec_c(N);
    int *pointer_a = thrust::raw_pointer_cast(&d_vec_a[0]);
    int *pointer_b = thrust::raw_pointer_cast(&d_vec_b[0]);
    int *pointer_c = thrust::raw_pointer_cast(&d_vec_c[0]);

    // invoke add kernal with correct parameters
    add<<<10,1>>> (pointer_a, pointer_b, pointer_c);

    // copy the array 'c' back from the GPU to the CPU
    thrust::copy(d_vec_c.begin(), d_vec_c.end(), h_vec_c.begin());

    //display the results
    for (int i=0; i<N; i++) {
        printf( "%d + %d = %d\n", h_vec_a[i], h_vec_b[i], h_vec_c[i] );
    }

    return 0;
}


