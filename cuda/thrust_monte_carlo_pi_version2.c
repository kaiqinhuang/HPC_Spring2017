#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <thrust/generate.h>
#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <cmath>

struct montecarlo :
public thrust::unary_function<unsigned int, float>
{
    //use the __host__ __device__ decorators so that
    //thrust knows that this is a gpu function.
    //think of operator() as similar to a CUDA kernel
    __host__ __device__
    float operator()(unsigned int thread_id)
    {
        unsigned int seed;
        //set the seed
        seed = 49868^thread_id;
        int i;
        float x,y,z,sum=0.0;
        
        //define the random number generator
        thrust::default_random_engine rng(seed);
        thrust::uniform_real_distribution u01(0,1);
        for (i = 0; i < 1000; i++)
        {
            //get the random numbers
            x = u01(rng);
            y = u01(rng);
            z = ((x*x)+(y*y));
            if (z<=1)
                sum += 1;
        }
        return sum;
    }
};

int main(int argc, char* argv[])
{
    float pi;
    float count = 0.0;
    int niter = 1000;
    count = thrust::transform_reduce(thrust::counting_iterator(0),
                                     thrust::counting_iterator(niter),
                                     montecarlo(),
                                     0.0,
                                     thrust::plus());
    pi = (count/(niter*niter))*4.0;
    printf("Pi: %f\n", pi);
}
