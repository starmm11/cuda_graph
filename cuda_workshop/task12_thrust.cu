// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_error_handling.h"
#include <cublas.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>

using namespace thrust;
using namespace thrust::placeholders;

struct is_even
{
    __host__ __device__
    bool operator()(const int x)
    {
        return (x % 2) == 0;
    }
};

int main(int argc, char **argv)
{
    int n = 20;
    thrust::host_vector<int> d_v(n);
    for (int i = 0; i < n; ++i) {
        d_v[i] = i*i;
    }
    //thrust::generate(d_v.begin(), d_v.end(), rand);

    thrust::host_vector<int> copy(n);
    //copy_if(thrust::host, d_v.begin(), d_v.end(), copy.begin(), is_even());
    copy_if(make_counting_iterator<int>(0),
            make_counting_iterator<int>(d_v.size()),
            d_v.begin(),
            copy.begin(),
            is_even());
    for (int i = 0; i < copy.size(); ++i) {
        printf("%d ", copy[i]);
    }
    printf("\n\n");
    return 0;
}
