//
//  main.cpp
//  
//
//  Created by Elijah Afanasiev on 25.09.2018.
//
//

// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_error_handling.h"

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

__global__ void vectorAddGPU(float *a, float *b, float *c, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (idx < N)
    {
        c[idx] = a[idx] + b[idx];
    }
}

void unified_sample(int size = 1048576)
{
    printf("Unified device memory\n");
    int n = size;

    int nBytes = n*sizeof(float);

    float *a, *b;  // host data
    float *c;  // results

    printf("Allocating unified device memory on host..\n");
    SAFE_CALL(cudaMallocManaged((void **)&a,n*sizeof(float)));
    SAFE_CALL(cudaMallocManaged((void **)&b,n*sizeof(float)));
    SAFE_CALL(cudaMallocManaged((void **)&c,n*sizeof(float)));

    dim3 block(256);
    dim3 grid((unsigned int)ceil(n/(float)block.x));

    for(int i=0;i<n;i++)
    {
        a[i] = rand() / (float)RAND_MAX;
        b[i] = rand() / (float)RAND_MAX;
        c[i] = 0;
    }

    cudaEvent_t start, stop;
    SAFE_CALL(cudaEventCreate(&start));
    SAFE_CALL(cudaEventCreate(&stop));

    SAFE_CALL(cudaEventRecord(start));

    printf("Doing GPU Vector add\n");

    SAFE_KERNEL_CALL((vectorAddGPU<<<grid, block>>>(a, b, c, n)));

    SAFE_CALL(cudaEventRecord(stop));
    SAFE_CALL(cudaEventSynchronize(stop));
    float milliseconds = 0;
    SAFE_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("time: %f ms\n\n", milliseconds);

    SAFE_CALL(cudaThreadSynchronize());

    SAFE_CALL(cudaFree(a));
    SAFE_CALL(cudaFree(b));
    SAFE_CALL(cudaFree(c));
}

void unified_sample_prefetch(int size = 1048576)
{
    printf("Unified device memory PREFETCH\n");
    int n = size;

    int nBytes = n*sizeof(float);

    float *a, *b;  // host data
    float *c;  // results


    printf("Allocating unified device memory on host..\n");
    cudaEvent_t start_alloc, stop_alloc;
    SAFE_CALL(cudaEventCreate(&start_alloc));
    SAFE_CALL(cudaEventCreate(&stop_alloc));
    SAFE_CALL(cudaEventRecord(start_alloc));

    SAFE_CALL(cudaMallocManaged((void **)&a,n*sizeof(float)));
    SAFE_CALL(cudaMallocManaged((void **)&b,n*sizeof(float)));
    SAFE_CALL(cudaMallocManaged((void **)&c,n*sizeof(float)));
    SAFE_CALL(cudaEventRecord(stop_alloc));
    SAFE_CALL(cudaEventSynchronize(stop_alloc));
    float ms_alloc = 0;
    SAFE_CALL(cudaEventElapsedTime(&ms_alloc, start_alloc, stop_alloc));

    dim3 block(256);
    dim3 grid((unsigned int)ceil(n/(float)block.x));

    for(int i=0;i<n;i++)
    {
        a[i] = rand() / (float)RAND_MAX;
        b[i] = rand() / (float)RAND_MAX;
        c[i] = 0;
    }
    int device;
    cudaGetDevice ( &device );
    printf ( "Device %d\n", device );
    cudaEvent_t start, start_kernel, stop;
    SAFE_CALL(cudaEventCreate(&start));
    SAFE_CALL(cudaEventCreate(&start_kernel));
    SAFE_CALL(cudaEventCreate(&stop));

    SAFE_CALL(cudaEventRecord(start));
    printf("Prefetch unified device memory on host..\n");
    SAFE_CALL(cudaMemPrefetchAsync ( a, n*sizeof(float), device ));
    SAFE_CALL(cudaMemPrefetchAsync ( b, n*sizeof(float), device ));
    SAFE_CALL(cudaMemPrefetchAsync ( c, n*sizeof(float), device ));

    printf("Doing GPU Vector add\n");
    SAFE_CALL(cudaEventRecord(start_kernel));
    SAFE_KERNEL_CALL((vectorAddGPU<<<grid, block>>>(a, b, c, n)));

    SAFE_CALL(cudaEventRecord(stop));
    SAFE_CALL(cudaEventSynchronize(stop));
    float milliseconds = 0, milliseconds_kernel;
    SAFE_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    SAFE_CALL(cudaEventElapsedTime(&milliseconds_kernel, start_kernel, stop));
    printf("allocation time: %f ms\n", ms_alloc);
    printf("time: %f ms\n", milliseconds);
    printf("time kernel: %f ms\n\n", milliseconds_kernel);

    SAFE_CALL(cudaThreadSynchronize());

    SAFE_CALL(cudaFree(a));
    SAFE_CALL(cudaFree(b));
    SAFE_CALL(cudaFree(c));
}

void pinned_sample(int size = 1048576)
{
    printf("PINNED device memory\n");
    int n = size;

    int nBytes = n*sizeof(float);

    float *a, *b;  // host data
    float *c;  // results
    cudaEvent_t start_alloc, stop_alloc;
    SAFE_CALL(cudaEventCreate(&start_alloc));
    SAFE_CALL(cudaEventCreate(&stop_alloc));
    SAFE_CALL(cudaEventRecord(start_alloc));

    SAFE_CALL(cudaMallocHost((void **)&a,n*sizeof(float)));
    SAFE_CALL(cudaMallocHost((void **)&b,n*sizeof(float)));
    SAFE_CALL(cudaMallocHost((void **)&c,n*sizeof(float)));
    SAFE_CALL(cudaEventRecord(stop_alloc));
    SAFE_CALL(cudaEventSynchronize(stop_alloc));
    float ms_alloc = 0;
    SAFE_CALL(cudaEventElapsedTime(&ms_alloc, start_alloc, stop_alloc));

    float *a_d,*b_d,*c_d;

    dim3 block(256);
    dim3 grid((unsigned int)ceil(n/(float)block.x));

    for(int i=0;i<n;i++)
    {
        a[i] = rand() / (float)RAND_MAX;
        b[i] = rand() / (float)RAND_MAX;
        c[i] = 0;
    }

    printf("Allocating pinned device memory on host..\n");
    SAFE_CALL(cudaMalloc((void **)&a_d,n*sizeof(float)));
    SAFE_CALL(cudaMalloc((void **)&b_d,n*sizeof(float)));
    SAFE_CALL(cudaMalloc((void **)&c_d,n*sizeof(float)));

    printf("Copying to device..\n");
    cudaEvent_t start, start_kernel, stop;
    SAFE_CALL(cudaEventCreate(&start));
    SAFE_CALL(cudaEventCreate(&stop));

    SAFE_CALL(cudaEventRecord(start));
    SAFE_CALL(cudaMemcpy(a_d,a,n*sizeof(float), cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(b_d,b,n*sizeof(float), cudaMemcpyHostToDevice));

    printf("Doing GPU Vector add\n");
    SAFE_CALL(cudaEventCreate(&start_kernel));
    SAFE_CALL(cudaEventRecord(start_kernel));
    SAFE_KERNEL_CALL((vectorAddGPU<<<grid, block>>>(a_d, b_d, c_d, n)));

    SAFE_CALL(cudaEventRecord(stop));
    SAFE_CALL(cudaEventSynchronize(stop));
    float milliseconds = 0, milliseconds_kernel =0;
    SAFE_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    SAFE_CALL(cudaEventElapsedTime(&milliseconds_kernel, start_kernel, stop));
    printf("allocation time: %f ms\n", ms_alloc);
    printf("time: %f ms\n", milliseconds);
    printf("time kernel: %f ms\n\n", milliseconds_kernel);

    SAFE_CALL(cudaThreadSynchronize());

    SAFE_CALL(cudaFree(a_d));
    SAFE_CALL(cudaFree(b_d));
    SAFE_CALL(cudaFree(c_d));
}

void usual_sample(int size = 1048576)
{
    printf("USUAL device memory\n");
    int n = size;
    
    int nBytes = n*sizeof(float);
    
    float *a, *b;  // host data
    float *c;  // results
    cudaEvent_t start_alloc, stop_alloc;
    SAFE_CALL(cudaEventCreate(&start_alloc));
    SAFE_CALL(cudaEventCreate(&stop_alloc));
    SAFE_CALL(cudaEventRecord(start_alloc));

    a = (float *)malloc(nBytes);
    b = (float *)malloc(nBytes);
    c = (float *)malloc(nBytes);
    SAFE_CALL(cudaEventRecord(stop_alloc));
    SAFE_CALL(cudaEventSynchronize(stop_alloc));
    float ms_alloc = 0;
    SAFE_CALL(cudaEventElapsedTime(&ms_alloc, start_alloc, stop_alloc));


    
    float *a_d,*b_d,*c_d;
    
    dim3 block(256);
    dim3 grid((unsigned int)ceil(n/(float)block.x));
    
    for(int i=0;i<n;i++)
    {
        a[i] = rand() / (float)RAND_MAX;
        b[i] = rand() / (float)RAND_MAX;
        c[i] = 0;
    }
    
    printf("Allocating device memory on host..\n");
    
    cudaMalloc((void **)&a_d,n*sizeof(float));
    cudaMalloc((void **)&b_d,n*sizeof(float));
    cudaMalloc((void **)&c_d,n*sizeof(float));
    
    printf("Copying to device..\n");

    cudaEvent_t start, start_kernel, stop;
    SAFE_CALL(cudaEventCreate(&start));
    SAFE_CALL(cudaEventCreate(&start_kernel));
    SAFE_CALL(cudaEventCreate(&stop));

    SAFE_CALL(cudaEventRecord(start));
    
    cudaMemcpy(a_d,a,n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d,b,n*sizeof(float), cudaMemcpyHostToDevice);
    
    printf("Doing GPU Vector add\n");
    SAFE_CALL(cudaEventRecord(start_kernel));
    vectorAddGPU<<<grid, block>>>(a_d, b_d, c_d, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0, milliseconds_kernel = 0;
    SAFE_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    SAFE_CALL(cudaEventElapsedTime(&milliseconds_kernel, start_kernel, stop));
    printf("allocation time: %f ms\n", ms_alloc);
    printf("time: %f ms\n", milliseconds);
    printf("time kernel: %f ms\n\n", milliseconds_kernel);
    
    cudaThreadSynchronize();
    
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}


int main(int argc, char **argv)
{
    usual_sample(atoi(argv[1]));
    pinned_sample(atoi(argv[1]));
    unified_sample(atoi(argv[1]));
    unified_sample_prefetch(atoi(argv[1]));
    return 0;
}
