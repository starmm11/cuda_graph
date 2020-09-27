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
#include <vector>
using namespace std;

// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_error_handling.h"

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

__global__ void vectorAddGPU(float *a, float *b, float *c, int N, int offset)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (offset + idx < N)
    {
        c[offset + idx] = a[offset + idx] + b[offset + idx];
    }
}

void sample_vec_add(int size = 1048576)
{
    int n = size;
    
    int nBytes = n*sizeof(int);
    
    float *a, *b;  // host data
    float *c;  // results
    
    a = (float *)malloc(nBytes);
    b = (float *)malloc(nBytes);
    c = (float *)malloc(nBytes);
    
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
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    cudaMemcpy(a_d,a,n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d,b,n*sizeof(float), cudaMemcpyHostToDevice);
    
    printf("Doing GPU Vector add\n");
    
    vectorAddGPU<<<grid, block>>>(a_d, b_d, c_d, n, 0);
    cudaMemcpy(c,c_d,n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("time: %f ms\n", milliseconds);
    
    cudaDeviceSynchronize();
    
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}

void streams_vec_add(int size = 1048576, int n_streams = 0)
{
    int n = size;
    int nBytes = n*sizeof(int);
    
    float *a, *b;  // host data
    float *c;  // results
    
    cudaHostAlloc( (void**) &a, n * sizeof(float) ,cudaHostAllocDefault );
    cudaHostAlloc( (void**) &b, n * sizeof(float) ,cudaHostAllocDefault );
    cudaHostAlloc( (void**) &c, n * sizeof(float) ,cudaHostAllocDefault );
    
    float *a_d,*b_d,*c_d;
    
    for(int i=0;i<n;i++)
    {
        a[i] = rand() / (float)RAND_MAX;
        b[i] = rand() / (float)RAND_MAX;
        c[i] = 0;
    }
    
    printf("Allocating device memory on host..\n");

    cudaMalloc((void **)&a_d,n*sizeof(float));
    printf("here\n");
    cudaMalloc((void **)&b_d,n*sizeof(float));
    cudaMalloc((void **)&c_d,n*sizeof(float));

    printf("Copying to device..\n");
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    printf("Doing GPU Vector add\n");
    
    //vectorAddGPU<<<grid, block>>>(a_d, b_d, c_d, n);
    
    const int NbStreams = n_streams;
    const int StreamSize = n / NbStreams;
    vector<cudaStream_t> Stream(NbStreams);
    for ( int i = 0; i < NbStreams; i++ )
        SAFE_CALL(cudaStreamCreate(&Stream[i]));
    
    for ( int i = 0; i < NbStreams; i++ )
    {
        int Offset = i * StreamSize;
        
        cudaMemcpyAsync(&a_d[Offset], &a[Offset], StreamSize * sizeof(float),
                                  cudaMemcpyHostToDevice, Stream[i]);
        cudaMemcpyAsync(&b_d[Offset], &b[Offset], StreamSize * sizeof(float),
                                  cudaMemcpyHostToDevice, Stream[i]);
//        cudaMemcpyAsync(&c_d[Offset], &c[Offset], StreamSize * sizeof(float),
//                                  cudaMemcpyHostToDevice, Stream[i]);
        
        dim3 block(1024);
        dim3 grid((StreamSize - 1)/1024 + 1);
        vectorAddGPU<<<grid, block, 0, Stream[i]>>>(a_d, b_d, c_d, StreamSize, Offset);

//        cudaMemcpyAsync(&a[Offset], &a_d[Offset], StreamSize * sizeof(float),
//                                  cudaMemcpyDeviceToHost, Stream[i]);
//        cudaMemcpyAsync(&b[Offset], &b_d[Offset], StreamSize * sizeof(float),
//                                  cudaMemcpyDeviceToHost, Stream[i]);
        cudaMemcpyAsync(&c[Offset], &c_d[Offset], StreamSize * sizeof(float),
                                  cudaMemcpyDeviceToHost, Stream[i]);

    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("time: %f ms\n", milliseconds);
    
    cudaDeviceSynchronize();
    
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    
    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);
}


int main(int argc, char **argv)
{
    if (argc < 3) {
        printf("Incorrect number of arguments\n");
    }
    sample_vec_add(atoi(argv[1]));
    streams_vec_add(atoi(argv[1]), atoi(argv[2]));

    return 0;
}
