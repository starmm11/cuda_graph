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
#include <cublas.h>
#include <omp.h>

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

void sample_vec_add(int size = 1048576)
{
    int n = size;
    
    int nBytes = n*sizeof(int);
    
    float *a, *b;  // host data
    float *c;  // results
    
    a = (float *)malloc(nBytes);
    b = (float *)malloc(nBytes);
    c = (float *)malloc(nBytes);
    float *c_ans = (float *)malloc(nBytes);
    
    float *a_d,*b_d,*c_d;
    
    dim3 block(256);
    dim3 grid((unsigned int)ceil(n/(float)block.x));
    
    for(int i=0;i<n;i++)
    {
        a[i] = rand() / (float)RAND_MAX;
        b[i] = rand() / (float)RAND_MAX;
        c[i] = 0;
        c_ans[i] = a[i] + b[i];
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
    
    vectorAddGPU<<<grid, block>>>(a_d, b_d, c_d, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaMemcpy(c, c_d, size * sizeof(float), cudaMemcpyDeviceToHost);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("time: %f ms\n", milliseconds);
    
    cudaThreadSynchronize();
    for (int i = 0; i < size; ++i) {
        if (c_ans[i] != c[i]) {
            printf("Test is failed\n");
            printf("%f %f", c_ans[i],c[i]);
            return;
        }

    }
    printf("Tests are good\n");
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    free(a);
    free(b);
    free(c);
}

void streams_vec_add(int size = 1048576)
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("gpu count: %d\n", deviceCount);
    float *a, *b;  // host data
    float *c;  // results
    float *cpu_ans;
    a = new float[size];
    b = new float[size];
    c = new float[size];
    cpu_ans = new float[size];
    //cudaHostRegister(a,size * sizeof(float),0);
    //cudaHostRegister(b,size * sizeof(float),0);
    for(int i=0;i<size;i++)
    {
        a[i] = rand() / (float)RAND_MAX;
        b[i] = rand() / (float)RAND_MAX;
        c[i] = 0;
    }
    for(int i=0;i<size;i++)
    {
        cpu_ans[i] = a[i] + b[i];
    }
    double start = omp_get_wtime();
    #pragma omp parallel num_threads(deviceCount)
    {
        int device = omp_get_thread_num();
        printf("device: %d\n", device);
        cudaSetDevice(device); // устанавливаем для каждого потока свой контекст
        int elemsPerDevice = size/deviceCount;
        float *a_d,*b_d,*c_d;
        SAFE_CALL(cudaMalloc(&a_d, elemsPerDevice * sizeof(float)));
        SAFE_CALL(cudaMalloc(&b_d, elemsPerDevice * sizeof(float)));
        SAFE_CALL(cudaMalloc(&c_d, elemsPerDevice * sizeof(float)));
        SAFE_CALL(cudaMemcpy(a_d, a + device*elemsPerDevice, elemsPerDevice * sizeof(float), cudaMemcpyHostToDevice));
        SAFE_CALL(cudaMemcpy(b_d, b + device*elemsPerDevice, elemsPerDevice * sizeof(float), cudaMemcpyHostToDevice));
        int blockSize = 128;
        int gridSize = (elemsPerDevice-1)/ blockSize + 1;
        SAFE_KERNEL_CALL((vectorAddGPU<<<gridSize, blockSize>>>(a_d, b_d, c_d, elemsPerDevice)));
        SAFE_CALL(cudaMemcpy(c + device*elemsPerDevice, c_d, elemsPerDevice * sizeof(float), cudaMemcpyDeviceToHost));
        SAFE_CALL(cudaThreadSynchronize());
        SAFE_CALL(cudaDeviceSynchronize());
    }
    double end = omp_get_wtime();
    printf("time %f\n", (end - start)*1e4);
    for (int i = 0; i < size; ++i) {
        if (cpu_ans[i] != c[i]) {
            printf("Test is failed\n");
            printf("%f %f\n", cpu_ans[i],c[i]);
            return;
        }

    }
    printf("Tests are good\n");
}


int main(int argc, char **argv)
{
    sample_vec_add(atoi(argv[1]));
    streams_vec_add(atoi(argv[1]));

    return 0;
}
