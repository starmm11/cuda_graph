// UCSC CMPE220 Advanced Parallel Processing 
// Prof. Heiner Leitz
// Author: Marcelo Siero.
// Modified from code by:: Andreas Goetz (agoetz@sdsc.edu)
// CUDA program to perform 1D stencil operation in parallel on the GPU
//
// /* FIXME */ COMMENTS ThAT REQUIRE ATTENTION

#include <iostream>
#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include "cuda_error_handling.h"

// define vector length, stencil radius, 

int N = 1024*1024*128+20;
const int RADIUS = 7;
const int BLOCKSIZE = 1024;
const int GRIDSIZE = (N-1)/BLOCKSIZE + 1;
int gridSize = GRIDSIZE;
int blockSize = BLOCKSIZE;

void start_timer(cudaEvent_t& start, cudaEvent_t& stop) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
}

float stop_timer(cudaEvent_t& start, cudaEvent_t& stop) {
    cudaDeviceSynchronize();
    // time counting terminate
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_time_ms = 0;
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    return (elapsed_time_ms);
}

cudaDeviceProp prop;

void getDeviceProperties() {
    printf("Get device properties\n");
}

void newline() { std::cout << std::endl; };

void printThreadSizes() {
    int noOfThreads = gridSize * blockSize;
    printf("Blocks            = %d\n", gridSize);  // no. of blocks to launch.
    printf("Threads per block = %d\n", blockSize); // no. of threads to launch.
    printf("Total threads     = %d\n", noOfThreads);
    printf("Number of grids   = %d\n", (N + noOfThreads - 1) / noOfThreads);
}

// ----------------------------------------------------------
// CUDA global device function that performs 1D stencil operation
// ---------------------------------------------------------
__global__ void stencil_1D_global(double *in, double *out, long dim){

    long gindex = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Go through all data
    // Step all threads in a block to avoid synchronization problem
    while ( gindex < (dim + blockDim.x) ) {

        /* FIXME PART 2 - MODIFIY PROGRAM TO USE SHARED MEMORY. */

        // Apply the stencil
        double result = 0;
        for (int offset = -RADIUS; offset <= RADIUS; offset++) {
            if ( gindex + offset < dim && gindex + offset > -1)
                result += in[gindex + offset];
        }

        // Store the result
        if (gindex < dim)
            out[gindex] = result;

        // Update global index and quit if we are done
        gindex += stride;

        __syncthreads();

    }
}


// ----------------------------------------------------------
// CUDA shared device function that performs 1D stencil operation
// ---------------------------------------------------------
__global__ void stencil_1D(double *in, double *out, long dim) {
    long gindex = threadIdx.x + blockDim.x * blockIdx.x;
    int tx = threadIdx.x + RADIUS;
    __shared__ double array[BLOCKSIZE + 2 * RADIUS];
    double result = 0;
    if (gindex < dim) {
        array[tx] = in[gindex];
    } else {
        array[tx] = 0;
    }
    if (threadIdx.x < RADIUS) {
        if (gindex - RADIUS >= 0) {
            array[tx - RADIUS] = in[gindex - RADIUS];
        } else {
            array[tx - RADIUS] = 0;
        }
    }

//    if (threadIdx.x < RADIUS) {
//        if (gindex + BLOCKSIZE < dim) {
//            array[tx + BLOCKSIZE] = in[gindex + BLOCKSIZE];
//        } else {
//            array[tx + BLOCKSIZE] = 0;
//        }
//    }

    if (threadIdx.x >= BLOCKSIZE - RADIUS) {
        if (gindex + RADIUS < dim) {
            array[tx + RADIUS] = in[gindex + RADIUS];
        } else {
            array[tx + RADIUS] = 0;
        }
    }

    __syncthreads();

    for (int offset = -RADIUS; offset <= RADIUS; offset++) {
        result += array[tx + offset];
    }
    __syncthreads();
    if (gindex < dim) {
        out[gindex] = result;
    }
}

#define True  1
#define False 0

void checkResults(double *h_in, double *h_out, int DoCheck = True) {
    // DO NOT CHANGE THIS CODE.
    // CPU calculates the stencil from data in *h_in
    // if DoCheck is True (default) it compares it with *h_out
    // to check the operation of this code.
    // If DoCheck is set to False, it can be used to time the CPU.
    int i, j, ij;
    double result;
    int err = 0;
    for (i = 0; i < N; i++) {  // major index.
        result = 0;
        for (j = -RADIUS; j <= RADIUS; j++) {
            ij = i + j;
            if (ij >= 0 && ij < N)
                result += h_in[ij];
        }
        if (DoCheck) {  // print out some errors for debugging purposes.
            if (h_out[i] != result) { // count errors.
                err++;
                if (err < 8) { // help debug
                    printf("h_out[%d]=%f should be %f\n", i, h_out[i], result);
                };
            }
        } else {  // for timing purposes.
            h_out[i] = result;
        }
    }

    if (DoCheck) { // report results.
        if (err != 0) {
            printf("Error, %d elements do not match!\n", err);
        } else {
            printf("Success! All elements match CPU result.\n");
        }
    }
}

// ------------
// main program
// ------------
int main(void) {
    double *h_in, *h_out;
    double *d_in, *d_out;
    long size = N * sizeof(double);
    int i;
    getDeviceProperties();
    // allocate host memory
    h_in = new double[N];
    h_out = new double[N];
    // initialize vector
    for (i = 0; i < N; i++) {
        //    h_in[i] = i+1;
        h_in[i] = 1;
    }

    // allocate device memory
    SAFE_CALL(cudaMalloc((void **) &d_in, size));
    SAFE_CALL(cudaMalloc((void **) &d_out, size));

    // copy input data to device
    SAFE_CALL(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    // Apply stencil by launching a sufficient number of blocks
    printf("\n---------------------------\n");
    printf("Launching 1D stencil kernel\n");
    printf("---------------------------\n");
    printf("Vector length     = %ld (%ld MB)\n", N, N * sizeof(double) / 1024 / 1024);
    printf("Stencil radius    = %d\n", RADIUS);

    //----------------------------------------------------------
    // CODE TO RUN AND TIME THE STENCIL KERNEL.
    //----------------------------------------------------------
    newline();
    printThreadSizes();
    cudaEvent_t start, stop;
    start_timer(start, stop);
    SAFE_KERNEL_CALL((stencil_1D<<<gridSize, blockSize>>>(d_in, d_out, N)));
    std::cout << "Elapsed shared time: " << stop_timer(start, stop) << std::endl;
    // copy results back to host
    SAFE_CALL(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));
    //checkResults(h_in, h_out);

    //----------------------------------------------------------
    //-----GLOBAL VERSION
    start_timer(start, stop);
    SAFE_KERNEL_CALL((stencil_1D_global<<<gridSize, blockSize>>>(d_in, d_out, N)));
    std::cout << "Elapsed global time: " << stop_timer(start, stop) << std::endl;
    // copy results back to host
    SAFE_CALL(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));
    //checkResults(h_in, h_out);


    // deallocate device memory
    cudaFree(d_in);
    cudaFree(d_out);
    //=====================================================
    // Evaluate total time of execution with just the CPU.
    //=====================================================
    newline();
    std::cout << "Running stencil with the CPU.\n";
    start_timer(start, stop);
    // Use checkResults to time CPU version of the stencil with False flag.
    checkResults(h_in, h_out, False);
    std::cout << "Elapsed time: " << stop_timer(start, stop) << std::endl;
    //=====================================================

    // deallocate host memory
    free(h_in);
    free(h_out);

    return 0;
}
