#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_error_handling.h"
#include <iostream>
#include <algorithm>
#include <ctime>
#include <cstdio>
#include <vector>
#include <chrono>
#include <string>

using namespace std;
using namespace std::chrono;

const double ERR = 1e-5;
__global__ void vector_add(const double* d_a, const double* d_b, double* d_c, int n) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id < n) {
        d_c[id] = d_a[id] + d_b[id];
    }
}

__global__ void vector_add_div(const double* d_a, const double* d_b, double* d_c, int n) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;

    if (d_a[id] < 0.3) {
        d_c[id] = d_a[id] - d_b[id];
    } else if (d_a[id] < 0.6) {
        d_c[id] = d_a[id] + d_b[id];
    } else if (d_a[id] < 1) {
        d_c[id] = d_a[id] + 2*d_b[id];
    } else {
        return;
    }
    while(d_a[id] < 0.5) {
        id++;
    }
    
}

template<class T>
bool CheckAnswers(const vector<T>& a, const vector<T>& b) {
    size_t n = min(a.size(), b.size());
    for (int i = 0; i < n; ++i) {
        if (fabs(a[i]-b[i]) > ERR) {
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n",
               static_cast<int>(error_id), cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }
    if (deviceCount == 0) {
        printf("There are no available device(s) that support CUDA\n");
    } else {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    if (argc < 2) {
        cout << "Incorrect number of arguments\n";
        return 0;
    }
    int n = stoi(argv[1]);
    size_t size = n * sizeof(double);
    vector<double> h_a(n), h_b(n), h_c1(n), h_c2(n);
    //double *h_a = (double *)malloc(size);
    //double *h_b = (double *)malloc(size);
    //double *h_c = (double *)malloc(size);
    vector<double> ans_c(n);
    for (int i = 0; i < n; ++i) {
        h_a[i] = (double)rand()/RAND_MAX;
        h_b[i] = (double)rand()/RAND_MAX;
    }

    double *d_a1, *d_b1, *d_c1;
    double *d_a2, *d_b2, *d_c2;

    SAFE_CALL(cudaMalloc(&d_a1, size));
    SAFE_CALL(cudaMalloc(&d_b1, size));
    SAFE_CALL(cudaMalloc(&d_c1, size));
    SAFE_CALL(cudaMalloc(&d_a2, size));
    SAFE_CALL(cudaMalloc(&d_b2, size));
    SAFE_CALL(cudaMalloc(&d_c2, size));

    SAFE_CALL(cudaMemcpy(d_a1, &h_a[0], size, cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(d_b1, &h_b[0], size, cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(d_a2, &h_a[0], size, cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(d_b2, &h_b[0], size, cudaMemcpyHostToDevice));

    int blockSize, gridSize;
    blockSize = 1024;
    gridSize = (n-1)/blockSize + 1;
    //blockSize = 1000;
    //gridSize = 1;
    cout << "blockSize " << blockSize << '\n';
    cout << "gridSize "  << gridSize << '\n';
    cout << "n_elements "  << n << '\n';
    cudaEvent_t start, stop;
    SAFE_CALL(cudaEventCreate(&start));
    SAFE_CALL(cudaEventCreate(&stop));
    SAFE_CALL(cudaEventRecord(start));

    SAFE_KERNEL_CALL((vector_add<<<gridSize, blockSize>>>(d_a1, d_b1, d_c1, n)));

    SAFE_CALL(cudaDeviceSynchronize());
    SAFE_CALL(cudaEventRecord(stop));

    SAFE_CALL(cudaMemcpy(&h_c1[0], d_c1, size, cudaMemcpyDeviceToHost));

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaFree(d_a1);
    cudaFree(d_b1);
    cudaFree(d_c1);

    cudaEvent_t start2, stop2;
    SAFE_CALL(cudaEventCreate(&start2));
    SAFE_CALL(cudaEventCreate(&stop2));
    SAFE_CALL(cudaEventRecord(start2));

    SAFE_KERNEL_CALL((vector_add_div<<<gridSize, blockSize>>>(d_a2, d_b2, d_c2, n)));

    SAFE_CALL(cudaDeviceSynchronize());
    SAFE_CALL(cudaEventRecord(stop2));

    SAFE_CALL(cudaMemcpy(&h_c2[0], d_c2, size, cudaMemcpyDeviceToHost));

    cudaEventSynchronize(stop2);
    float milliseconds_div = 0;
    cudaEventElapsedTime(&milliseconds_div, start2, stop2);

    cout << "CUDA time " << milliseconds << '\n';
    cout << "CUDA time div " << milliseconds_div << '\n';

    cudaFree(d_a2);
    cudaFree(d_b2);
    cudaFree(d_c2);
}
