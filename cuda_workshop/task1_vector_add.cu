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
    vector<double> h_a(n), h_b(n), h_c(n);
    //double *h_a = (double *)malloc(size);
    //double *h_b = (double *)malloc(size);
    //double *h_c = (double *)malloc(size);
    vector<double> ans_c(n);
    for (int i = 0; i < n; ++i) {
        h_a[i] = (double)rand()/RAND_MAX;
        h_b[i] = (double)rand()/RAND_MAX;
    }
    auto cpu_start = high_resolution_clock::now();
    for (int i = 0; i < n; ++i) {
        ans_c[i] = h_a[i]+h_b[i];
    }
    auto cpu_stop = high_resolution_clock::now();
    duration<double> cpu_time = (cpu_stop - cpu_start)*1e4;

    double *d_a, *d_b, *d_c;

    SAFE_CALL(cudaMalloc(&d_a, size));
    SAFE_CALL(cudaMalloc(&d_b, size));
    SAFE_CALL(cudaMalloc(&d_c, size));

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
    SAFE_CALL(cudaEventSynchronize(start));

    SAFE_CALL(cudaMemcpy(d_a, &h_a[0], size, cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(d_b, &h_b[0], size, cudaMemcpyHostToDevice));
    SAFE_KERNEL_CALL((vector_add<<<gridSize, blockSize>>>(d_a, d_b, d_c, n)));
    SAFE_CALL(cudaMemcpy(&h_c[0], d_c, size, cudaMemcpyDeviceToHost));

    SAFE_CALL(cudaDeviceSynchronize());
    SAFE_CALL(cudaEventRecord(stop));
    SAFE_CALL(cudaEventSynchronize(stop));

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    if (CheckAnswers(h_c,ans_c)) {
        cout << "Tessed is passed\n";
    } else {
        cout << "Tessed is failed\n";
    }

    cout << "CPU time " << cpu_time.count() << '\n';
    cout << "CUDA time " << milliseconds << '\n';
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
