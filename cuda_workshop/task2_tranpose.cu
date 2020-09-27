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

const double ERR = 1e-6;
__global__ void transpose_matrix(double* init, double* transposed, int n) {
    int idx = blockIdx.x*gridDim.x + threadIdx.x;
    int idy = blockIdx.y*gridDim.y + threadIdx.y;
    if (idx < n && idy < n) {
        transposed[idx*n + idy] = init[idy*n + idx];
    }
}

void cpu_transpose(const vector<double>& init, vector<double>& transposed, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            transposed[i*n + j] = init[j*n + i];
        }
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

int main() {
    int n = 1000;
    size_t size = n*n*sizeof(double);
    vector<double> h_a(n*n), h_b(n*n), h_c(n*n);
    srand(time(NULL));
    for (int i = 0; i < n*n; ++i) {
        h_a[i] = i*i;
    }
    auto cpu_start = high_resolution_clock::now();
    cpu_transpose(h_a, h_b, n);
    auto cpu_stop = high_resolution_clock::now();
    duration<double> cpu_time = (cpu_stop - cpu_start)*1e4;

    double *d_a, *d_b;

    SAFE_CALL(cudaMalloc(&d_a, size));
    SAFE_CALL(cudaMalloc(&d_b, size));

    SAFE_CALL(cudaMemcpy(d_a, &h_a[0], size, cudaMemcpyHostToDevice));

    int blockMax = 1024;
    dim3 blockSize(32, blockMax/32, 1);
    int gridx = (n-1)/blockSize.x + 1;
    int gridy = (n-1)/blockSize.y + 1;
    dim3 gridSize(gridx, gridy, 1);

    cout << "blockSize " << blockSize.x << ' ' << blockSize.y << '\n';
    cout << "gridSize "  << gridSize.x << ' ' << gridSize.y << '\n';
    cout << "n_elements "  << n*n << '\n';
    cudaEvent_t start, stop;
    SAFE_CALL(cudaEventCreate(&start));
    SAFE_CALL(cudaEventCreate(&stop));
    SAFE_CALL(cudaEventRecord(start));

    SAFE_KERNEL_CALL((transpose_matrix<<<gridSize, blockSize>>>(d_a, d_b, n)));

    SAFE_CALL(cudaDeviceSynchronize());
    SAFE_CALL(cudaEventRecord(stop));

    SAFE_CALL(cudaMemcpy(&h_c[0], d_b, size, cudaMemcpyDeviceToHost));

    cudaEventSynchronize(stop);
    if (CheckAnswers(h_c, h_b)) {
        cout << "Test is passed\n";
    } else {
        cout << "Test is failed\n";
    }
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "CPU time " << cpu_time.count() << '\n';
    cout << "CUDA time " << milliseconds << '\n';
    cudaFree(d_a);
    cudaFree(d_b);
}
