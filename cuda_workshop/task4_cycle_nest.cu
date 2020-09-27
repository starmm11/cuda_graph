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

__global__ void nest_add(double* d_a, const double* d_b, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    int idz = threadIdx.z + blockDim.z * blockIdx.z;
    int id = (idz*n+idy)*n+idx;
    int id_inc = (idz*n+idy+1)*n+idx;
    if (id_inc < n*n*n) {
        d_a[id] = d_a[id_inc] + d_b[id_inc];
    }
}

void cpu_nest_add (vector<vector<vector<double> > >& h_a,
                   const vector<vector<vector<double> > >& h_b, int n)
{
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n-1; ++j) {
            for (int k = 0; k < n; ++k) {
                h_a[i][j][k] = h_a[i][j+1][k] + h_b[i][j+1][k];
            }
        }
    }
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
    size_t size = n*n*n * sizeof(double);

    // initialize vectors n^3
    vector<double> h_a(n*n*n), h_b(n*n*n);

    for (int i = 0; i < n*n*n; ++i) {
        h_a[i] = (double)rand()/RAND_MAX;
        h_b[i] = (double)rand()/RAND_MAX;
    }
    // initialize vector of matrices for direct cpu nest loop
    vector<vector<vector<double> > > m_a(n, vector<vector<double>>(n, vector<double>(n)));
    vector<vector<vector<double> > > m_b(n, vector<vector<double>>(n, vector<double>(n)));

    // copy values from 1D array to 3D arrays
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                m_a[i][j][k] = h_a[(i*n+j)*n+k];
                m_b[i][j][k] = h_b[(i*n+j)*n+k];
            }
        }
    }

    // Start cpu nest loop
    auto cpu_start = high_resolution_clock::now();
    cpu_nest_add(m_a, m_b, n);
    auto cpu_stop = high_resolution_clock::now();
    duration<double> cpu_time = (cpu_stop - cpu_start)*1e4;

    // Allocate CUDA 1D arrays
    double *d_a, *d_b;

    SAFE_CALL(cudaMalloc(&d_a, size));
    SAFE_CALL(cudaMalloc(&d_b, size));

    // Copy values from Device 1D arrays
    SAFE_CALL(cudaMemcpy(d_a, &h_a[0], size, cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(d_b, &h_b[0], size, cudaMemcpyHostToDevice));

    dim3 blockSize(16,16,4);
    dim3 gridSize((n-1)/blockSize.x + 1,(n-1)/blockSize.y + 1,(n-1)/blockSize.z + 1);
    printf("blockSize %d %d %d \n", blockSize.x, blockSize.y, blockSize.z);
    printf("gridSize %d %d %d \n", gridSize.x, gridSize.y, gridSize.z);
    printf("n_elements %d\n", n*n*n);

    cudaEvent_t start, stop;
    SAFE_CALL(cudaEventCreate(&start));
    SAFE_CALL(cudaEventCreate(&stop));
    SAFE_CALL(cudaEventRecord(start));

    SAFE_KERNEL_CALL((nest_add<<<gridSize, blockSize>>>(d_a, d_b, n)));

    SAFE_CALL(cudaDeviceSynchronize());
    SAFE_CALL(cudaEventRecord(stop));

    // Initialize device vector for Cuda transfer
    vector<double> h_ans(n*n*n);
    SAFE_CALL(cudaMemcpy(&h_ans[0], d_a, size, cudaMemcpyDeviceToHost));

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << "CPU time " << cpu_time.count() << '\n';
    cout << "CUDA time " << milliseconds << '\n';
    cudaFree(d_a);
    cudaFree(d_b);
}