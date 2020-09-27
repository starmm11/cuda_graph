#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cublas_v2.h>
#include "cuda_error_handling.h"

#define BLOCK_SIZE 32

#ifdef USE_DOUBLES
typedef double data_t;
#else
typedef float data_t;
#endif

template<class _T>
__global__ void gpu_matrix_mult_global(_T *a,
                                       _T *b,
                                       _T *c,
                                       int m,
                                       int n,
                                       int k)
{
    int idy = threadIdx.y + blockIdx.y*blockDim.y;
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    int ida = idy * n; // thread of the idy row in matrix A
    int idb = idx; // thread of the idx column in matrix B
    if (idy < m && idx < k) {
        _T tmp = 0.0;
        for (int i = 0; i < n; ++i) {
            tmp += a[ida + i] * b[idb + n*i];
        }
        c[idx + n*idy] = tmp;
    }
}

template<class _T>
__global__ void gpu_square_matrix_mult(_T *d_a, _T *d_b, _T *d_result, int n)
{
    int idy = threadIdx.y + blockIdx.y*blockDim.y;
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    int ida = blockIdx.y * n * BLOCK_SIZE; // left up corner of submatrix along row in matrix A
    int idb = blockIdx.x * BLOCK_SIZE; // left up corner of submatrix along column in matrix B
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    __shared__ _T as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ _T bs[BLOCK_SIZE][BLOCK_SIZE];
    _T sum = 0;
    for (int i = 0; i < (n-1)/BLOCK_SIZE+1; ++i) {
        if (idy < n && tx+i*BLOCK_SIZE < n) {
            as[ty][tx] = d_a[ida+i*BLOCK_SIZE+tx+n*ty];
        } else {
            as[ty][tx] = 0;
        }

        if (idx < n && ty+i*BLOCK_SIZE < n) {
            bs[ty][tx] = d_b[idb+i*n*BLOCK_SIZE+tx+ty*n];
        } else {
            bs[ty][tx] = 0;
        }
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += as[ty][k]*bs[k][tx];
        }
        __syncthreads();
    }
    if (idx < n && idy < n) {
        d_result[ida+idb+tx+n*ty] = sum;
    }
}

template<class _T>
__global__ void gpu_square_matrix_mult_bank(_T *d_a, _T *d_b, _T *d_result, int n)
{
    int idy = threadIdx.y + blockIdx.y*blockDim.y;
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    int ida = blockIdx.y * n * BLOCK_SIZE; // left up corner of submatrix along row in matrix A
    int idb = blockIdx.x * BLOCK_SIZE; // left up corner of submatrix along column in matrix B
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    __shared__ _T as[BLOCK_SIZE][BLOCK_SIZE+1];
    __shared__ _T bs[BLOCK_SIZE][BLOCK_SIZE+1];
    _T sum = 0;
    for (int i = 0; i < (n-1)/BLOCK_SIZE+1; ++i) {
        if (idy < n && tx+i*BLOCK_SIZE < n) {
            as[ty][tx] = d_a[ida+i*BLOCK_SIZE+tx+n*ty];
        } else {
            as[ty][tx] = 0;
        }

        if (idx < n && ty+i*BLOCK_SIZE < n) {
            bs[ty][tx] = d_b[idb+i*n*BLOCK_SIZE+tx+ty*n];
        } else {
            bs[ty][tx] = 0;
        }
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += as[ty][k]*bs[k][tx];
        }
        __syncthreads();
    }
    if (idx < n && idy < n) {
        d_result[ida+idb+tx+n*ty] = sum;
    }
}

template<class _T>
void cpu_matrix_mult(_T *h_a, _T *h_b, _T *h_result, int m, int n, int k)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            _T tmp = 0.0;
            for (int h = 0; h < n; ++h)
            {
                tmp += h_a[i * n + h] * h_b[h * k + j];
            }
            h_result[i * k + j] = tmp;
        }
    }
}

int main(int argc, char const *argv[])
{
    int m, n, k;
    /* Fixed seed for illustration */
    srand(3333);
    m = atoi(argv[1]);
    n = atoi(argv[1]);
    k = atoi(argv[1]);
    
    // allocate memory in host RAM, h_cc is used to store CPU result
    data_t *h_a, *h_b, *h_c_shared, *h_c_global, *h_cc;
    SAFE_CALL(cudaMallocHost((void **) &h_a, sizeof(data_t)*m*n));
    SAFE_CALL(cudaMallocHost((void **) &h_b, sizeof(data_t)*n*k));
    SAFE_CALL(cudaMallocHost((void **) &h_c_shared, sizeof(data_t)*m*k));
    SAFE_CALL(cudaMallocHost((void **) &h_c_global, sizeof(data_t)*m*k));
    SAFE_CALL(cudaMallocHost((void **) &h_cc, sizeof(data_t)*m*k));
    
    // random initialize matrix A
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            h_a[i * n + j] = rand() % 1024;
        }
    }
    
    // random initialize matrix B
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            h_b[i * k + j] = rand() % 1024;
        }
    }
    
    float gpu_shared_elapsed_time_ms, gpu_shared_bank_time_ms,
          gpu_global_elapsed_time_ms, cpu_elapsed_time_ms;
    
    // some events to count the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // start to count execution time of GPU version
    // Allocate memory space on the device
    data_t *d_a, *d_b, *d_c;
    SAFE_CALL(cudaMalloc((void **) &d_a, sizeof(data_t)*m*n));
    SAFE_CALL(cudaMalloc((void **) &d_b, sizeof(data_t)*n*k));
    SAFE_CALL(cudaMalloc((void **) &d_c, sizeof(data_t)*m*k));
    
    // copy matrix A and B from host to device memory
    SAFE_CALL(cudaMemcpy(d_a, h_a, sizeof(data_t)*m*n, cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(d_b, h_b, sizeof(data_t)*n*k, cudaMemcpyHostToDevice));
    
    // TODO compute grid size
    dim3 blockSize(32, 32, 1);
    dim3 gridSize((n-1)/blockSize.x + 1, (n-1)/blockSize.y + 1, 1);
    printf("blockSize %d %d %d\n", blockSize.x, blockSize.y, blockSize.z);
    printf("gridSize %d %d %d\n", gridSize.x, gridSize.y, gridSize.z);
    cudaEventRecord(start, 0);
    SAFE_KERNEL_CALL((gpu_square_matrix_mult<<<gridSize, blockSize>>>(d_a, d_b, d_c, n)));
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_shared_elapsed_time_ms, start, stop);
    printf("Time elapsed on shared matrix multiplication of %dx%d . %dx%d on GPU: %f ms.\n\n",
           m, n, n, k, gpu_shared_elapsed_time_ms);

    cudaEventRecord(start, 0);
    SAFE_KERNEL_CALL((gpu_square_matrix_mult_bank<<<gridSize, blockSize>>>(d_a, d_b, d_c, n)));
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_shared_bank_time_ms, start, stop);
    printf("Time elapsed on shared bank matrix multiplication of %dx%d . %dx%d on GPU: %f ms.\n\n",
           m, n, n, k, gpu_shared_bank_time_ms);

    SAFE_CALL(cudaMemcpy(h_c_shared, d_c, sizeof(data_t)*m*k, cudaMemcpyDeviceToHost));

    cudaEventRecord(start, 0);
    SAFE_KERNEL_CALL((gpu_matrix_mult_global<<<gridSize, blockSize>>>(d_a, d_b, d_c, m, n, k)));
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_global_elapsed_time_ms, start, stop);
    
    // Transefr results from device to host
    SAFE_CALL(cudaMemcpy(h_c_global, d_c, sizeof(data_t)*m*k, cudaMemcpyDeviceToHost));

    printf("Time elapsed on global matrix multiplication of %dx%d . %dx%d on GPU: %f ms.\n\n",
           m, n, n, k, gpu_global_elapsed_time_ms);
    
    // start the CPU version
    cudaEventRecord(start, 0);
    
    cpu_matrix_mult(h_a, h_b, h_cc, m, n, k);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on CPU: %f ms.\n\n",
           m, n, n, k, cpu_elapsed_time_ms);
    
    // validate results computed by GPU
    int all_ok = 1;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            if(h_cc[i*k + j] != h_c_shared[i*k + j] || h_cc[i*k + j] != h_c_global[i*k + j])
            {
                all_ok = 0;
            }
        }
    }
    
    // roughly compute speedup
    if(all_ok)
    {
        printf("all results are correct!!!, speedup = %f %f\n",
               cpu_elapsed_time_ms / gpu_shared_elapsed_time_ms, cpu_elapsed_time_ms / gpu_global_elapsed_time_ms);
        printf("GFLOP GPU Shared: %f\n", 2*n*n*n/gpu_shared_elapsed_time_ms/1e6);
        printf("GFLOP GPU Global: %f\n", 2*n*n*n/gpu_global_elapsed_time_ms/1e6);
    }
    else
    {
        printf("incorrect results\n");
    }
    
    // free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c_shared);
    cudaFreeHost(h_c_global);
    cudaFreeHost(h_cc);
    
    return 0;
}
