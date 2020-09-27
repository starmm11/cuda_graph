#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_error_handling.h"
#include <vector>
#include <stdio.h>
#include <stdlib.h>

using namespace std;


__global__ void naiveReverse(int *d, int *ans, int n)
{
    int i = threadIdx.x;
    ans[i] = d[n-i-1];
}

__global__ void staticReverse(int *d, int n)
{
    __shared__ int array[1024];
    int i = threadIdx.x;
    array[i] = d[i];
    __syncthreads();
    d[i] = array[n-i-1];
}

__global__ void dynamicReverse(int *d, int n)
{
    extern __shared__ int array[];
    int i = threadIdx.x;
    array[i] = d[i];
    __syncthreads();
    d[i] = array[n-i-1];
  /* FIX ME */
}

int main(void)
{
  const int n = 1024; // FIX ME TO max possible size
  //int a[n], r[n], d[n]; // FIX ME TO dynamic arrays if neccesary
  vector<int> a(n), r(n), d(n);
  for (int i = 0; i < n; i++) {
    a[i] = i;
    r[i] = n-i-1;
    d[i] = 0;
  }

  int *d_d, *d_ans;
  cudaMalloc(&d_d, n * sizeof(int));
  cudaMalloc(&d_ans, n * sizeof(int));

    // run version with static shared memory
  SAFE_CALL(cudaMemcpy(d_d, &a[0], n*sizeof(int), cudaMemcpyHostToDevice));
  SAFE_KERNEL_CALL((staticReverse<<<1, n>>>(d_d, n))); // FIX kernel execution params
  SAFE_CALL(cudaMemcpy(&d[0], d_d, n*sizeof(int), cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; i++) {
      if (d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)n", i, i, d[i], r[i]);
  }

  // run dynamic shared memory version
  SAFE_CALL(cudaMemcpy(d_d, &a[0], n*sizeof(int), cudaMemcpyHostToDevice));
  SAFE_KERNEL_CALL((dynamicReverse<<<1,n,n*sizeof(int)>>>(d_d, n)));
  SAFE_CALL(cudaMemcpy(&d[0], d_d, n * sizeof(int), cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; i++) {
      if (d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)n", i, i, d[i], r[i]);
  }

  SAFE_CALL(cudaMemcpy(d_d, &a[0], n*sizeof(int), cudaMemcpyHostToDevice));
  SAFE_KERNEL_CALL((naiveReverse<<<1,n>>>(d_d, d_ans, n)));
  SAFE_CALL(cudaMemcpy(&d[0], d_ans, n * sizeof(int), cudaMemcpyDeviceToHost));

  for (int i = 0; i < n; i++) {
      if (d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)n", i, i, d[i], r[i]);
  }
}
