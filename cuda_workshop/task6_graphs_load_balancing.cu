#include <cfloat>
#include <chrono>
#include <vector>
#include <string>
#include <cuda_profiler_api.h>
#include <iostream>
#include "cuda_error_handling.h"

using namespace std;

void __global__ gather(int *ptrs, int *connections, int *outgoing_ids, int vertices_count, int *data, int *result)
{
    const long long src_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (src_id < vertices_count)
    {
        const int first_edge_ptr = ptrs[src_id];
        const int connections_count = connections[src_id];

        for(register int cur_edge = 0; cur_edge < connections_count; cur_edge++)
        {
            int dst_id = outgoing_ids[first_edge_ptr + cur_edge];
            int val = data[dst_id];
            result[first_edge_ptr + cur_edge] = val;

            // данную программу можно легко переделать во многие графовые алгоритмы, например:
            /* BFS
            int src_level = data[src_id];
            int dst_level = data[dst_id];
            if((src_level == current_level) && (dst_level == UNVISITED_VERTEX))
            {
                data[dst_id] = current_level + 1;
            }
            */

            /* SSSP
            float weight = outgoing_weights[first_edge_ptr + cur_edge];
            float src_weight = data[src_id];
            float dst_weight = data[dst_id];

            if(dst_weight > src_weight + weight)
            {
                data[dst_id] = src_weight + weight;
            }
            */
        }
    }
}

void __global__ gather_opt(int *ptrs, int *connections, int *outgoing_ids,
                           int vertices_count, int *data, int *result)
{
    const long long id = (blockIdx.x*blockDim.x+threadIdx.x);
    const int i = id / 32;
    const int first_edge_ptr = ptrs[i];
    const int connections_count = connections[i];
    int rep = (connections_count/32)+1;
    for (int k = 0; k < rep; ++k) {
        int cur_edge = threadIdx.x%32 + (k*32);
        if (cur_edge < connections_count) {
            int dst_id = outgoing_ids[first_edge_ptr + cur_edge];
            int val = data[dst_id];
            result[first_edge_ptr + cur_edge] = val;
        }
    }
}

void cpu_gather(int *ptrs, int *connections, int *outgoing_ids,
                int vertices_count, int *data, vector<int>& result)
{
    for (int i = 0; i < vertices_count; ++i) {
        const int first_edge_ptr = ptrs[i];
        const int connections_count = connections[i];

        for (int cur_edge = 0; cur_edge < connections_count; cur_edge++)
        {
            int dst_id = outgoing_ids[first_edge_ptr + cur_edge];
            int val = data[dst_id];
            result[first_edge_ptr + cur_edge] = val;
        }
    }
}

bool checkAnswers(const vector<int>& res1, const vector<int>& res2) {
    for (int i = 0; i < res1.size(); ++i) {
        if (res1[i] != res2[i]) {
            printf("%d %d %d are different\n", i, res1[i], res2[i]);
            return false;
        }
    }
    return true;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
    if (argc < 2) {
        cout << "Incorrect number of arguments\n";
        return 0;
    }
    int vertices_count = 1024*1024;

    int *ptrs = new int[vertices_count];
    int *data = new int[vertices_count];
    int *connections = new int[vertices_count];

    int pos = 0;
    double percent = stod(argv[1]);
    srand(time(NULL));
    for(int i = 0; i < vertices_count; i++) // TODO (bonus) граф с несколькими "большими" вершинами
    {
        ptrs[i] = pos;
        if ((double)rand()/RAND_MAX < percent) {
            connections[i] = 1000 + rand()%9000;
            pos += connections[i];
        } else {
            connections[i] = 16 + rand()%32;
            pos += connections[i];
        }
        data[i] = rand();
    }

    int edges_count = pos;
    srand(time(NULL));
    printf("Edges count %d\n", edges_count);
    int *outgoing_ids = new int[edges_count];
    for(int i = 0; i < edges_count; i++)
    {
        outgoing_ids[i] = rand()%vertices_count;
    }

    int *dev_ptrs; int *dev_connections; int *dev_outgoing_ids; int *dev_data; int *dev_result;
    cudaMalloc((void**)&dev_ptrs, vertices_count*sizeof(int));
    cudaMalloc((void**)&dev_connections, vertices_count*sizeof(int));
    cudaMalloc((void**)&dev_data, vertices_count*sizeof(int));

    cudaMalloc((void**)&dev_outgoing_ids, edges_count*sizeof(int));
    cudaMalloc((void**)&dev_result, edges_count*sizeof(int));

    SAFE_CALL(cudaMemcpy(dev_ptrs, ptrs, vertices_count * sizeof(int), cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(dev_connections, connections, vertices_count * sizeof(int), cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(dev_data, data, vertices_count * sizeof(int), cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(dev_outgoing_ids, outgoing_ids, edges_count * sizeof(int), cudaMemcpyHostToDevice));

    dim3 compute_threads(1024);
    dim3 compute_blocks(32*((vertices_count - 1) / compute_threads.x + 1));

    for(int i = 0; i < 5; i++)
    {
        auto start = std::chrono::steady_clock::now();
        SAFE_KERNEL_CALL((gather_opt<<< compute_blocks, compute_threads >>> (dev_ptrs, dev_connections, dev_outgoing_ids, vertices_count, dev_data, dev_result)));
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        cout << "time: " << (elapsed_seconds.count())*1000.0 << " ms" << endl;
        cout << "bandwidth: " << 3.0*sizeof(int)*edges_count/((elapsed_seconds.count())*1e9) << " GB/s" << endl << endl;
    }

    for(int i = 0; i < 5; i++)
    {
        auto start = std::chrono::steady_clock::now();
        SAFE_KERNEL_CALL((gather<<< compute_blocks, compute_threads >>> (dev_ptrs, dev_connections, dev_outgoing_ids, vertices_count, dev_data, dev_result)));
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        cout << "time: " << (elapsed_seconds.count())*1000.0 << " ms" << endl;
        cout << "bandwidth: " << 3.0*sizeof(int)*edges_count/((elapsed_seconds.count())*1e9) << " GB/s" << endl << endl;
    }

    vector<int> gpu_result(edges_count);
    SAFE_CALL(cudaMemcpy(&gpu_result[0], dev_result, edges_count * sizeof(int), cudaMemcpyDeviceToHost));

    vector<int> cpu_result(edges_count);
    cpu_gather(ptrs, connections, outgoing_ids, vertices_count, data, cpu_result);

    if (checkAnswers(gpu_result, cpu_result)) {
        printf("Test is passed\n");
    } else {
        printf("Test is failed\n");
    }
    // TODO какие 3 недостатка у текущей версии ядра?

    // TODO отпрофилировать текущую версию, сделать выводы о её производитлеьности

    // TODO сделать оптимизированную версию ядра

    // TODO (bonus) реализовать базовую версию BFS алгоритма (выделить структуры данных и реализовать сам алгоритм)

    cudaFree(dev_data);
    cudaFree(dev_ptrs);
    cudaFree(dev_connections);
    cudaFree(dev_result);
    cudaFree(dev_outgoing_ids);
    
    delete[]data;
    delete[]ptrs;
    delete[]outgoing_ids;
    delete[]connections;
    
    return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
