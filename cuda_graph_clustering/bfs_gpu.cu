#include "cuda_error_hadling.h"
#include "bfs_gpu.cuh"
#include <omp.h>
#include <limits>
#include <iostream>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <cuda.h>
#include <sys/time.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "const.h"

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// init levels
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ init_kernel(int *_levels, int _vertices_count, int _source_vertex)
{
    register const int idx = (blockIdx.x * blockDim.x + threadIdx.x) + blockIdx.y * blockDim.x * gridDim.x;
    
    // все вершины кроме источника еще не посещены
    if (idx < _vertices_count)
        _levels[idx] = UNVISITED_VERTEX;

    _levels[_source_vertex] = 1; // вершина-источник помещается на первый "уровень"
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// main computational algorithm
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ bfs_kernel_old(int *_levels,
                           long long *_outgoing_ptrs,
                           int *_outgoing_ids,
                           int _vertices_count,
                           long long _edges_count,
                           int *_changes,
                           int _current_level)
{
    register const int src_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (src_id < _vertices_count) // для всех графовых вершин выполнить следующее
    {
        if(_levels[src_id] == _current_level) // если графовая вершина принадлежит текущему (ранее посещенному уровню)
        {
            const long long edge_start = _outgoing_ptrs[src_id]; // получаем положение первого ребра вершины
            const int connections_count = _outgoing_ptrs[src_id + 1] - _outgoing_ptrs[src_id]; // получаем число смежных ребер вершины

            for(int edge_pos = 0; edge_pos < connections_count; edge_pos++) // для каждого смежного ребра делаем:
            {
                int dst_id = _outgoing_ids[edge_start + edge_pos]; // загружаем ID напарвляющей вершины ребра

                if (_levels[dst_id] == UNVISITED_VERTEX) // если направляющая вершина - не посещенная
                {
                    _levels[dst_id] = _current_level + 1; // то помечаем её следующим уровнем
                    _changes[0] = 1;
                }
            }
        }
    }

}

void __global__ bfs_kernel(int *_frontier_ids,
                           int _frontier_ids_size,
                           int *_levels,
                           long long *_outgoing_ptrs,
                           int *_outgoing_ids,
                           int _vertices_count,
                           long long _edges_count,
                           int *_changes,
                           int _current_level,
                           int threads_per_vertex)
{
    const long long src_id = (blockIdx.x*blockDim.x+threadIdx.x);
    const int i = src_id / threads_per_vertex;
    if (i < _frontier_ids_size) // для всех графовых вершин выполнить следующее
    {
        int idx = _frontier_ids[i];
        const int first_edge_ptr = _outgoing_ptrs[idx];
        const int connections_count = _outgoing_ptrs[idx + 1] - _outgoing_ptrs[idx];
        for (int cur_edge = threadIdx.x % threads_per_vertex; cur_edge < connections_count; cur_edge+=threads_per_vertex) {
            int dst_id = _outgoing_ids[first_edge_ptr + cur_edge];
            if (_levels[dst_id] == UNVISITED_VERTEX) // если направляющая вершина - не посещенная
            {
                _levels[dst_id] = _current_level + 1; // то помечаем её следующим уровнем
                _changes[0] = 1;
            }
        }
    }
}



////////////////////////////////
//// Find indices of level /////
////////////////////////////////

void __global__ copy_current_level_frontier(int * _low_ids,
                                            int _low_threshold,
                                            int * _bunched_ids,
                                            int _high_threshold,
                                            int * _frontier_ids,
                                            long long * _outgoing_ptrs,
                                            int * _levels,
                                            int _current_level,
                                            int _vertices_counts)
{
    register const int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx < _vertices_counts) {
        if (_levels[idx] == _current_level) {
            long long connections = _outgoing_ptrs[idx+1]-_outgoing_ptrs[idx];
            if (connections > _high_threshold) {
                _low_ids[idx] = -1;
                _bunched_ids[idx] = idx;
                _frontier_ids[idx] = -1;
            } else if (connections < _low_threshold){
                _low_ids[idx] = idx;
                _bunched_ids[idx] = -1;
                _frontier_ids[idx] = -1;
            } else {
                _low_ids[idx] = -1;
                _bunched_ids[idx] = -1;
                _frontier_ids[idx] = idx;
            }
        } else {
            _low_ids[idx] = -1;
            _bunched_ids[idx] = -1;
            _frontier_ids[idx] = -1;
        }
    }
}

struct is_inactive
{
    __host__ __device__
    bool operator()(const int x)
    {
        return (x == -1);
    }
};

tuple<int, int, int> generate_frontier(int * _low_ids,
                                       int _low_threshold,
                                       int * _bunched_ids,
                                       int _high_threshold,
                                       int * _frontier_ids,
                                       long long * _outgoing_ptrs,
                                       int * _levels,
                                       int _current_level,
                                       int _vertices_counts)
{
    int blockDim = 1024;
    int gridDim = (_vertices_counts-1)/blockDim + 1;
    SAFE_KERNEL_CALL((copy_current_level_frontier<<<gridDim, blockDim>>>
                                    (_low_ids, _low_threshold, _bunched_ids, _high_threshold,
                                     _frontier_ids, _outgoing_ptrs, _levels, _current_level,
                                     _vertices_counts)));
    int *new_end = remove_if(thrust::device, _frontier_ids, _frontier_ids+_vertices_counts,is_inactive());
    int frontier_ids_size = new_end - _frontier_ids;
    new_end = remove_if(thrust::device, _bunched_ids, _bunched_ids+_vertices_counts, is_inactive());
    int bunched_ids_size = new_end - _bunched_ids;
    new_end = remove_if(thrust::device, _low_ids, _low_ids+_vertices_counts, is_inactive());
    int low_ids_size = new_end - _low_ids;
    return make_tuple(frontier_ids_size, bunched_ids_size, low_ids_size);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// single GPU implememntation
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void gpu_bfs_wrapper(long long *_outgoing_ptrs, int *_outgoing_ids, int _vertices_count,
                     long long _edges_count, int _source_vertex, int *_levels,
                     int *_frontier_ids, int *_bunched_ids, int *_low_ids,
                     const vector<cudaStream_t>& Stream)
{
    dim3 init_threads(1024);
    dim3 init_blocks((_vertices_count - 1) / init_threads.x + 1);
    
    // call init kernel
    SAFE_KERNEL_CALL((init_kernel <<< init_blocks, init_threads >>> (_levels, _vertices_count, _source_vertex)));

    // device variable to stop iterations, for each source vertex
    int *changes;
    SAFE_CALL(cudaMallocManaged((void**)&changes, sizeof(int)));
    
    // set grid size
    dim3 compute_threads(1024);
    dim3 compute_blocks_low(((_vertices_count - 1) / compute_threads.x + 1)/2);
    dim3 compute_blocks_front;
    dim3 compute_blocks_bunched(4*((_vertices_count - 1) / compute_threads.x + 1));
    int current_level = 1;

    using namespace thrust;
    using namespace thrust::placeholders;
    int low_threshold = 8; // less then 10 edges
    int high_threshold = 1024; // over 400 edges
    // compute shortest paths
    do
    {
        changes[0] = 0;
        std::tuple<int,int,int> ids_sizes = generate_frontier(_low_ids, low_threshold,
                                                            _bunched_ids, high_threshold, _frontier_ids,
                                                            _outgoing_ptrs, _levels, current_level,
                                                            _vertices_count);

        int frontier_ids_size = std::get<0>(ids_sizes);
        int bunched_ids_size = std::get<1>(ids_sizes);
        int low_ids_size = std::get<2>(ids_sizes);

        //cout << frontier_ids_size << ' ' << bunched_ids_size << ' ' << low_ids_size << "\n";

        if (low_ids_size != 0) {
            bfs_kernel <<< compute_blocks_low, compute_threads, 0, Stream[2] >>>
                    (_low_ids, low_ids_size, _levels, _outgoing_ptrs, _outgoing_ids, _vertices_count, _edges_count,
                     changes, current_level, 1);
        }
        if (bunched_ids_size != 0) {
            bfs_kernel <<< compute_blocks_bunched, compute_threads, 0, Stream[1] >>>
                    (_bunched_ids, bunched_ids_size, _levels, _outgoing_ptrs, _outgoing_ids, _vertices_count, _edges_count,
                     changes, current_level, 1024);
        }

        if (low_ids_size != 0 || bunched_ids_size != 0) {
            compute_blocks_front.x = 24*((_vertices_count - 1) / compute_threads.x + 1);
        } else {
            compute_blocks_front.x = 32*((_vertices_count - 1) / compute_threads.x + 1);
        }

        bfs_kernel <<< compute_blocks_front, compute_threads, 0, Stream[0] >>>
                (_frontier_ids, frontier_ids_size, _levels, _outgoing_ptrs, _outgoing_ids, _vertices_count, _edges_count,
                 changes, current_level, 32);

        cudaDeviceSynchronize();
//        for ( int i = 0; i < 3; i++ ) {
//            SAFE_CALL(cudaStreamDestroy(Stream[i]));
//        }
        current_level++;
    }
    while(changes[0] > 0);


    SAFE_CALL(cudaFree(changes));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
