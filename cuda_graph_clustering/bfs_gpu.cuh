#pragma once
#include <vector>
#include <cuda_runtime.h>
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void gpu_bfs_wrapper(long long *_outgoing_ptrs, int *_outgoing_ids, int _vertices_count, long long _edges_count,
                     int _source_vertex, int *_levels, int *_frontier_ids, int *_bunched_ids, int *_low_ids,
                     const std::vector<cudaStream_t>& Stream);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
