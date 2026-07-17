#include <cuda_runtime.h>

#include <cstdio>

#include "cuda_test.hpp"
#include "device_graph.hpp"

__global__ void traverse_csr_kernel(const int64_t* row_offsets,
                                    const int*     col_indices,
                                    const double*  edge_weights,
                                    int            num_vertices) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_vertices) return;

    double local_sum = 0.0;
    for (int64_t i = row_offsets[u]; i < row_offsets[u + 1]; ++i) {
        local_sum += edge_weights[i];
    }

    // This block never executes, but prevents the NVCC compiler
    // from optimizing away the loop as "dead code".
    if (local_sum == -12345.6789) {
        printf("Dummy: %f\n", local_sum);
    }
}

void launch_csr_test(const DeviceGraph& d_graph) {
    int threads_per_block = 256;
    int blocks            = (d_graph.num_vertices + threads_per_block - 1) / threads_per_block;

    printf("[Host] Starting kernel with %d blocks and %d threads per block...\n", blocks, threads_per_block);

    traverse_csr_kernel<<<blocks, threads_per_block>>>(
        d_graph.d_row_offsets, d_graph.d_col_indices, d_graph.d_edge_weights, d_graph.num_vertices);

    cudaDeviceSynchronize();
}
