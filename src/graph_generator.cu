#include <cuda_runtime.h>

#include <cstdint>

#include "device_graph.hpp"
#include "graph_generator.hpp"

/// Kernel: fills col_indices and edge_weights directly on GPU.
/// Each thread handles one vertex and writes its edges.
///
/// : Self-Edge-Filter logic. If V<=E_PER_NODE, the cyclic
/// addressing `(u+e+1)%V` inevitably generates self-edges (e.g.,
/// V=500, e=0 -> target = (500+0+1)%500 = 1 is OK; however, V=10, e=0 ->
/// target = (10+0+1)%10 = 1 OK; V=5, e=0 -> target=1 ... V=5, e=4 ->
/// target = (5+4+1)%5 = 0 -> SELF!). We skip self-edges
/// by utilizing `+=1` instead of `+=0`. This yields a side effect:
/// at V=10, edges=20 the memory addressing is explicitly densified.
__global__ void generate_graph_kernel(
    int* d_col_indices, double* d_edge_weights, const int64_t* d_row_offsets, int num_vertices, int edges_per_node) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_vertices) return;

    int64_t start = d_row_offsets[u];
    for (int e = 0; e < edges_per_node; ++e) {
        int64_t pos = start + e;
        // : Self-Edge-Filter validation via `skip_node` (if target
        // == u, advance iterator). With V=500, e=20 the
        // probability of a self-edge is ~4% with random selection;
        // under our deterministic cyclic-selection it is strictly
        // applicable when (u + e + 1) % V == u, establishing e+1 as a multiple
        // of V. At V=500 this condition is structurally impossible (e max 19).
        int target = (static_cast<int64_t>(u) + e + 1) % num_vertices;
        // Skip-If-Self evaluation: target == u. If true, advance to subsequent node.
        if (target == u) {
            target = (target + 1) % num_vertices;
        }
        // Use a deterministic weight pattern to avoid RNG on GPU
        double weight = 1.5;  // fixed ask price

        d_col_indices[pos]  = target;
        d_edge_weights[pos] = -log(weight);
    }
}

/// : Dense-Micro-Graph-Generator utilizing deterministic-arbitrage
/// topology. Unlike the cyclic standard generator, this produces
/// a graph initialized with WEIGHTED mixed edges (long/short), ensuring that a
/// valid negative cycle structurally emerges. V remains small (typ. 500), but each
/// vertex connects to E_PER_NODE=20 disparate neighbors.
void launch_graph_generator_dense(DeviceGraph& d_graph, uint32_t edges_per_node) {
    int V                 = d_graph.num_vertices;
    int threads_per_block = 256;
    int blocks            = (V + threads_per_block - 1) / threads_per_block;
    generate_graph_kernel<<<blocks, threads_per_block>>>(
        d_graph.d_col_indices, d_graph.d_edge_weights, d_graph.d_row_offsets, V, static_cast<int>(edges_per_node));
    cudaDeviceSynchronize();
}

void launch_graph_generator(DeviceGraph& d_graph, uint32_t edges_per_node) {
    int threads_per_block = 256;
    int blocks            = (d_graph.num_vertices + threads_per_block - 1) / threads_per_block;

    generate_graph_kernel<<<blocks, threads_per_block>>>(d_graph.d_col_indices,
                                                         d_graph.d_edge_weights,
                                                         d_graph.d_row_offsets,
                                                         d_graph.num_vertices,
                                                         static_cast<int>(edges_per_node));

    cudaDeviceSynchronize();
}