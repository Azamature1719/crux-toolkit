#ifndef GPU_SCORING_CUH
#define GPU_SCORING_CUH

// Getting access to CUDA API for GPU scoring implementation
#include <vector>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

__global__ void score(int *d_peptides, int *d_cache, int *d_result, size_t pep_length); 
void transferPeptides(std::vector<std::vector<int>> peptides); 
std::vector<int> applyScoring(const int *cache, unsigned int cache_size);


#endif GPU_SCORING_CUH
