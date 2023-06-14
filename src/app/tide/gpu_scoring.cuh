#ifndef GPU_SCORING_CUH
#define GPU_SCORING_CUH

// Getting access to CUDA API for GPU scoring implementation
#include <vector>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

 
__global__ void score(unsigned int *d_peaks, int *d_cache, int *d_result, size_t peaks_size); 
void transferPeaks(unsigned int deviceNum, std::vector<unsigned int> peaks, size_t _pep_num); 
std::vector<int> applyScoring(size_t warpSize, const int *cache, unsigned int cache_size); 
// void transferCache(); 

#endif GPU_SCORING_CUH
