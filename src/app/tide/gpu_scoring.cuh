#ifndef GPU_SCORING_CUH
#define GPU_SCORING_CUH

// Getting access to CUDA API for GPU scoring implementation
#include <vector>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

void transferPeaks(int deviceNum, std::vector<unsigned int> peaks);
void transferCache(size_t warpSize, const int *cache, unsigned int size_cache);
int applyScoring();
__global__ void score(unsigned int *d_peaks, int *d_cache, int *d_res, size_t peaks_size);

#endif GPU_SCORING_CUH
