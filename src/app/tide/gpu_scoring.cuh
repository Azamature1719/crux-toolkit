#ifndef GPU_SCORING_CUH
#define GPU_SCORING_CUH

// Getting access to CUDA API for GPU scoring implementation
#include <vector>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

//#include "io/carp.h"

std::string setDeviceProperties(int deviceNum, size_t warpSize, size_t spectrumMatchingOnce, std::vector<unsigned int> peptides);
// cudaError_t transferDataToDevice(std::vector<unsigned int> peptides);
__global__ void score(unsigned int *d_peptides, unsigned int *d_res, size_t peptides_size);

#endif GPU_SCORING_CUH
