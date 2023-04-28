// Getting access to CUDA API for GPU scoring implementation
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

//#include "io/carp.h"

extern unsigned int *d_peptides;

void setDeviceProperties(int deviceNum, size_t warpSize, size_t spectrumMatchingOnce);
void transferDataToDevice(std::vector<unsigned int> peptides);