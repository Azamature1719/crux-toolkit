// Getting access to CUDA API for GPU scoring implementation
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

// Peptides array
unsigned int *d_peptides;