#include "gpu_scoring.cuh"

unsigned int *d_peptides;

void setDeviceProperties(int deviceNum, size_t warpSize, size_t spectrumMatchingOnce){
  int devices = 0; 
  cudaError_t err = cudaGetDeviceCount(&devices); 

  if (devices > 0 && err == cudaSuccess) 
  { 
    cudaSetDevice(deviceNum);
      
    // Get GPU device properties - could be used for memory configuration
    cudaDeviceProp deviceProp; 
    cudaGetDeviceProperties(&deviceProp, deviceNum);

    // Get all free memory on GPU device
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    cudaMalloc((void **)&d_peptides, free);

    // Calculating property values
    size_t block_size = warpSize;
    size_t grid_size = (spectrumMatchingOnce + 1) / block_size + 1;
  } 
  else
  { 
    //carp(CARP_FATAL, "There are no GPU devices");
  } 
}

void transferDataToDevice(std::vector<unsigned int> peptides){
    cudaMemcpy(d_peptides, peptides.data(), peptides.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
}

// scoring function
__global__ void score(){
  
}
