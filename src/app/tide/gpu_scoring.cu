#include "gpu_scoring.cuh"

unsigned int *d_peptides;
unsigned int *d_res;

std::string setDeviceProperties(int deviceNum, size_t warpSize, size_t spectrumMatchingOnce, std::vector<unsigned int> peptides){
  std::string result;
  
  int devices = 0; 
  cudaError_t err = cudaGetDeviceCount(&devices); 

  if (devices > 0 && err == cudaSuccess) 
  { 
    result += std::to_string(devices);
    cudaSetDevice(deviceNum);
      
    // Get GPU device properties - could be used for memory configuration
    // cudaDeviceProp deviceProp; 
    // cudaGetDeviceProperties(&deviceProp, deviceNum);

    // Get all free memory on GPU device
    size_t free, total;

    cudaMemGetInfo(&free, &total);
    result += " FREE: " + std::to_string(free);
    result += " TOTAL: " + std::to_string(total);

    err = cudaMalloc((void **)&d_peptides, peptides.size() * sizeof(unsigned int));
    result += " MALLOC D_PEPTIDES: " + std::to_string(err);

    err = cudaMalloc((void **)&d_res, sizeof(unsigned int));
    result += " MALLOC D_RES: " + std::to_string(err);

    // Calculating property values
    size_t block_size = warpSize;
    size_t grid_size = (spectrumMatchingOnce + 1) / block_size + 1;

    cudaMemcpy(d_peptides, peptides.data(), peptides.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
    score <<<1,1>>>(d_peptides, d_res, peptides.size());

  //  unsigned int *res;
  //  cudaMemcpy(res, d_res, sizeof(unsigned int), cudaMemcpyDeviceToHost);

  //  result += " SCORING: " + res[0];

  } 
  return result;
}

// cudaError_t transferDataToDevice(std::vector<unsigned int> peptides){
//   return cudaMemcpy(d_peptides, peptides.data(), peptides.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
// }

// scoring function
__global__ void score(unsigned int *d_peptides, unsigned int *d_res, size_t peptides_size){
  for(size_t i = 0; i < peptides_size; ++i){
    d_res[0] += d_peptides[i];
  }
}
