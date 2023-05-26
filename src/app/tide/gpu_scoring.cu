#include "gpu_scoring.cuh"

unsigned int *d_peaks;
int *d_cache;
int *d_result;

size_t peaks_size;

void transferPeaks(unsigned int deviceNum, std::vector<unsigned int> peaks){

  int devices = 0; 
  cudaError_t err = cudaGetDeviceCount(&devices); 

  if (devices > 0 && err == cudaSuccess) 
  { 
    cudaSetDevice(deviceNum);
      
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    cudaMalloc((void **)&d_peaks, free);
    cudaMemcpy(d_peaks, peaks.data(), peaks.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);

    peaks_size = peaks.size();
  } 
}

void transferCache(size_t warpSize, const int *cache, unsigned int size_cache){

    cudaMalloc((void **)&d_cache, size_cache * sizeof(int));
    cudaMemcpy(d_cache, cache, size_cache * sizeof(int), cudaMemcpyHostToDevice);
}

int applyScoring(){
  
  //size_t block_size = warpSize;
  //size_t grid_size = (peptides.size() + 1) / block_size + 1;

  int result = 0;

  score <<<1,1>>>(d_peaks, d_cache, d_result, peaks_size);
  cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
  
  return result;
}

__global__ void score(unsigned int *d_peaks, int *d_cache, int d_result, size_t peaks_size){

  d_result = 0;

  for(size_t i = 0; i < peaks_size; ++i){
    d_result += d_cache[d_peaks[i]];
  }
}
