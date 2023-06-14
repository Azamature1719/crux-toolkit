#include <iostream>
#include "gpu_scoring.cuh"

unsigned int *d_peaks;
int *d_cache;
int *d_result;

size_t peaks_size;
size_t pep_num;

__global__ void score(unsigned int *d_peaks, int *d_cache, int *d_result, size_t peaks_size){
  d_result[0] = 0;
  for(size_t i = 0; i < peaks_size; ++i){
    d_result[0] += d_cache[d_peaks[i]];
  }
  
  // -- Align the peptides -- 
  
  // int id = threadIdx.x;
  // d_result[id] = 0;
  // for(size_t i = 0; i < peaks_size; ++i){
  //   d_result[id] += d_cache[d_peaks[i]];
  // }
}

void transferPeaks(unsigned int deviceNum, std::vector<unsigned int> peaks, size_t _pep_num){

  int devices = 0; 
  cudaError_t err = cudaGetDeviceCount(&devices); 

  std::cout << "\nPEAKS: " << "SIZE: " << peaks.size() << "\n";
  for(size_t i = 0; i < 10; ++i){
    std::cout << peaks[i] << "\n";
  }

  if (devices > 0 && err == cudaSuccess) 
  { 
    cudaSetDevice(deviceNum);
      
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    cudaMalloc((void **)&d_peaks, free);
    cudaMemcpy(d_peaks, peaks.data(), peaks.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);

    peaks_size = peaks.size();
    pep_num = _pep_num;
  } 
}

std::vector<int> applyScoring(size_t warpSize, const int *cache, unsigned int cache_size){

  std::cout << "\nCACHE: " << "SIZE: " << cache_size << "\n";
  for(size_t i = 0; i < 10; ++i){
    std::cout << cache[i] << "\n";
  }

  size_t block_size = warpSize;
  size_t grid_size = (peaks_size + 1) / block_size + 1;

  // transfer cache
  d_cache = (int*)(d_peaks + peaks_size);
  cudaMalloc((void **)&d_cache, cache_size * sizeof(int));
  cudaMemcpy(d_cache, cache, cache_size * sizeof(int), cudaMemcpyHostToDevice);

  // allocate memory for result variable
  std::vector<int> result;

  d_result = (int*)(d_cache + cache_size);
  cudaMalloc((void **)&d_result, pep_num * sizeof(int));

  // score
  score <<<block_size, grid_size>>>(d_peaks, d_cache, d_result, peaks_size);
  cudaMemcpy(result.data(), d_result, pep_num * sizeof(int), cudaMemcpyDeviceToHost);
  
  return result;
}
