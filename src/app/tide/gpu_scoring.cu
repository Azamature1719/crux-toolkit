#include <iostream>
#include "gpu_scoring.cuh"

int *d_peptides;

size_t pep_length;
size_t pep_num;

__global__ void score(int *d_peptides, int *d_cache, int *d_result, size_t pep_length){
  
  int current_peptide = blockDim.x * blockIdx.x + threadIdx.x;
  int result = 0;
  for (size_t i = 0; i < pep_length; ++i) {
      int peak = d_peptides[current_peptide * pep_length + i];
      if(peak != -1){
        result += d_cache[peak];
      }
  }
  d_result[current_peptide] = result;

}

void transferPeptides(std::vector<std::vector<int>> peptides){

   int devices = 0; 
   cudaError_t err = cudaGetDeviceCount(&devices); 

   if (devices > 0 && err == cudaSuccess) 
   { 
     // Set default GPU. If there are several devices, one of them could be set
     cudaSetDevice(0);
      
     // Get all free memory
     // size_t free, total;
     // cudaMemGetInfo(&free, &total);

     // Form a single vector 
     std::vector<int> peptides_to_transfer;
     for(auto pep_iter = peptides.begin(); pep_iter != peptides.end(); pep_iter++){
       copy((*pep_iter).begin(), (*pep_iter).end(), back_inserter(peptides_to_transfer));
     }

     cudaError_t err = cudaMalloc((void **)&d_peptides, peptides_to_transfer.size() * sizeof(int));
     // std::cout << "\nPeptides malloc error: " << err;
     err = cudaMemcpy(d_peptides, peptides_to_transfer.data(), peptides_to_transfer.size() * sizeof(int), cudaMemcpyHostToDevice);
     // std::cout << "\nPeptides memcpy error: " << err;
  
     // Set a number of transmitted peptides and a single peptide's length
     pep_num = peptides.size();
     pep_length = peptides[0].size();
   } 
 }

 std::vector<int> applyScoring(const int *cache, unsigned int cache_size){
   
   // Configure GPU thread grid parameters. 32 is a number of threads executed in a single warp
   size_t block_size = 32;
   size_t grid_size = (pep_num + 1) / block_size + 1;

   // Allocate mem for cache peptides
   int *d_cache, *d_result;
   cudaError_t err = cudaMalloc((void **)&d_cache, cache_size * sizeof(int));

   // Copy cache peptides to GPU
   err = cudaMemcpy(d_cache, cache, cache_size * sizeof(int), cudaMemcpyHostToDevice);
   // std::cout << "\nMemcpy cache error: " << err;

   err = cudaMalloc((void **)&d_result, pep_num * sizeof(int));
   // std::cout << "\nMalloc result error: " << err;
   
   // Run scoring function on GPU
   score <<<block_size, grid_size>>>(d_peptides, d_cache, d_result, pep_length);
  
   // Get results from GPU
   std::vector<int> result(pep_num);
   err = cudaMemcpy(result.data(), d_result, pep_num * sizeof(int), cudaMemcpyDeviceToHost);
   // std::cout << "\nMemcpy result error: " << err;

   cudaFree(d_cache);
   cudaFree(d_peptides);
   cudaFree(d_result);

   return result;
}
