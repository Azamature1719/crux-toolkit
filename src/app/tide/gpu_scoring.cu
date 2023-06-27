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
     cudaSetDevice(0);
      
     size_t free, total;
     cudaMemGetInfo(&free, &total);

     // Form a single vector 
     std::vector<int> peptides_to_transfer;
     for(auto pep_iter = peptides.begin(); pep_iter != peptides.end(); pep_iter++){
       copy((*pep_iter).begin(), (*pep_iter).end(), back_inserter(peptides_to_transfer));
     }

     for(int i = 0; i < 100; ++i){
	std::cout << "\nPep " << i << " : "<< peptides_to_transfer[i];
     }     

     cudaError_t err = cudaMalloc((void **)&d_peptides, peptides_to_transfer.size() * sizeof(int));
     cudaMemcpy(d_peptides, peptides_to_transfer.data(), peptides_to_transfer.size() * sizeof(int), cudaMemcpyHostToDevice);

     pep_num = peptides.size();
     pep_length = peptides[0].size();
   } 
 }

 std::vector<int> applyScoring(const int *cache, unsigned int cache_size){

   size_t block_size = 32;
   size_t grid_size = (pep_num + 1) / block_size + 1;

   // Allocate mem for cache peptides and storing results
   int *d_cache, *d_result;
   
   std::cout << "\nCACHE_SIZE: " << cache_size;
   cudaError_t err = cudaMalloc((void **)&d_cache, cache_size * sizeof(int));
   std::cout << "\nMALLOC CACHE: " << err;
   for(int i = 0, j = 0; i < 100; ++j){
	if(cache[j] != 0){
		std::cout << "\nCACHE " << j << " : "<< cache[j];
		++i;
	}
   }     

   // Copy cache peptides to GPU
   err = cudaMemcpy(d_cache, cache, cache_size * sizeof(int), cudaMemcpyHostToDevice);
   std::cout << "\nMemcpy CACHE: " << err;

   err = cudaMalloc((void **)&d_result, pep_num * sizeof(int));
   std::cout << "\nMALLOC result: " << err;
   
   std::cout << "\nPEP_LEN: " << pep_length << "\nPEP_NUM: " << pep_num;
   // Run scoring function on GPU
   score <<<block_size, grid_size>>>(d_peptides, d_cache, d_result, pep_length);
  
   // Get results
   std::vector<int> result(pep_num);
   err = cudaMemcpy(result.data(), d_result, pep_num * sizeof(int), cudaMemcpyDeviceToHost);
   std::cout << "\nMemcpy result: " << err;

   cudaFree(d_cache);
   cudaFree(d_peptides);
   cudaFree(d_result);

   return result;
}
