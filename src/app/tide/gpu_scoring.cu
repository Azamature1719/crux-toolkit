#include <iostream>
#include "gpu_scoring.cuh"

// int *d_peptides;
// int *d_cache;

// size_t pep_length;
// size_t pep_num;

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

std::vector<int> applyScoring(std::vector<std::vector<int>> peptides, const int *cache, unsigned int cache_size){

  int devices = 0; 
  cudaError_t err = cudaGetDeviceCount(&devices); 

  if (devices > 0 && err == cudaSuccess) 
  { 
    cudaSetDevice(0);
      
    // size_t free, total;
    // cudaMemGetInfo(&free, &total);

    // Form a single vector 
    std::vector<int> peptides_to_transfer;
    for(auto pep_iter = peptides.begin(); pep_iter != peptides.end(); pep_iter++){
      copy((*pep_iter).begin(), (*pep_iter).end(), back_inserter(peptides_to_transfer));
    }

    int *d_peptides;
    cudaError_t err = cudaMalloc((void **)&d_peptides, peptides_to_transfer.size() * sizeof(int));
   // std::cout << "\nPep allocation: " << err;
    err = cudaMemcpy(d_peptides, peptides_to_transfer.data(), peptides_to_transfer.size() * sizeof(int), cudaMemcpyHostToDevice);
   // std::cout << "\nPeptides memcpy: " << err;

    size_t block_size = 32;
    size_t grid_size = (peptides.size() + 1) / block_size + 1;

   // std::cout << "\nBlock_size = " << block_size << " Grid_size = " << grid_size;

    // transfer cache
    // d_cache = d_peptides + pep_num * pep_length + 1;

    int *d_cache, *d_result;
    err = cudaMalloc((void **)&d_cache, cache_size * sizeof(int));
   //  std::cout << "\nCache malloc: " << err;
    err = cudaMemcpy(d_cache, cache, cache_size * sizeof(int), cudaMemcpyHostToDevice);
   //  std::cout << "\nCache memcpy: " << err;

    // allocate memory for result variable
    //int *d_result = d_cache + cache_size + 1;

    err = cudaMalloc((void **)&d_result, peptides.size() * sizeof(int));
   //  std::cout << "\nResults malloc: " << err;

    // score
   // std::cout << "\nPepLength: " << peptides[0].size();
    score <<<grid_size, block_size>>>(d_peptides, d_cache, d_result, peptides[0].size());
    
    // get results
    // cudaDeviceSynchronize();
    std::vector<int> result(peptides.size());
    int *res = new int[peptides.size()];
    err = cudaMemcpy(res, d_result, peptides.size() * sizeof(int), cudaMemcpyDeviceToHost);


    for(int i = 0; i < peptides.size(); ++i){
      int j = 0;
      int pep_res = 0;
      while(peptides[i][j] != -1){
        pep_res += cache[peptides[i][j]];
        j++;
      }
      std::cout << " Pep_res " << i << " :" << pep_res;
    }

    std::cout << "\n\n";

    for(int i = 0; i < peptides.size(); ++i){
      std::cout << " Res " << i << " :" << res[i];
    }

    // cudaDeviceSynchronize();
    // std::cout << "\nResults got: " << err;

    cudaFree(d_cache);
    cudaFree(d_peptides);
    cudaFree(d_result);

    return result;
  } 
}

// void transferPeptides(std::vector<std::vector<int>> peptides){

//   int devices = 0; 
//   cudaError_t err = cudaGetDeviceCount(&devices); 

//   if (devices > 0 && err == cudaSuccess) 
//   { 
//     cudaSetDevice(0);
      
//     size_t free, total;
//     cudaMemGetInfo(&free, &total);

//     // Form a single vector 
//     std::vector<int> peptides_to_transfer;
//     for(auto pep_iter = peptides.begin(); pep_iter != peptides.end(); pep_iter++){
//       copy((*pep_iter).begin(), (*pep_iter).end(), back_inserter(peptides_to_transfer));
//     }

//     std::cout << "peps allocation";
//     cudaMalloc((void **)&d_peptides, peptides_to_transfer.size() * sizeof(int));
//     cudaMemcpy(d_peptides, peptides_to_transfer.data(), peptides_to_transfer.size() * sizeof(int), cudaMemcpyHostToDevice);
//     std::cout << "peps allocated";

//     pep_num = peptides.size();
//     pep_length = peptides[0].size();
//   } 
// }

// std::vector<int> applyScoring(const int *cache, unsigned int cache_size){

//   size_t block_size = 32;
//   size_t grid_size = (pep_num + 1) / block_size + 1;

//   // transfer cache
//   //  d_cache = d_peptides + pep_num * pep_length + 1;

//   int *d_cache, *d_result;
//   cudaMalloc((void **)&d_cache, cache_size * sizeof(int));
//   cudaMemcpy(d_cache, cache, cache_size * sizeof(int), cudaMemcpyHostToDevice);

//   // allocate memory for result variable
//   //int *d_result = d_cache + cache_size + 1;

//   cudaMalloc((void **)&d_result, pep_num * sizeof(int));

//   // score
//   std::cout << "mem allocated";
//   score <<<block_size, grid_size>>>(d_peptides, d_cache, d_result, pep_length);
  
//   // get results
//   std::vector<int> result;
//   cudaMemcpy(result.data(), d_result, pep_num * sizeof(int), cudaMemcpyDeviceToHost);

//   std::cout << "scored";

//   cudaFree(d_cache);
//   cudaFree(d_peptides);
//   cudaFree(d_result);

//   return result;
// }
