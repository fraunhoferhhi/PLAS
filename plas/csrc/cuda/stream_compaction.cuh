#include <torch/extension.h>
#include <iostream>

#ifndef _STREAM_COMPACTION_KERNEL_H_
#define _STREAM_COMPACTION_KERNEL_H_

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4

// Define this to more rigorously avoid bank conflicts, even at the lower (root) levels of the tree
//#define ZERO_BANK_CONFLICTS 

#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS + (index) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS)
#endif

#ifdef CHECK_BANK_CONFLICTS
#define TEMP(index)   CUT_BANK_CHECKER(temp, index)
#else
#define TEMP(index)   temp[index]
#endif

__global__ void scan_exclusive(int n, int *g_odata, int *g_idata)
{
    // Dynamically allocated shared memory for scan kernels
    extern  __shared__  int temp[];

    int thid = threadIdx.x;

    int ai = thid;
    int bi = thid + (n/2);

    // compute spacing to avoid bank conflicts
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

    // Cache the computational window in shared memory
    TEMP(ai + bankOffsetA) = g_idata[ai]; 
    TEMP(bi + bankOffsetB) = g_idata[bi]; 

    int offset = 1;

    // build the sum in place up the tree
    for (int d = n/2; d > 0; d >>= 1)
    {
        __syncthreads();

        if (thid < d)      
        {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            TEMP(bi) += TEMP(ai);
        }

        offset *= 2;
    }

    // scan back down the tree

    // clear the last element
    if (thid == 0)
    {
        int index = n - 1;
        index += CONFLICT_FREE_OFFSET(index);
        TEMP(index) = 0;
    }   

    // traverse down the tree building the scan in place
    for (int d = 1; d < n; d *= 2)
    {
        offset /= 2;

        __syncthreads();

        if (thid < d)
        {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            int t  = TEMP(ai);
            TEMP(ai) = TEMP(bi);
            TEMP(bi) += t;
        }
    }

    __syncthreads();

    // write results to global memory
    g_odata[ai] = TEMP(ai + bankOffsetA); 
    g_odata[bi] = TEMP(bi + bankOffsetB); 
}

__global__ void scatter(int n, int *input, int *output, int *scatter_indices, int *mask) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n && mask[index]) {
        output[int(scatter_indices[index])] = input[index];
    }
}

at::Tensor filter_less_than(at::Tensor values, int n) {
    at::Tensor flags = (values < n).to(torch::kInt32);
    at::Tensor indices = torch::empty_like(flags);
    const int block_size = 1024;
    const int elements_per_block = 2 * block_size;
    int num_blocks = (n + elements_per_block - 1) / elements_per_block;
    
    // Add error checking for kernel launch
    cudaError_t launch_error = cudaSuccess;
    scan_exclusive<<<num_blocks, block_size>>>(
        n,
        flags.data_ptr<int>(), 
        indices.data_ptr<int>()
    );
    launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        throw std::runtime_error(std::string("Kernel launch failed: ") + cudaGetErrorString(launch_error));
    }

    cudaError_t sync_error = cudaDeviceSynchronize();
    if (sync_error != cudaSuccess) {
        throw std::runtime_error(std::string("cudaDeviceSynchronize failed: ") + cudaGetErrorString(sync_error));
    }

    std::cout << "here" << std::endl;
    
    // Add a check for the indices tensor
    if (!indices.defined() || indices.numel() == 0) {
        throw std::runtime_error("indices tensor is not properly defined or empty");
    }

    // // After "here" is printed
    // std::cout << "First few elements of indices: "
    //           << indices.slice(0, 0, std::min(5, indices.numel())).to(torch::kCPU) << std::endl;

    std::cout << "indices size: " << indices.sizes() << ", type: " << indices.dtype() << std::endl;

    return indices;
}

#endif // _STREAM_COMPACTION_KERNEL_H_
