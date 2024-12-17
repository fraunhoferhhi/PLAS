#include <cudnn.h>
#include <curand.h>
#include <iostream>
#include <vector>

void apply_filter(const float *input, float *output, int width, int height,
                  int channels, int kernelSizeX, int kernelSizeY) {
  cudnnHandle_t cudnn;
  cudnnCreate(&cudnn);

  // Create tensor descriptors
  cudnnTensorDescriptor_t inputDesc, outputDesc;
  cudnnCreateTensorDescriptor(&inputDesc);
  cudnnCreateTensorDescriptor(&outputDesc);

  // Set tensor descriptors for multi-channel input
  cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1,
                             channels, height, width);
  cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1,
                             channels, height, width);

  // Create convolution descriptor
  cudnnConvolutionDescriptor_t convDesc;
  cudnnCreateConvolutionDescriptor(&convDesc);
  cudnnSetConvolution2dDescriptor(convDesc, kernelSizeX / 2, kernelSizeY / 2, 1,
                                  1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);

  // Set number of groups equal to number of channels for layerwise operation
  cudnnSetConvolutionGroupCount(convDesc, channels);

  // Create filter descriptor
  cudnnFilterDescriptor_t filterDesc;
  cudnnCreateFilterDescriptor(&filterDesc);
  cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NHWC,
                             channels, 1, kernelSizeX, kernelSizeY);

  // Allocate memory for the kernel
  float *d_kernel;
  // Note: Each channel gets its own kernel
  cudaMalloc(&d_kernel, channels * kernelSizeX * kernelSizeY * sizeof(float));

  // Initialize the kernel with box blur values
  float kernelValue = 1.0f / (kernelSizeX * kernelSizeY);
  std::vector<float> h_kernel(channels * kernelSizeX * kernelSizeY, kernelValue);
  cudaMemcpy(d_kernel, h_kernel.data(),
             channels * kernelSizeX * kernelSizeY * sizeof(float),
             cudaMemcpyHostToDevice);

  // Get convolution algorithm
  cudnnConvolutionFwdAlgo_t algo;
  int returnedAlgoCount;
  cudnnConvolutionFwdAlgoPerf_t perfResults;
  cudnnFindConvolutionForwardAlgorithm(cudnn, inputDesc, filterDesc, convDesc,
                                      outputDesc, 1, &returnedAlgoCount, &perfResults);
  algo = perfResults.algo;

  // Get workspace size
  size_t workspaceSize;
  cudnnGetConvolutionForwardWorkspaceSize(
      cudnn, inputDesc, filterDesc, convDesc, outputDesc, algo, &workspaceSize);

  // Allocate workspace
  void *d_workspace;
  cudaMalloc(&d_workspace, workspaceSize);

  // Perform the convolution
  const float alpha = 1.0f, beta = 0.0f;
  cudnnConvolutionForward(cudnn, &alpha, inputDesc, input, filterDesc, d_kernel,
                          convDesc, algo, d_workspace, workspaceSize, &beta,
                          outputDesc, output);

  // Cleanup
  cudaFree(d_workspace);
  cudaFree(d_kernel);
  cudnnDestroyTensorDescriptor(inputDesc);
  cudnnDestroyTensorDescriptor(outputDesc);
  cudnnDestroyFilterDescriptor(filterDesc);
  cudnnDestroyConvolutionDescriptor(convDesc);
  cudnnDestroy(cudnn);
}

// int main() {
//   // Example usage
//   int width = 1024, height = 768, channels = 10, kernelSize = 301;
//   float *d_input, *d_output;
//   cudaMalloc(&d_input, width * height * channels * sizeof(float));
//   cudaMalloc(&d_output, width * height * channels * sizeof(float));
  
//   // Create random number generator
//   curandGenerator_t gen;
//   curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  
//   // Set random seed
//   curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
  
//   // Fill input with random numbers from standard normal distribution
//   curandGenerateNormal(gen, d_input, width * height * channels, 0.0f, 1.0f);
  
//   // Cleanup generator
//   curandDestroyGenerator(gen);

//   // Call the box blur function
//   apply_filter(d_input, d_output, width, height, channels, kernelSize);

//   // Free resources
//   cudaFree(d_input);
//   cudaFree(d_output);

//   std::cout << "Completed without errors" << std::endl;
// }