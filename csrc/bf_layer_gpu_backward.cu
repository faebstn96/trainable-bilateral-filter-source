/*
Author: Fabian Wagner
Contact: fabian.wagner@fau.de

This file contains modified code, originally published under the following licence:
Copyright 2020 - 2021 MONAI Consortium
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <cuda.h>
#include <cuda_runtime.h>

#include "bilateral.h"
#include "../utils/cuda_error_check.h"
#include "../utils/meta_macros.h"

__constant__ int cBatchStrideBack;
__constant__ int cColorStrideBack;

__constant__ int cSizesBack[3];
__constant__ int cStridesBack[3];

__constant__ int cKernelSizesBack[3];
__constant__ int cHalfWindowSize_arrBack[3];
__constant__ float cGaussianKernel_xBack[256];
__constant__ float cGaussianKernel_yBack[256];
__constant__ float cGaussianKernel_zBack[256];
__constant__ float cXDistanceSquaredBack[256];
__constant__ float cYDistanceSquaredBack[256];
__constant__ float cZDistanceSquaredBack[256];
__constant__ float cColorExponentConstantBack;
__constant__ float cSigma_xBack;
__constant__ float cSigma_yBack;
__constant__ float cSigma_zBack;
__constant__ float cColorSigmaBack;


template <typename scalar_t, int C>
__global__ void BilateralFilterCudaKernel3DBackward(scalar_t* gradientInputTensor,
                                                    scalar_t* gradientOutputTensor,
                                                    scalar_t* inputTensor,
                                                    scalar_t* outputTensor,
                                                    scalar_t* outputWeightsTensor,
                                                    scalar_t* dO_dx_ki) {
  int homeOffset = blockIdx.x * blockDim.x + threadIdx.x;
  int batchOffset = blockIdx.y * cBatchStrideBack;

  if (homeOffset >= cColorStrideBack)
    return;

  int homeX = homeOffset / cStridesBack[0];
  int homeY = (homeOffset - homeX * cStridesBack[0]) / cStridesBack[1];
  int homeZ = (homeOffset - homeX * cStridesBack[0] - homeY * cStridesBack[1]) / cStridesBack[2];
  int homeIndex[] = {homeX, homeY, homeZ};

  // Zero kernel aggregates.
  scalar_t valueSum = 0;
  scalar_t weightSum = 0;

  for (int kernelX = 0; kernelX < cKernelSizesBack[0]; kernelX++) {
    int neighbourX = max(0, min(homeX + (kernelX - cHalfWindowSize_arrBack[0]), cSizesBack[0] - 1));
    scalar_t gaussianX = cGaussianKernel_xBack[kernelX];

    for (int kernelY = 0; kernelY < cKernelSizesBack[1]; kernelY++) {
      int neighbourY = max(0, min(homeY + (kernelY - cHalfWindowSize_arrBack[1]), cSizesBack[1] - 1));
      scalar_t gaussianY = cGaussianKernel_yBack[kernelY];

      for (int kernelZ = 0; kernelZ < cKernelSizesBack[2]; kernelZ++) {
        int neighbourZ = max(0, min(homeZ + (kernelZ - cHalfWindowSize_arrBack[2]), cSizesBack[2] - 1));
        scalar_t gaussianZ = cGaussianKernel_zBack[kernelZ];

        int neighbourOffset = neighbourX * cStridesBack[0] + neighbourY * cStridesBack[1] + neighbourZ;

        bool flagNotClamped = true;
        int kernelIndex[] = {kernelX, kernelY, kernelZ};
        int dimensions = 3;  // Must equal the number of spatial dimensions.

        for (int i = 0; i < dimensions; i++) {
            int HalfWindowSizeBack = cHalfWindowSize_arrBack[i];  // Define constant memory as new variable here (!!), otherwise: cudaErrorMisalignedAddress
            int neighbourIndex = homeIndex[i] + kernelIndex[i] - HalfWindowSizeBack;
            int neighbourIndexClamped = min(cSizesBack[i] - 1, max(0, neighbourIndex));
            if (neighbourIndex != neighbourIndexClamped) { flagNotClamped = false; }
        }

        scalar_t colorDistance = 0;
        scalar_t colorDistanceSquared = 0;

#pragma unroll
        for (int c = 0; c < C; c++) {
          scalar_t a = inputTensor[batchOffset + neighbourOffset + c * cColorStrideBack];
          scalar_t b = inputTensor[batchOffset + homeOffset + c * cColorStrideBack];  // Be careful: Here it is (X_k - X_i) and not (X_i - X_q)
          scalar_t diff = a - b;
          colorDistance += diff; // Do not take the absolute value here. Be careful with the signs.
          colorDistanceSquared += diff * diff;
        }

        scalar_t spatialWeight = gaussianX * gaussianY * gaussianZ;
        scalar_t colorWeight = exp(cColorExponentConstantBack * colorDistanceSquared);
        scalar_t totalWeight = spatialWeight * colorWeight;

        // Aggregating values. Only do this if flagNotClamped: Pixels outside the image are disregarded.
        if (flagNotClamped) {
            scalar_t filter_kernel_back;

#pragma unroll
            for (int c = 0; c < C; c++) {
                // Distinguish cases for k!=i (calculation is done here)
                // and k==i (partial derivatives are precalculated).
                // If statement replaces center element of neighborhood/kernel.
                if (kernelX != cHalfWindowSize_arrBack[0] || kernelY != cHalfWindowSize_arrBack[1] ||
                    kernelZ != cHalfWindowSize_arrBack[2]) {

                    filter_kernel_back =
                            -(1 /
                              outputWeightsTensor[batchOffset + neighbourOffset + c * cColorStrideBack]) *
                            outputTensor[batchOffset + neighbourOffset + c * cColorStrideBack] *
                            totalWeight *
                            colorDistance / (cColorSigmaBack * cColorSigmaBack) +
                            (1 /
                             outputWeightsTensor[batchOffset + neighbourOffset + c * cColorStrideBack]) *
                            totalWeight *
                            (1 +
                             inputTensor[batchOffset + homeOffset + c * cColorStrideBack] * colorDistance /
                             (cColorSigmaBack * cColorSigmaBack));  // inputTensorData[homeOffset] !!
                } else {

                    filter_kernel_back = dO_dx_ki[batchOffset + homeOffset + c * cColorStrideBack];
                }

                valueSum +=
                        gradientInputTensor[batchOffset + neighbourOffset + c * cColorStrideBack] *
                                filter_kernel_back;

            }

            weightSum += totalWeight;

        }
      }
    }
  }

#pragma unroll
  for (int c = 0; c < C; c++) {
    gradientOutputTensor[batchOffset + homeOffset + c * cColorStrideBack] = valueSum;

  }
}

template <int C, int D>
void BilateralFilterCudaBackwardFunction(torch::Tensor gradientInputTensor,
                                         torch::Tensor gradientOutputTensor,
                                         torch::Tensor inputTensor,
                                         torch::Tensor outputTensor,
                                         torch::Tensor outputWeightsTensor,
                                         torch::Tensor dO_dx_ki,
                                         float sigma_x,
                                         float sigma_y,
                                         float sigma_z,
                                         float colorSigma) {
  // Getting tensor description.
  TensorDescription desc = TensorDescription(inputTensor);

  // Pre-calculating gaussian kernel.
  int windowSize_x = std::max(((int)ceil(5.0f * sigma_x) | 1), 5); // ORing last bit to ensure odd window size
  int windowSize_y = std::max(((int)ceil(5.0f * sigma_y) | 1), 5); // ORing last bit to ensure odd window size
  int windowSize_z = std::max(((int)ceil(5.0f * sigma_z) | 1), 5); // ORing last bit to ensure odd window size
  int halfWindowSize_x = floor(0.5f * windowSize_x);
  int halfWindowSize_y = floor(0.5f * windowSize_y);
  int halfWindowSize_z = floor(0.5f * windowSize_z);
  int halfWindowSize_arr[] = {halfWindowSize_x, halfWindowSize_y, halfWindowSize_z};
  float spatialExpConstant_x = -1.0f / (2 * sigma_x * sigma_x);
  float spatialExpConstant_y = -1.0f / (2 * sigma_y * sigma_y);
  float spatialExpConstant_z = -1.0f / (2 * sigma_z * sigma_z);
  float colorExpConstant = -1.0f / (2 * colorSigma * colorSigma);

  int* kernelSizes = new int[desc.dimensions];
  kernelSizes[0] = windowSize_x;
  kernelSizes[1] = windowSize_y;
  kernelSizes[2] = windowSize_z;

  auto* gaussianKernel_x = new float[windowSize_x];
  auto* gaussianKernel_y = new float[windowSize_y];
  auto* gaussianKernel_z = new float[windowSize_z];
  auto* xDistanceSquared = new float[windowSize_x];
  auto* yDistanceSquared = new float[windowSize_y];
  auto* zDistanceSquared = new float[windowSize_z];

  for (int i = 0; i < windowSize_x; i++) {
      int distance = i - halfWindowSize_x;
      gaussianKernel_x[i] = exp(distance * distance * spatialExpConstant_x);
      xDistanceSquared[i] = distance * distance;
  }
  for (int i = 0; i < windowSize_y; i++) {
      int distance = i - halfWindowSize_y;
      gaussianKernel_y[i] = exp(distance * distance * spatialExpConstant_y);
      yDistanceSquared[i] = distance * distance;
  }
  for (int i = 0; i < windowSize_z; i++) {
      int distance = i - halfWindowSize_z;
      gaussianKernel_z[i] = exp(distance * distance * spatialExpConstant_z);
      zDistanceSquared[i] = distance * distance;
  }

  // Writing constant memory.
  cudaMemcpyToSymbol(cBatchStrideBack, &desc.batchStride, sizeof(int));
  cudaMemcpyToSymbol(cColorStrideBack, &desc.channelStride, sizeof(int));
  cudaMemcpyToSymbol(cSizesBack, desc.sizes, sizeof(int) * 3);
  cudaMemcpyToSymbol(cStridesBack, desc.strides, sizeof(int) * 3);
  cudaMemcpyToSymbol(cKernelSizesBack, kernelSizes, sizeof(int) * desc.dimensions);
  cudaMemcpyToSymbol(cHalfWindowSize_arrBack, halfWindowSize_arr, sizeof(int) * desc.dimensions);
  cudaMemcpyToSymbol(cGaussianKernel_xBack, gaussianKernel_x, sizeof(float) * windowSize_x);
  cudaMemcpyToSymbol(cGaussianKernel_yBack, gaussianKernel_y, sizeof(float) * windowSize_y);
  cudaMemcpyToSymbol(cGaussianKernel_zBack, gaussianKernel_z, sizeof(float) * windowSize_z);
  cudaMemcpyToSymbol(cXDistanceSquaredBack, xDistanceSquared, sizeof(float) * windowSize_x);
  cudaMemcpyToSymbol(cYDistanceSquaredBack, yDistanceSquared, sizeof(float) * windowSize_y);
  cudaMemcpyToSymbol(cZDistanceSquaredBack, zDistanceSquared, sizeof(float) * windowSize_z);
  cudaMemcpyToSymbol(cColorExponentConstantBack, &colorExpConstant, sizeof(float));
  cudaMemcpyToSymbol(cSigma_xBack, &sigma_x, sizeof(float));
  cudaMemcpyToSymbol(cSigma_yBack, &sigma_y, sizeof(float));
  cudaMemcpyToSymbol(cSigma_zBack, &sigma_z, sizeof(float));
  cudaMemcpyToSymbol(cColorSigmaBack, &colorSigma, sizeof(float));

  cuda_error_check("Cuda check before kernel call.");

#define BLOCK_SIZE 32

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      inputTensor.scalar_type(), "BilateralFilterCudaKernel3DBackward", ([&] {
            BilateralFilterCudaKernel3DBackward<scalar_t, C>
                <<<dim3(int(desc.channelStride / BLOCK_SIZE) + 1, desc.batchCount), dim3(BLOCK_SIZE, 1)>>>(
                    gradientInputTensor.data_ptr<scalar_t>(),
                    gradientOutputTensor.data_ptr<scalar_t>(),
                    inputTensor.data_ptr<scalar_t>(),
                    outputTensor.data_ptr<scalar_t>(),
                    outputWeightsTensor.data_ptr<scalar_t>(),
                    dO_dx_ki.data_ptr<scalar_t>());
      }));

  cuda_error_check("Cuda check after kernel call.");
//  delete[] kernel;
  delete[] kernelSizes;
  delete[] gaussianKernel_x;
  delete[] gaussianKernel_y;
  delete[] gaussianKernel_z;
  delete[] xDistanceSquared;
  delete[] yDistanceSquared;
  delete[] zDistanceSquared;
}

// Function to choose template implementation based on dynamic, channels and dimensions
torch::Tensor BilateralFilterCudaBackward(torch::Tensor gradientInputTensor,
                                          torch::Tensor inputTensor,
                                          torch::Tensor outputTensor,
                                          torch::Tensor outputWeightsTensor,
                                          torch::Tensor dO_dx_ki,
                                          float sigma_x,
                                          float sigma_y,
                                          float sigma_z,
                                          float colorSigma) {
  torch::Tensor gradientOutputTensor = torch::zeros_like(gradientInputTensor);
  cuda_error_check("beginning");

#define CASE(c, d) BilateralFilterCudaBackwardFunction<c, d>(gradientInputTensor, gradientOutputTensor, inputTensor, outputTensor, outputWeightsTensor, dO_dx_ki, sigma_x, sigma_y, sigma_z, colorSigma);
  SWITCH_AB(CASE, BF_CUDA_MAX_CHANNELS, BF_CUDA_MAX_SPATIAL_DIMENSION, gradientInputTensor.size(1), gradientInputTensor.dim() - 2);

  return gradientOutputTensor;
}
