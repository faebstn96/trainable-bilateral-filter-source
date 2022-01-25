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

#pragma once

#include <torch/extension.h>
#include "../utils/common_utils.h"
#include "../utils/tensor_description.h"
#include <vector>
#include <iostream>
#include <algorithm>

#define BF_CUDA_MAX_CHANNELS 16
#define BF_CUDA_MAX_SPATIAL_DIMENSION 3

//#ifdef WITH_CUDA
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> BilateralFilterCudaForward(torch::Tensor inputTensor,
                                                                                                                                               float sigma_x,
                                                                                                                                               float sigma_y,
                                                                                                                                               float sigma_z,
                                                                                                                                               float colorSigma);
torch::Tensor BilateralFilterCudaBackward(torch::Tensor gradientInputTensor,
                                          torch::Tensor inputTensor,
                                          torch::Tensor outputTensor,
                                          torch::Tensor outputWeightsTensor,
                                          torch::Tensor dO_dx_ki,
                                          float sigma_x,
                                          float sigma_y,
                                          float sigma_z,
                                          float colorSigma);
//#endif

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> BilateralFilterCpuForward(torch::Tensor inputTensor,
                                                                                                                                                   float sigma_x,
                                                                                                                                                   float sigma_y,
                                                                                                                                                   float sigma_z,
                                                                                                                                                   float colorSigma);

torch::Tensor BilateralFilterCpuBackward(torch::Tensor gradientInputTensor,
                                              torch::Tensor inputTensor,
                                              torch::Tensor outputTensor,
                                              torch::Tensor outputWeightsTensor,
                                              torch::Tensor dO_dx_ki,
                                              float sigma_x,
                                              float sigma_y,
                                              float sigma_z,
                                              float colorSigma);
