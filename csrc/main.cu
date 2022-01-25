#include <torch/extension.h>
#include <ATen/ATen.h>
#include <iostream>
#include "bilateral.h"

int main() {
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
        device = torch::Device(torch::kCUDA);
    }

    // Example script to call the gpu accelerated forward and backward pass using libtorch.

    torch::Tensor inputTensor = torch::randn({1, 1, 10, 10, 10});
    torch::Tensor inputTensor_gpu = inputTensor.to(device);
    torch::Tensor gradientInputTensor = torch::randn({1, 1, 10, 10, 10});
    torch::Tensor gradientInputTensor_gpu = gradientInputTensor.to(device);

    float sigma_x0 = 0.2;
    float sigma_y0 = 0.2;
    float sigma_z0 = 0.2;
    float colorSigma0 = 0.5;

    torch::Tensor outputTensor, outputWeightsTensor, dO_dx_ki, dO_dsig_r, dO_dsig_x, dO_dsig_y, dO_dsig_z;

    // Call forward and backward pass.
    std::tie(outputTensor, outputWeightsTensor, dO_dx_ki, dO_dsig_r, dO_dsig_x, dO_dsig_y, dO_dsig_z) = BilateralFilterCudaForward(inputTensor_gpu, sigma_x0, sigma_y0, sigma_z0 , colorSigma0);
    torch::Tensor gradientOutputTensor = BilateralFilterCudaBackward(gradientInputTensor_gpu, inputTensor.to(device), outputTensor.to(device), outputWeightsTensor.to(device), dO_dx_ki.to(device), sigma_x0, sigma_y0, sigma_z0 , colorSigma0);

    std::cout << inputTensor << std::endl;
    std::cout << gradientOutputTensor << std::endl;


    torch::Tensor inputTensorCPU = torch::randn({1, 1, 5, 5, 5});
    torch::Tensor gradientInputTensorCPU = torch::randn({1, 1, 5, 5, 5});

    std::tie(outputTensor, outputWeightsTensor, dO_dx_ki, dO_dsig_r, dO_dsig_x, dO_dsig_y, dO_dsig_z) = BilateralFilterCpuForward(inputTensor, sigma_x0, sigma_y0, sigma_z0, colorSigma0);
    torch::Tensor gradientOutputTensorCPU = BilateralFilterCpuBackward(gradientInputTensor, inputTensor, outputTensor, outputWeightsTensor, dO_dx_ki, sigma_x0, sigma_y0, sigma_z0, colorSigma0);

    std::cout << inputTensorCPU << std::endl;
    std::cout << outputTensor << std::endl;

    return 0;
}
