"""
Trainable bilateral filter layer.

Author: Fabian Wagner
Contact: fabian.wagner@fau.de
"""
import torch
import torch.nn as nn
import bilateralfilter_cpu_lib
import bilateralfilter_gpu_lib


class BilateralFilterFunction3dCPU(torch.autograd.Function):
    """
    3D Differentiable bilateral filter to remove noise while preserving edges. C++ accelerated layer (CPU).
    See:
        Paris, S. (2007). A gentle introduction to bilateral filtering and its applications: https://dl.acm.org/doi/pdf/10.1145/1281500.1281604
    Args:
        input_img: input tensor: [B, C, X, Y, Z]
        sigma_x: standard deviation of the spatial blur in x direction.
        sigma_y: standard deviation of the spatial blur in y direction.
        sigma_z: standard deviation of the spatial blur in z direction.
        color_sigma: standard deviation of the range kernel.
    Returns:
        output (torch.Tensor): Filtered tensor.
    """

    @staticmethod
    def forward(ctx, input_img, sigma_x, sigma_y, sigma_z, color_sigma):
        assert len(input_img.shape) == 5, "Input shape of 3d bilateral filter layer must equal [B, C, X, Y, Z]."
        assert input_img.shape[1] == 1, "Currently channel dimensions >1 are not supported."

        # Use c++ implementation for better performance.
        outputTensor, outputWeightsTensor, dO_dx_ki, dO_dsig_r, dO_dsig_x, dO_dsig_y, dO_dsig_z = bilateralfilter_cpu_lib.forward_3d_cpu(input_img, sigma_x, sigma_y, sigma_z, color_sigma)

        ctx.save_for_backward(input_img,
                              sigma_x,
                              sigma_y,
                              sigma_z,
                              color_sigma,
                              outputTensor,
                              outputWeightsTensor,
                              dO_dx_ki,
                              dO_dsig_r,
                              dO_dsig_x,
                              dO_dsig_y,
                              dO_dsig_z)  # save for backward

        return outputTensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_sig_x = None
        grad_sig_y = None
        grad_sig_z = None
        grad_color_sigma = None

        input_img = ctx.saved_tensors[0]  # input image
        sigma_x = ctx.saved_tensors[1]
        sigma_y = ctx.saved_tensors[2]
        sigma_z = ctx.saved_tensors[3]
        color_sigma = ctx.saved_tensors[4]
        outputTensor = ctx.saved_tensors[5]  # filtered image
        outputWeightsTensor = ctx.saved_tensors[6]  # weights
        dO_dx_ki = ctx.saved_tensors[7]  # derivative of output with respect to input, while k==i
        dO_dsig_r = ctx.saved_tensors[8]  # derivative of output with respect to range sigma
        dO_dsig_x = ctx.saved_tensors[9]  # derivative of output with respect to sigma x
        dO_dsig_y = ctx.saved_tensors[10]  # derivative of output with respect to sigma y
        dO_dsig_z = ctx.saved_tensors[11]  # derivative of output with respect to sigma z

        # calculate gradient with respect to the sigmas
        grad_color_sigma = torch.sum(grad_output * dO_dsig_r)
        grad_sig_x = torch.sum(grad_output * dO_dsig_x)
        grad_sig_y = torch.sum(grad_output * dO_dsig_y)
        grad_sig_z = torch.sum(grad_output * dO_dsig_z)

        grad_output_tensor = bilateralfilter_cpu_lib.backward_3d_cpu(grad_output,
                                                                     input_img,
                                                                     outputTensor,
                                                                     outputWeightsTensor,
                                                                     dO_dx_ki,
                                                                     sigma_x,
                                                                     sigma_y,
                                                                     sigma_z,
                                                                     color_sigma)

        return grad_output_tensor, grad_sig_x, grad_sig_y, grad_sig_z, grad_color_sigma


class BilateralFilterFunction3dGPU(torch.autograd.Function):
    """
    3D Differentiable bilateral filter to remove noise while preserving edges. CUDA accelerated layer.
    See:
        Paris, S. (2007). A gentle introduction to bilateral filtering and its applications: https://dl.acm.org/doi/pdf/10.1145/1281500.1281604
    Args:
        input_img: input tensor: [B, C, X, Y, Z]
        sigma_x: standard deviation of the spatial blur in x direction.
        sigma_y: standard deviation of the spatial blur in y direction.
        sigma_z: standard deviation of the spatial blur in z direction.
        color_sigma: standard deviation of the range kernel.
    Returns:
        output (torch.Tensor): Filtered tensor.
    """

    @staticmethod
    def forward(ctx, input_img, sigma_x, sigma_y, sigma_z, color_sigma):
        assert len(input_img.shape) == 5, "Input shape of 3d bilateral filter layer must equal [B, C, X, Y, Z]."
        assert input_img.shape[1] == 1, "Currently channel dimensions >1 are not supported."

        # Use c++ implementation for better performance.
        outputTensor, outputWeightsTensor, dO_dx_ki, dO_dsig_r, dO_dsig_x, dO_dsig_y, dO_dsig_z = bilateralfilter_gpu_lib.forward_3d_gpu(input_img, sigma_x, sigma_y, sigma_z, color_sigma)

        ctx.save_for_backward(input_img,
                              sigma_x,
                              sigma_y,
                              sigma_z,
                              color_sigma,
                              outputTensor,
                              outputWeightsTensor,
                              dO_dx_ki,
                              dO_dsig_r,
                              dO_dsig_x,
                              dO_dsig_y,
                              dO_dsig_z)  # save for backward

        return outputTensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_sig_x = None
        grad_sig_y = None
        grad_sig_z = None
        grad_color_sigma = None

        input_img = ctx.saved_tensors[0]  # input image
        sigma_x = ctx.saved_tensors[1]
        sigma_y = ctx.saved_tensors[2]
        sigma_z = ctx.saved_tensors[3]
        color_sigma = ctx.saved_tensors[4]
        outputTensor = ctx.saved_tensors[5]  # filtered image
        outputWeightsTensor = ctx.saved_tensors[6]  # weights
        dO_dx_ki = ctx.saved_tensors[7]  # derivative of output with respect to input, while k==i
        dO_dsig_r = ctx.saved_tensors[8]  # derivative of output with respect to range sigma
        dO_dsig_x = ctx.saved_tensors[9]  # derivative of output with respect to sigma x
        dO_dsig_y = ctx.saved_tensors[10]  # derivative of output with respect to sigma y
        dO_dsig_z = ctx.saved_tensors[11]  # derivative of output with respect to sigma z

        # calculate gradient with respect to the sigmas
        grad_color_sigma = torch.sum(grad_output * dO_dsig_r)
        grad_sig_x = torch.sum(grad_output * dO_dsig_x)
        grad_sig_y = torch.sum(grad_output * dO_dsig_y)
        grad_sig_z = torch.sum(grad_output * dO_dsig_z)

        grad_output_tensor = bilateralfilter_gpu_lib.backward_3d_gpu(grad_output,
                                                                     input_img,
                                                                     outputTensor,
                                                                     outputWeightsTensor,
                                                                     dO_dx_ki,
                                                                     sigma_x,
                                                                     sigma_y,
                                                                     sigma_z,
                                                                     color_sigma)

        return grad_output_tensor, grad_sig_x, grad_sig_y, grad_sig_z, grad_color_sigma


class BilateralFilter3d(nn.Module):
    def __init__(self, sigma_x, sigma_y, sigma_z, color_sigma, use_gpu=True):
        super(BilateralFilter3d, self).__init__()

        self.use_gpu = use_gpu

        # make sigmas trainable parameters
        self.sigma_x = nn.Parameter(torch.tensor(sigma_x))
        self.sigma_y = nn.Parameter(torch.tensor(sigma_y))
        self.sigma_z = nn.Parameter(torch.tensor(sigma_z))
        self.color_sigma = nn.Parameter(torch.tensor(color_sigma))

    def forward(self, input_tensor):

        assert len(input_tensor.shape) == 5, "Input shape of 3d bilateral filter layer must equal [B, C, X, Y, Z]."
        assert input_tensor.shape[1] == 1, "Currently channel dimensions >1 are not supported."

        # Choose between CPU processing and CUDA acceleration.
        if self.use_gpu:
            return BilateralFilterFunction3dGPU.apply(input_tensor,
                                                      self.sigma_x,
                                                      self.sigma_y,
                                                      self.sigma_z,
                                                      self.color_sigma)
        else:
            return BilateralFilterFunction3dCPU.apply(input_tensor,
                                                      self.sigma_x,
                                                      self.sigma_y,
                                                      self.sigma_z,
                                                      self.color_sigma)
