"""
Example script: Simple bilateral filtering for image denoising.

Author: Fabian Wagner
Contact: fabian.wagner@fau.de
"""
import matplotlib.pyplot as plt
import torch
from bilateral_filter_layer import BilateralFilter3d
import time
from skimage.data import camera


#############################################################
####             PARAMETERS (to be modified)             ####
#############################################################
# Set device.
use_gpu = True
# Filter parameters.
sigma_x = 1.5
sigma_y = 1.5
sigma_z = 1.0
sigma_r = 0.9
# Image parameters.
downsample_factor = 2
n_slices = 1
#############################################################

if use_gpu:
    dev = "cuda"
else:
    dev = "cpu"

# Initialize filter layer.
layer_BF = BilateralFilter3d(sigma_x, sigma_y, sigma_z, sigma_r, use_gpu=use_gpu)

# Load cameraman image.
image = camera()[::downsample_factor, ::downsample_factor]
tensor_gt = torch.tensor(image).unsqueeze(2).repeat(1, 1, n_slices).unsqueeze(0).unsqueeze(0)
tensor_gt = tensor_gt / torch.max(tensor_gt)

# Prepare noisy input.
noise = 0.1 * torch.randn(tensor_gt.shape)
tensor_in = (tensor_gt + noise).to(dev)
tensor_in.requires_grad = True
print("Input shape: {}".format(tensor_in.shape))

# Forward pass.
start = time.time()
prediction = layer_BF(tensor_in)
end = time.time()
print("Runtime forward pass: {} s".format(end - start))

# Backward pass.
loss = prediction.sum()
start = time.time()
loss.backward()
end = time.time()
print("Runtime backward pass: {} s".format(end - start))

# Visual results.
vmin_img = 0
vmax_img = 1
idx_center = int(tensor_in.shape[4] / 2)
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(7, 3))
axes[0].imshow(tensor_in[0, 0, :, :, idx_center].detach().cpu(), vmin=vmin_img, vmax=vmax_img, cmap='gray')
axes[0].set_title('Noisy input', fontsize=14)
axes[0].axis('off')
axes[1].imshow(prediction[0, 0, :, :, idx_center].detach().cpu(), vmin=vmin_img, vmax=vmax_img, cmap='gray')
axes[1].set_title('Filtered output', fontsize=14)
axes[1].axis('off')
axes[2].imshow(tensor_gt[0, 0, :, :, idx_center].detach().cpu(), vmin=vmin_img, vmax=vmax_img, cmap='gray')
axes[2].set_title('Ground truth', fontsize=14)
axes[2].axis('off')
plt.show()
