"""
Example script: Optimizing the parameters of a bilateral filter layer for image denoising.

Author: Fabian Wagner
Contact: fabian.wagner@fau.de
"""
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from bilateral_filter_layer import BilateralFilter3d
import time
from skimage.data import camera


#############################################################
####             PARAMETERS (to be modified)             ####
#############################################################
# Set device.
use_gpu = True
# Filter parameter initialization.
sigma_x = 1.0
sigma_y = 1.0
sigma_z = 1.0
sigma_r = 0.01
# Image parameters.
downsample_factor = 2
n_slices = 1
# Training parameters.
n_epochs = 1000
#############################################################

if use_gpu:
    dev = "cuda"
else:
    dev = "cpu"

# Initialize filter layer.
layer_BF = BilateralFilter3d(sigma_x, sigma_y, sigma_z, sigma_r, use_gpu=use_gpu)

# Load cameraman image.
image = camera()[::downsample_factor, ::downsample_factor]
target = torch.tensor(image).unsqueeze(2).repeat(1, 1, n_slices).unsqueeze(0).unsqueeze(0)
target = target / torch.max(target)

# Prepare noisy input.
noise = 0.1 * torch.randn(target.shape)
tensor_in = (target + noise).to(dev)
target = target.to(dev)
tensor_in.requires_grad = True
print("Input shape: {}".format(tensor_in.shape))

# Define optimizer and loss.
optimizer = optim.Adam(layer_BF.parameters(), lr=0.1)
loss_function = nn.MSELoss()

# Training loop.
for i in range(n_epochs):
    optimizer.zero_grad()

    prediction = layer_BF(tensor_in)
    loss = loss_function(prediction, target)
    loss.backward()

    optimizer.step()

print("Sigma x: {}".format(layer_BF.sigma_x))
print("Sigma y: {}".format(layer_BF.sigma_y))
print("Sigma z: {}".format(layer_BF.sigma_z))
print("Sigma range: {}".format(layer_BF.color_sigma))

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
axes[2].imshow(target[0, 0, :, :, idx_center].detach().cpu(), vmin=vmin_img, vmax=vmax_img, cmap='gray')
axes[2].set_title('Ground truth', fontsize=14)
axes[2].axis('off')
# plt.savefig('out/example_optimization.png')
plt.show()
