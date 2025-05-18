from src.utils import My_Resnet, My_Dataset
import torch
import numpy as np
import torch.nn as nn

# Get one image

dataset = My_Dataset(r'C:\github\resnet-pytorch\dataset\train', type='train')
print(len(dataset.labels))

import sys
sys.exit()
train_dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=16,
    num_workers=0, 
    shuffle=True,
    pin_memory=True,
)

image = next(iter(train_dataloader))
print(image[0].shape)  # Check the shape of the image tensor

import matplotlib.pyplot as plt

images = image[0]  # Assuming image is a tuple (data, label), and data is (B, C, H, W)
plt.figure(figsize=(12, 8))
for i in range(16):
    img = images[i]
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
        if img.shape[0] == 1:  # grayscale
            img = img[0]
        else:  # RGB
            img = np.transpose(img, (1, 2, 0))
    plt.subplot(4, 4, i + 1)
    plt.imshow(img, cmap='gray' if img.ndim == 2 else None)
    plt.axis('off')
plt.tight_layout()
plt.show()
