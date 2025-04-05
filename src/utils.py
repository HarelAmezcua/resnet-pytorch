import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

class My_Resnet(nn.Module):
    def __init__(self):
        super(My_Resnet, self).__init__()
        self.resnet = resnet18(weights = 'DEFAULT')
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)

    def forward(self, x):
        return self.resnet(x)              

class My_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.num_classes = len(self.classes)
        self.file_list = []
        self.labels = []
        
        for class_idx, class_name in enumerate(self.classes): # 
            class_path = os.path.join(root_dir, class_name)
            files = os.listdir(class_path)
            self.file_list.extend([os.path.join(class_name, f) for f in files])
            self.labels.extend([class_idx] * len(files))
        
        # Convert labels to one-hot encoding
        self.labels = np.eye(self.num_classes)[self.labels]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.file_list[idx])
        image = Image.open(img_path).convert('RGB')        
        
        image = self.transform(image)
        label = self.labels[idx]
        
        return image, label