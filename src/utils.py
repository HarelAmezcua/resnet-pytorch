import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

import albumentations as A
import cv2
import torch

class My_Resnet(nn.Module):
    def __init__(self, num_classes = 3):
        super(My_Resnet, self).__init__()
        self.resnet = resnet18(weights = 'DEFAULT')
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

class My_Dataset(Dataset):
    def __init__(self, root_dir, type = 'train'):        
        self.root_dir = root_dir
        self.transform = self.get_train_transform() if type == 'train' else self.get_test_transform()
        self.classes = sorted(os.listdir(root_dir))
        self.num_classes = len(self.classes)
        self.file_list = []
        self.labels = []
        
        for class_idx, class_name in enumerate(self.classes): # 
            class_path = os.path.join(root_dir, class_name)
            files = os.listdir(class_path)
            self.file_list.extend([os.path.join(class_name, f) for f in files])
            self.labels.extend([class_idx] * len(files))
        
        # Convert labels to one-hot encoding
        self.labels = np.array(self.labels, dtype=np.int64)  # ✅ Class indices



    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.file_list[idx])
        image = Image.open(img_path).convert('RGB')       
        image = np.array(image) 
        
        image = self.transform(image=image)['image']
        label = torch.tensor(self.labels[idx])

        
        return image, label
    
    @staticmethod
    def get_train_transform():
        IMG_SIZE = 256          # final square size fed to the network        

        # ---------- TRAIN ------------
        train_tf = A.Compose([
            A.LongestMaxSize(max_size=IMG_SIZE, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE,
                        border_mode=cv2.BORDER_CONSTANT),
            A.Affine(translate_percent={"x": 0.05, "y": 0.05},
                    scale=(0.95, 1.05),
                    rotate=(-15, 15),
                    p=0.5), # affine transform

            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),       # proven for CXRs :contentReference[oaicite:3]{index=3}
            A.RandomBrightnessContrast(brightness_limit=0.1,
                                    contrast_limit=0.1, p=0.3),
            #A.GaussNoise(var_limit=(10., 50.), p=0.2),
            A.CoarseDropout(max_height=32, max_width=32, min_height=32, min_width=32, mask_fill_value=0, p=0.5), # mimics ECG leads / markers
            A.Normalize(mean=[0.485, 0.456, 0.406],   # or your dataset stats
                    std=[0.229, 0.224, 0.225]),
            A.ToTensorV2()
        ])

        return train_tf        
    
    @staticmethod
    def get_test_transform():
        IMG_SIZE = 256          # final square size fed to the network        

        valid_tf = A.Compose([
            A.LongestMaxSize(max_size=IMG_SIZE, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE,
                        border_mode=cv2.BORDER_CONSTANT ),            
            A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
            A.ToTensorV2()
        ])

        return valid_tf