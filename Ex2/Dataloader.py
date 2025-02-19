import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.utils.data import Subset, Dataset, DataLoader
from Trainer import Trainer
import os 


# Standard MNIST transform.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST train and test.
ds_train   = MNIST(root='./dataset', train = True,  download = False)
ds_test    = MNIST(root='./dataset', train = False, download = False)
batch_size = 64

from PIL import Image

class MNISTSegmentationDataset(Dataset):
    def __init__(self,  dataset_train, transform=transform):
        self.mnist_data = dataset_train
        self.transform = transform
    
    def __len__(self):
        return len(self.mnist_data)
    
    def __getitem__(self, idx):
        img, label = self.mnist_data[idx]
        img = np.array(img)
        segmentation_mask = np.zeros_like(img)
        segmentation_mask[img > 0] = label+1
        
        
        #img = torch.from_numpy(img).float()
        #img /= 255
        img = self.transform(img)
        segmentation_mask = torch.tensor(segmentation_mask, dtype= torch.long)
        
        return img, segmentation_mask