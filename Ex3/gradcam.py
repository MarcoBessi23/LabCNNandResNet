import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset, Dataset, DataLoader
from NeuralNet import CNN, Trainer
from utils import denormalize_image, save_image, compute_class_activation_map, overlay_heatmap
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean, std = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]

data_transform = transforms.Compose([
    transforms.ToTensor() ,
    transforms.Normalize(mean, std)
])


train_data = torchvision.datasets.CIFAR10(
    root="data", # where to download data to?
    train=True, # get training data
    download=False, # download data if it doesn't exist on disk
    transform=data_transform, 
    target_transform=None 
    )


test_data = torchvision.datasets.CIFAR10(
    root="data",
    train=False, # get test data
    download=False,
    transform=data_transform
)

batch_size = 1

train_dataloader = DataLoader(dataset = train_data,
                              batch_size = batch_size,
                              shuffle = True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size = batch_size,
                             shuffle = False)

loss = torch.nn.CrossEntropyLoss()

class Hooked_CNN(nn.Module):
    def __init__(self):
        super(Hooked_CNN, self).__init__()

        self.cnn = CNN()
        self.cnn.load_state_dict(torch.load('Ex3/model.pt', map_location=device ))

        self.features_conv = self.cnn.conv[:-2] #from the first layer to the last convolutional layer

        self.downsample = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2)
            )

        self.classifier = self.cnn.classification
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features_conv(x)
        h = x.register_hook(self.activations_hook)
        x = self.downsample(x)
        x = self.classifier(x)
        return x
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.features_conv(x)
    

cnn = Hooked_CNN()
im, lab = next(iter(test_dataloader))
img = im[0]
image = denormalize_image(img, mean, std)
save_image(image, 'Ex3/images/cat.png')

heatmap = compute_class_activation_map(cnn, img.unsqueeze(0), class_idx=3)
plt.matshow(heatmap)
plt.savefig('./Ex3/images/heatmap.png')
plt.close()

overlay_heatmap('Ex3/images/cat.png', heatmap, './Ex3/images/cat_heatmap.png')
