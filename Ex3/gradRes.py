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
from ResidualNetwork import ResNet
import matplotlib.pyplot as plt
from utils import denormalize_image, save_image, compute_class_activation_map, overlay_heatmap



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mean, std = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]

data_transform = transforms.Compose([
    transforms.ToTensor() ,
    transforms.Normalize(mean, std)
])


train_data = torchvision.datasets.CIFAR10(
    root="data",
    train=True,  
    download=False,
    transform=data_transform, 
    target_transform=None 
    )


test_data = torchvision.datasets.CIFAR10(
    root="data",
    train=False,
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


class Hooked_Net(nn.Module):
    ''' 
    Class for Hooked version of the ResNet model
    '''
    def __init__(self):
        super(Hooked_Net, self).__init__()

        self.resnet = ResNet()
        self.resnet.load_state_dict(torch.load('Ex3/parameters/ResNet_weights.pth', map_location=device ))

        self.features_conv = self.resnet.conv[:-1] #from the first layer to the last convolutional layer

        self.classifier = self.resnet.classification
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features_conv(x)
        h = x.register_hook(self.activations_hook)
        x = self.classifier(x)
        return x
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.features_conv(x)
    

res = Hooked_Net()
im, lab = next(iter(test_dataloader))
img = im[0]
image = denormalize_image(img, mean, std)

save_image(image, 'Ex3/images/cat.png')

heatmap = compute_class_activation_map(res, img.unsqueeze(0), class_idx=3)
plt.matshow(heatmap)
plt.savefig('./Ex3/images/heatmap_res.png')
plt.close()

overlay_heatmap('Ex3/images/cat.png', heatmap, './Ex3/images/cat_heatmap_res.png')