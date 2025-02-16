import os
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
from NeuralNet import  Trainer, ResNet, CNN_Paper, ResNet_Paper
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean, std = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]

data_transform = transforms.Compose([
    # Flip the images randomly on the horizontal
    transforms.RandomHorizontalFlip(p=0.5),
    # Randomly rotate some images by 20 degrees
    transforms.RandomRotation(20),
    # Randomly adjust color jitter of the images
    transforms.ColorJitter(brightness = 0.1,contrast = 0.1,saturation = 0.1),
    # Randomly adjust sharpness
    transforms.RandomAdjustSharpness(sharpness_factor = 2,p = 0.2),
    # Turn the image into a torch.Tensor
    transforms.ToTensor() ,
    #randomly erase a pixel
    transforms.Normalize(mean, std),
    transforms.RandomErasing(p=0.75,scale=(0.02, 0.1),value=1.0, inplace=False)
])


train_data = torchvision.datasets.CIFAR10(
    root="data", # where to download data to?
    train=True, # get training data
    download=False, # download data if it doesn't exist on disk
    transform=data_transform, 
    target_transform=None 
    )

# Setup testing data

test_data = torchvision.datasets.CIFAR10(
    root="data",
    train=False, # get test data
    download=False,
    transform=data_transform
)

batch_size = 128

train_dataloader = DataLoader(dataset = train_data,
                              batch_size = batch_size,
                              shuffle = True)

test_dataloader  = DataLoader(dataset=test_data,
                             batch_size = batch_size,
                             shuffle = False)

path_loss = os.path.join(os.getcwd(), 'Ex1/Results_loss', 'loss_ResNet.png')
path_accuracy = os.path.join(os.getcwd(), 'Ex1/Results_loss', 'accuracy_ResNet.png')
epochs = 50
loss = torch.nn.CrossEntropyLoss()
model = ResNet_Paper()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#trainer = Trainer(model, loss, optimizer, epochs,path_loss, path_accuracy device)
#trainer.train_model(train_dataloader, test_dataloader)
#trainer.plot_results()
