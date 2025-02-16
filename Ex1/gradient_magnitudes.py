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
from NeuralNet import  Trainer, CNN_Paper, ResNet_Paper, tanh_CNN_Paper, tanh_ResNet_Paper, ResNet
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
    root="data",
    train=True,
    download=False,
    transform=data_transform,
    target_transform=None
    )

# Setup testing data

test_data = torchvision.datasets.CIFAR10(
    root="data",
    train=False,
    download=False,
    transform=data_transform
)

batch_size = 128

train_dataloader = DataLoader(dataset = train_data,
                              batch_size = batch_size,
                              shuffle = True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size = batch_size,
                             shuffle = False)

loss = torch.nn.CrossEntropyLoss()



def grad_magnitude(model, epochs, loss, optimizer, train):
    for _ in range(epochs):
        run_loss = 0
        gradients = []
        for i, data in enumerate(train):
            print(f'iteration number {i}')
            images, labels = data
            optimizer.zero_grad()
            logits = model(images)
            l = loss(logits, labels)
            l.backward()

            total_grad_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    total_grad_norm += torch.norm(param.grad).item()

            gradients.append(total_grad_norm)
            
            optimizer.step()
            run_loss += l.item()

            #stop after iteration 200
            if i == 200:
                break
    return gradients

def plot_magnitudes(path, model1, model2):
    opt1 = torch.optim.SGD(model1.parameters(), lr=0.001, weight_decay= 1e-5, momentum=0.9)
    gradient1 = grad_magnitude(model1, epochs = 1, loss = loss, optimizer = opt1,
                                               train = train_dataloader)
    opt2 = torch.optim.SGD(model2.parameters(), lr=0.001, weight_decay= 1e-5, momentum=0.9)
    gradient2 = grad_magnitude(model2, epochs = 1, loss = loss, optimizer = opt2,
                                               train = train_dataloader)
    plt.plot(gradient1, color = '#1f77b4', label = 'Plain')
    plt.plot(gradient2, color = '#ff7f0e', label= 'Skip connections')
    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('gradient norm')
    plt.title('Plain vs Skip')

    plt.savefig(path)
    plt.close()

resnet = ResNet()  #35 layers
cnn = CNN_Paper()  #35 layers

#path_magnitudes = os.path.join(os.getcwd(), 'Ex1/Results_grad', 'grad_comparison.png')
#plot_magnitudes(path_magnitudes, cnn, resnet)

def grad_std(model, epochs, loss, optimizer, train):
    for _ in range(epochs):
        run_loss = 0
        gradients = []
        for i, data in enumerate(train):
            print(f'iteration number {i}')
            images, labels = data
            optimizer.zero_grad()
            logits = model(images)
            l = loss(logits, labels)
            l.backward()

            grad_norms = []
            for param in model.parameters():
                if param.grad is not None:
                    grad_norms.append(torch.norm(param.grad).item())
            
            grad_std = torch.std(torch.tensor(grad_norms)).item() 
            gradients.append(grad_std)
            
            optimizer.step()
            run_loss += l.item()

            #stop after iteration 200
            if i == 200:
                break
    return gradients


def plot_std(path, model1, model2):
    opt1 = torch.optim.SGD(model1.parameters(), lr=0.001, weight_decay= 1e-5, momentum=0.9)
    gradient1 = grad_std(model1, epochs = 1, loss = loss, optimizer = opt1,
                                               train = train_dataloader)
    opt2 = torch.optim.SGD(model2.parameters(), lr=0.001, weight_decay= 1e-5, momentum=0.9)
    gradient2 = grad_std(model2, epochs = 1, loss = loss, optimizer = opt2,
                                               train = train_dataloader)
    plt.plot(gradient1, color = '#1f77b4', label = 'Plain')
    plt.plot(gradient2, color = '#ff7f0e', label= 'Skip connections')
    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('gradient std')
    plt.title('Plain vs Skip')

    plt.savefig(path)
    plt.close()


path_std = os.path.join(os.getcwd(), 'Ex1/Results_grad', 'grad_std.png')
plot_std(path_std, cnn, resnet)
