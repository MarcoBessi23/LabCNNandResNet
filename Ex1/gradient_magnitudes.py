import os
import numpy as np
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
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mean, std = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]
data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness = 0.1,contrast = 0.1,saturation = 0.1),
    transforms.RandomAdjustSharpness(sharpness_factor = 2,p = 0.2),
    transforms.ToTensor(),
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

def compute_gradient_magnitudes(model, epochs, loss_fn, optimizer, dataloader, metric='norm'):
    gradients = []
    for _ in range(epochs):
        for i, (images, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            
            grad_values = [torch.norm(param.grad).item() for param in model.parameters() if param.grad is not None]
            gradients.append(np.std(grad_values) if metric == 'std' else sum(grad_values))
            
            optimizer.step()
            if i == 200:
                break
    return gradients

def plot_gradients(path, model1, model2, metric='norm'):
    opt1 = torch.optim.SGD(model1.parameters(), lr=0.001, weight_decay=1e-5, momentum=0.9)
    opt2 = torch.optim.SGD(model2.parameters(), lr=0.001, weight_decay=1e-5, momentum=0.9)
    
    gradient1 = compute_gradient_magnitudes(model1, epochs=1, loss_fn=loss, optimizer=opt1, dataloader=train_dataloader, metric=metric)
    gradient2 = compute_gradient_magnitudes(model2, epochs=1, loss_fn=loss, optimizer=opt2, dataloader=train_dataloader, metric=metric)
    
    plt.plot(gradient1, label='Plain', color='#1f77b4')
    plt.plot(gradient2, label='Skip connections', color='#ff7f0e')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Gradient ' + ('Std' if metric == 'std' else 'Norm'))
    plt.title('Plain vs Skip')
    plt.savefig(path)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Choose gradient metric: norm or std')
    parser.add_argument('--metric', type=str, choices=['norm', 'std'], default='norm', help='Metric to compute gradients (norm or std)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Model Initialization
    resnet = ResNet()
    cnn = CNN_Paper()

    # Plot Gradient Metric
    output_dir = os.path.join(os.getcwd(), 'Ex1/Results_grad')
    os.makedirs(output_dir, exist_ok=True)
    plot_gradients(os.path.join(output_dir, f'grad_{args.metric}.png'), cnn, resnet, metric=args.metric)