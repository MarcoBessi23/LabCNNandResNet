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
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mean, std = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]
data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness = 0.1,contrast = 0.1,saturation = 0.1),
    transforms.RandomAdjustSharpness(sharpness_factor = 2,p = 0.2),
    transforms.ToTensor() ,
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


def train_grad(model, epochs, loss, optimizer, train, residual):
    for _ in range(epochs):
        
        gradients1, gradients2 = [], []
        for i, data in enumerate(train):
            print(i)
            images, labels = data
            optimizer.zero_grad()
            logits = model(images)
            l = loss(logits, labels)
            l.backward()
            
            if residual: 
                grad_norm_first = torch.norm(model.conv[1].block[0].identity[0].convo[0].weight.grad).item()
                grad_norm_final = torch.norm(model.conv[-1].block[0].identity[0].convo[0].weight.grad).item()
                print(grad_norm_first)
                print(grad_norm_final)
            else :
                #I take the gradients with respect to the same parameters at the same level
                grad_norm_first = torch.norm(model.conv[1].block[0].convo[0].weight.grad).item()
                grad_norm_final = torch.norm(model.conv[-1].block[-2].convo[0].weight.grad).item()
                print(grad_norm_first)
                print(grad_norm_final)
            
            gradients1.append(grad_norm_first)
            gradients2.append(grad_norm_final)
            
            optimizer.step()
            
            #stop after iteration 50, used only for tanh gradients
            if i == 50:
                break
    return gradients1, gradients2

def plot_gradients(path, model, res):
    opt = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay= 1e-5, momentum=0.9)
    gradient_layer1, gradient_layerfinal = train_grad(model, epochs = 1, loss = loss, optimizer = opt,
                                               train = train_dataloader, residual = res)
    plt.plot(gradient_layer1, color = 'blue', label = 'initial_layer')
    plt.plot(gradient_layerfinal, color = 'red', label= 'final layers')
    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('gradient norm')
    plt.title('initial vs final layers ')

    plt.savefig(path)
    plt.close()



def main():
    parser = argparse.ArgumentParser(description='Choose the model for initial and finals layers gradients magnitude ')
    parser.add_argument('--model', choices=['CNN35', 'Residual35', 'CNN53', 'Residual53','CNN35_tanh', 
                                            'Residual_tanh35', 'CNN53_tanh', 'Residual_tanh53', 'trained_CNN',
                                            'Residual_trained'], required=True, help='Model to analyze')
    
    args = parser.parse_args()
    output_dir = os.path.join(os.getcwd(), 'Ex1/Results_grad')
    os.makedirs(output_dir, exist_ok=True)
    models = {
        "CNN35": CNN_Paper(),
        "Residual35": ResNet_Paper(),
        "CNN53": CNN_Paper(depth=16),
        "Residual53": ResNet_Paper(depth=8),
        "CNN_tanh35": tanh_CNN_Paper(),
        "Residual_tanh35": tanh_ResNet_Paper(),
        "CNN_tanh53": tanh_CNN_Paper(depth=16),
        "Residual_tanh53": tanh_ResNet_Paper(depth=8),
        "trainedCNN": CNN_Paper(),
        "Residual_trained": ResNet()
    }
    
    model_name = args.model
    model = models[model_name]
    
    if model_name == "trained_CNN":
        model_path = f'Ex1/parameters/CNN_weights.pth'
        model.load_state_dict(torch.load(model_path, map_location=device))
    elif model_name ==  "resnet_trained":
        model_path = f'Ex1/parameters/ResNet_weights.pth'
        model.load_state_dict(torch.load(model_path, map_location=device))

    path = os.path.join(output_dir, f'{model_name}.png')
    ##IF RESNET IS IN MODEL NAME THEN RESIDUAL IS TRUE
    residual = "Residual" in model_name
    plot_gradients(path, model, residual)

if __name__ == "__main__":
    main()
