### Exercise 2.1: Explain why Residual Connections are so effective
## Use your two models (with and without residual connections) you developed above to study and **quantify** why 
## the residual versions of the networks learn more effectively.
##
##**Hint**: A good starting point might be looking at the gradient 
##          magnitudes passing through the networks during backpropagation.

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


def train_grad(model, epochs, loss, optimizer, train, residual):
    for _ in range(epochs):
        run_loss = 0
        gradients1 = []
        gradients2 = []
        for i, data in enumerate(train):
            print(f'iteration number {i}')
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
            run_loss += l.item()

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

    return np.array(gradient_layer1).mean()
    




#PLOT GRADIENTS MAGNITUDE

#
#path_cnn35 = os.path.join(os.getcwd(), 'Ex1/Results_grad', 'CNN35.png')
#cnn_35  = CNN_Paper()
#plot_gradients(path_cnn35, cnn_35, False)
#
#path_res35 = os.path.join(os.getcwd(), 'Ex1/Results_grad', 'Residual35.png')
#resnet_35  = ResNet_Paper()
#plot_gradients(path_res35, resnet_35, True)
#
#
#path_cnn53 = os.path.join(os.getcwd(), 'Ex1/Results_grad', 'CNN53.png')
#cnn_53  = CNN_Paper(depth=16)
#plot_gradients(path_cnn53, cnn_53, False)
#
#path_res53 = os.path.join(os.getcwd(), 'Ex1/Results_grad', 'Residual53.png')
#resnet_53  = ResNet_Paper(depth=8) 
#plot_gradients(path_res53, resnet_53, True)


######### PLOT GRADIENTS MAGNITUDE OF TANH AND CNN WITHOUT BATCH NORMALIZATION AND USING TANH ACTIVATION ###########

def plot_gradients_table():

    path_cnn35_tanh = os.path.join(os.getcwd(), 'Ex1/Results_grad', 'CNN_tanh35.png')
    cnn_35_tanh  = tanh_CNN_Paper()
    grad_cnn35 = plot_gradients(path_cnn35_tanh, cnn_35_tanh, False)

    path_res35_tanh = os.path.join(os.getcwd(), 'Ex1/Results_grad', 'Residual_tanh35.png')
    resnet_35_tanh  = tanh_ResNet_Paper()
    grad_res35 = plot_gradients(path_res35_tanh, resnet_35_tanh, True)

    path_cnn53_tanh = os.path.join(os.getcwd(), 'Ex1/Results_grad', 'CNN_tanh53.png')
    cnn_53_tanh  = tanh_CNN_Paper(depth = 16)
    grad_cnn53 = plot_gradients(path_cnn53_tanh, cnn_53_tanh, False)

    path_res53_tanh = os.path.join(os.getcwd(), 'Ex1/Results_grad', 'Residual_tanh53.png')
    resnet_53_tanh  = tanh_ResNet_Paper(depth = 8)
    grad_res53 = plot_gradients(path_res53_tanh, resnet_53_tanh, True)
    
    path_table = os.path.join(os.getcwd(), 'Ex1/Results_grad', 'table.png')
    models = ["35 Layers", "53 Layers"]
    grad_RES = [grad_res35, grad_res53]
    grad_CNN = [grad_cnn35, grad_cnn53]

    fig, ax = plt.subplots(figsize=(4, 2))
    ax.axis('tight')
    ax.axis('off')

    data = [["CNN", grad_CNN[0], grad_CNN[1]],
            ["ResNet", grad_RES[0], grad_RES[1]]]

    table = ax.table(cellText=data, 
                       colLabels=["", "35 Layers", "53 Layers"], 
                       cellLoc='center', loc='center')

    plt.savefig(path_table)
    plt.close()

plot_gradients_table()


#Plot the gradient magnitude for the trained models
#path_resnet = os.path.join(os.getcwd(), 'Ex1/Results_grad', 'trained_ResNet.png')
#resnet = ResNet()
#resnet.load_state_dict(torch.load('Ex1/parameters/ResNet_weights.pth', map_location=device ))
#plot_gradients(path_resnet, resnet, True)

#path_cnn = os.path.join(os.getcwd(), 'Ex1/Results_grad', 'trained_CNN.png')
#cnn = CNN_Paper()
#cnn.load_state_dict(torch.load('Ex1/parameters/CNN_weights.pth', map_location=device ))
#plot_gradients(path_cnn, cnn, False)
