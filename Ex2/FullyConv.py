### Exercise 2.2: Fully-convolutionalize a network.
#Take one of your trained classifiers and fully-convolutionalize it. 
# That is, turn it into a network that can predict classification outputs at *all* pixels in an input image.
# Can you turn this into a **detector** of handwritten digits? Give it a try.
#Hint 1: Sometimes the process of fully-convolutionalization is called "network surgery".
#Hint 2: To test your fully-convolutionalized networks you might want to write some functions to take random MNIST 
######## samples and embed them into a larger image (i.e. in a regular grid or at random positions).

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.utils.data import Subset, Dataset, DataLoader
from NeuralNet import FCN, CNN, FullyConvPaper, plotting, grid_plotting, test_grid
from Trainer import train_fcn_model
import os
from Dataloader import MNISTSegmentationDataset, transform, ds_train, ds_test


mnist_test = MNIST(root='./dataset', train = False, download = False)

train_dataset = MNISTSegmentationDataset(ds_train, transform=transform)
train_loader  = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataset  = MNISTSegmentationDataset(ds_test, transform)
test_loader   =  DataLoader(test_dataset, batch_size=64, shuffle=True)

#name_model = 'FCN PAPER'
name_model = 'FCN'

##Load the pretrained parameters from CNN model

pretrained_cnn = CNN()
pretrained_cnn.load_state_dict(torch.load('Ex2/parameters/cnn_param.pth'))

if name_model == 'FCN PAPER':
    fcn = FullyConvPaper()
else:
    fcn = FCN()

#load pretrained encoder from CNN
fcn.Encoder.load_state_dict(pretrained_cnn.conv.state_dict())

#Define metric, num epochs, and optimizer
criterion = torch.nn.CrossEntropyLoss()
epochs    = 15
optimizer = torch.optim.Adam(fcn.parameters(), lr= 1e-3, weight_decay= 1e-5)

##FIT THE MODEL

#if name_model == 'FCN PAPER':
#    train_loss = train_fcn_model(fcn, train_loader, epochs, optimizer, criterion)
#    torch.save(fcn.state_dict(), 'Ex2/parameters/paper_fcn_parameters.pth')
#else:
#    train_loss = train_fcn_model(fcn, train_loader, epochs, optimizer, criterion)
#    torch.save(fcn.state_dict(), 'Ex2/parameters/fcn_parameters.pth')

if name_model == 'FCN PAPER':
    fcn.load_state_dict(torch.load('Ex2/parameters/paper_fcn_parameters.pth'))
else:
    fcn.load_state_dict(torch.load('Ex2/parameters/fcn_parameters.pth'))

test_grid(2, 3, fcn, mnist_test)

#im, mask = next(iter(test_loader))
#print(fcn(im).shape)

#plotting(im, mask, fcn, 2)

#grid_plotting(im, mask, fcn, 4, name_model)