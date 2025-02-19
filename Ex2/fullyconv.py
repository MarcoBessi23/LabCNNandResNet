### Exercise 2.2: Fully-convolutionalize a network.
#Take one of your trained classifiers and fully-convolutionalize it. 
# That is, turn it into a network that can predict classification outputs at *all* pixels in an input image.
# Can you turn this into a **detector** of handwritten digits? Give it a try.
#Hint 1: Sometimes the process of fully-convolutionalization is called "network surgery".
#Hint 2: To test your fully-convolutionalized networks you might want to write some functions to take random MNIST 
######## samples and embed them into a larger image (i.e. in a regular grid or at random positions).

import os
import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from NeuralNet import FCN, CNN, FullyConvPaper, plotting, grid_plotting, test_grid
from Trainer import train_fcn_model
from Dataloader import MNISTSegmentationDataset, transform, ds_train, ds_test

# Argument parser
parser = argparse.ArgumentParser(description='Train or test an FCN model.')
parser.add_argument('--model', choices=['standardFCN', 'fcn8'], required=True, help='Choose the model to use.')
parser.add_argument('--mode', choices=['train', 'test'], required=True, help='Choose whether to train or test the model.')
parser.add_argument('--test_type', choices=['grid', 'SegmentedMNIST'], required=False, help='Choose the test type (only required for testing).')
args = parser.parse_args()

# Load MNIST test dataset to build the grid
mnist_test = MNIST(root='./dataset', train=False, download=False)

train_dataset = MNISTSegmentationDataset(ds_train, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataset = MNISTSegmentationDataset(ds_test, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

MODEL_NAME = 'FullyConvPaper' if args.model == 'fcn8' else 'FCN'

pretrained_cnn = CNN()
pretrained_cnn.load_state_dict(torch.load('Ex2/parameters/cnn_param.pth'))

fcn = FullyConvPaper() if MODEL_NAME == 'FullyConvPaper' else FCN()

fcn.Encoder.load_state_dict(pretrained_cnn.conv.state_dict())
if args.mode == 'train':
    criterion = torch.nn.CrossEntropyLoss()
    epochs = 15
    optimizer = torch.optim.Adam(fcn.parameters(), lr=1e-3, weight_decay=1e-5)
    
    train_loss = train_fcn_model(fcn, train_loader, epochs, optimizer, criterion)
    save_path = 'Ex2/parameters/paper_fcn_parameters.pth' if MODEL_NAME == 'FullyConvPaper' else 'Ex2/parameters/fcn_parameters.pth'
    torch.save(fcn.state_dict(), save_path)
    print(f'Model trained and saved at {save_path}')

elif args.mode == 'test':
    model_path = 'Ex2/parameters/paper_fcn_parameters.pth' if MODEL_NAME == 'FullyConvPaper' else 'Ex2/parameters/fcn_parameters.pth'
    fcn.load_state_dict(torch.load(model_path))
    print(f'Model loaded from {model_path}')
    
    if args.test_type == 'grid':
        test_grid(2, 3, fcn, mnist_test)
    elif args.test_type == 'SegmentedMNIST':
        im, mask = next(iter(test_loader))
        grid_plotting(im, mask, fcn, 4, MODEL_NAME)
    else:
        print("Error: --test_type is required for testing.")
