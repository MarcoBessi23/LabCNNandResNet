import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.utils.data import Subset
from NeuralNet import CNN
from Trainer import Trainer
import os 
torch.optim.lr_scheduler.StepLR


# Standard MNIST transform.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST train and test.
ds_train   = MNIST(root='./dataset', train = True,  download = False, transform = transform)
ds_test    = MNIST(root='./dataset', train = False, download = False, transform = transform)
batch_size = 128
epochs = 4


dl_train = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=True)
dl_test  = torch.utils.data.DataLoader(ds_test, batch_size, shuffle=True)

cnn = CNN()
optimizer = torch.optim.Adam(cnn.parameters(), lr = 1e-3)
loss = torch.nn.CrossEntropyLoss()

trainer = Trainer(cnn, loss, optimizer, epochs)
trainer.train_model(dl_train, dl_test)
trainer.plot_results()
torch.save(cnn.state_dict(), 'Ex2/cnn_param.pth')
