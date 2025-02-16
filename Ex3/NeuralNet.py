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
import matplotlib.pyplot as plt


class conv_block(nn.Module):

    def __init__(self, in_c, out_c, size = 3, padding = 1, pool = False):
        super().__init__()
        if pool:
            self.convo = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(in_c, out_c, kernel_size = size, padding = padding),
                nn.BatchNorm2d(out_c),
                nn.ReLU()
                )

        else :
            self.convo = nn.Sequential(
                            nn.Conv2d(in_c, out_c, kernel_size = size ,padding = padding),
                            nn.BatchNorm2d(out_c),
                            nn.ReLU()
                            )

    def forward(self, x):
        return self.convo(x)


class CNN(nn.Module):
    def __init__(self, input_channels= 3, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            conv_block(input_channels, 32, size = 3),
            conv_block(32,32, size = 3),
            nn.Dropout2d(0.2),
            conv_block(32,64, size=3, pool= True),
            conv_block(64,64,size=3),
            nn.Dropout2d(0.2),
            conv_block(64,128,size=3, pool= True),
            conv_block(128,128,size=3),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2)
        )
        self.classification = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):

        return self.classification(self.conv(x))



class Shortcut(nn.Module):
    '''
    Shortcut block as in paper https://arxiv.org/pdf/1512.03385 using zero padding to connect layers with different 
    channels, height and width, is zero_padding = False the first and last layer of the block has the same size and so you
    can use identity mapping.
    '''
    def __init__(self, input_channels, output_channels, zero_padding = False):
        super().__init__()
        self.zero = zero_padding
        self.input = input_channels
        self.out = output_channels
        if zero_padding:
            self.identity = nn.Sequential(
                conv_block(input_channels, output_channels, size = 3, stride=True ),
                nn.Conv2d(output_channels, output_channels, kernel_size = 3 ,padding = 1),
                nn.BatchNorm2d(output_channels)
            )
        else :
            self.identity = nn.Sequential(
                conv_block(input_channels, output_channels, size = 3),
                nn.Conv2d(output_channels, output_channels, kernel_size = 3 ,padding = 1),
                nn.BatchNorm2d(output_channels)            
            )

    def forward(self, x):
        if self.zero:
            pooled_x = F.max_pool2d(x, 2)
            pad = torch.zeros((pooled_x.shape[0], self.out-self.input, pooled_x.shape[2], pooled_x.shape[3]))
            return F.relu(self.identity(x) +  torch.concat((pooled_x, pad), dim=1))

        else :
            return F.relu(self.identity(x) + x)



class block_builder(nn.Module):
    def __init__(self, n_layers, n_filters, res = False):
        super().__init__()
        layers = []
        if res:
            for _ in range(n_layers):
                layers.append(Shortcut(n_filters, n_filters))
                self.block = nn.Sequential(*layers)
        else :
            for _ in range(n_layers):
                layers.append(conv_block(n_filters, n_filters))
                self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)



class Res_block(nn.Module):
    '''
    Residual block with the linear projection shortcut as described by method (B), 
    used on Imagenet in paper https://arxiv.org/pdf/1512.03385
    '''
    def __init__(self, input_channels, output_channels, short = False):
        super().__init__()
        self.short = short
        self.input = input_channels
        self.out = output_channels
        self.linear_proj = nn.Conv2d(input_channels, output_channels, kernel_size = 1 ,stride=2)

        if short:
            self.identity = nn.Sequential(
                conv_block(input_channels, output_channels, size = 3, stride=True ),
                nn.Conv2d(output_channels, output_channels, kernel_size = 3 ,padding = 1),
                nn.BatchNorm2d(output_channels)
            )
        else :
            self.identity = nn.Sequential(
                conv_block(input_channels, output_channels, size = 3),
                nn.Conv2d(output_channels, output_channels, kernel_size = 3 ,padding = 1),
                nn.BatchNorm2d(output_channels)            
            )

    def forward(self, x):
        if self.short:
            y = self.linear_proj(x)
            return F.relu(self.identity(x) + y)

        else :
            return F.relu(self.identity(x) + x)




class ResNet(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 10, depth = 5):
        super().__init__()
        
        self.conv = nn.Sequential(
            conv_block(in_channels, 16 ,size = 3),
            block_builder(depth, 16, res=True),
            Res_block(16, 32, short= True),
            block_builder(depth, 32, res=True),
            Res_block(32, 64, short = True),
            block_builder(depth, 64, res=True)    
        )
        self.classification = nn.Sequential(
            nn.AvgPool2d(8),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

    def forward(self,x):
        conv = self.conv(x)
        return self.classification(conv)






class Trainer():
    def __init__(self, model, loss, optimizer, epochs, device = None):
        self.model = model
        self.loss  = loss
        self.optimizer = optimizer
        self.epochs    = epochs
        self.results = {'training_loss':[], 
                        'training_accuracy':[],                    
                        'test_loss':[],
                        'test_accuracy':[]}
        self.device = device


    def train_epoch(self, training_loader):
        
        self.model.train()
        running_loss = 0
        accuracy = 0
        for i, data in enumerate(training_loader):
            print(f'iteration {i+1} of {len(training_loader)}')
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            l = self.loss(outputs, labels)
            l.backward()
            self.optimizer.step()
            running_loss += l.item()

            #CALCULATE TRAIN ACCURACY
            _, predicted = torch.max(outputs, 1)
            #total   += labels.size(0)
            accuracy += (predicted == labels).sum().item()/len(labels) #accuracy on the specific batch


        #Divide for the batch size
        running_loss /= len(training_loader)
        accuracy    /= len(training_loader)

        return running_loss, accuracy 
            
    def evaluate_model(self, test_loader):
        
        self.model.eval()
        test_loss = 0
        accuracy = 0

        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            l = self.loss(outputs, labels)
            test_loss += l.item()
            
            _, predicted = torch.max(outputs, 1)
            accuracy += (predicted == labels).sum().item()/len(labels) #accuracy on the specific batch
        
        accuracy  /= len(test_loader)
        test_loss /= len(test_loader)

        return test_loss, accuracy


    def train_model(self, training_loader, test_loader):

        #scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = 15, gamma=0.1)
        for epoch in range(self.epochs):
            print(f'----------------------------EPOCH NUMBER {epoch}-------------------------')
            train_loss, train_acc = self.train_epoch(training_loader)
            test_loss , test_acc  = self.evaluate_model(test_loader)
            #scheduler.step()
            print(f':::::::train_loss = {train_loss}:::::::')
            print(f':::::::train_acc = {train_acc}:::::::')
            print(f':::::::test_loss = {test_loss}:::::::')
            print(f':::::::test_acc = {test_acc} :::::::')
            self.results['training_loss'].append(train_loss)
            self.results['training_accuracy'].append(train_acc)
            self.results['test_loss'].append(test_loss)
            self.results['test_accuracy'].append(test_acc)
        return self.results


    def plot_results(self, path_loss, path_accuracy):
        plt.plot(self.results['training_loss'], color = 'blue')
        plt.plot(self.results['test_loss'], color = 'red')
        plt.xlabel('epochs')
        plt.ylabel('training loss')
        plt.savefig(path_loss)
        plt.close()

        plt.plot(self.results['training_accuracy'], color= 'blue')
        plt.plot(self.results['test_accuracy'], color = 'red')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.savefig(path_accuracy)
        plt.close()
