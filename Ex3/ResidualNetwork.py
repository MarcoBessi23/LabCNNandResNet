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

    def __init__(self, in_c, out_c, size = 3, padding = 1, stride = False):
        super().__init__()
        if stride:
            self.convo = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size = size, padding = padding, stride=2),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                )

        else :
            self.convo = nn.Sequential(
                            nn.Conv2d(in_c, out_c, kernel_size = size ,padding = padding),
                            nn.BatchNorm2d(out_c),
                            nn.ReLU()
                            )

    def forward(self, x):
        return self.convo(x)




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



class Res_block(nn.Module):
    '''
    Residual block with the linear projection shortcut as described by method (B), 
    used on Imagenet in paper https://arxiv.org/pdf/1512.03385
    '''
    def __init__(self, input_channels, output_channels, short = False):
        super().__init__()
        self.short = short
        #self.input = input_channels
        #self.out = output_channels
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
