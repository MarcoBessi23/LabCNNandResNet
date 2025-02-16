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
from NeuralNet import CNN, Trainer

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
    #transforms.RandomErasing(p=0.75,scale=(0.02, 0.1),value=1.0, inplace=False)
])


train_data = torchvision.datasets.CIFAR10(
    root="data", # where to download data to?
    train=True, # get training data
    download=False, # download data if it doesn't exist on disk
    transform=data_transform, 
    target_transform=None 
    )


test_data = torchvision.datasets.CIFAR10(
    root="data",
    train=False, # get test data
    download=False,
    transform=data_transform
)

batch_size = 1

train_dataloader = DataLoader(dataset = train_data,
                              batch_size = batch_size,
                              shuffle = True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size = batch_size,
                             shuffle = False)

loss = torch.nn.CrossEntropyLoss()

#model = CNN()
#model.load_state_dict(torch.load('Ex3/model.pt', map_location=device ))
class Hooked_CNN(nn.Module):
    def __init__(self):
        super(Hooked_CNN, self).__init__()

        self.cnn = CNN()
        self.cnn.load_state_dict(torch.load('Ex3/model.pt', map_location=device ))

        self.features_conv = self.cnn.conv[:-2] #from the first layer to the last convolutional layer

        self.downsample = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2)
            )

        self.classifier = self.cnn.classification
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features_conv(x)
        h = x.register_hook(self.activations_hook)
        x = self.downsample(x)
        x = self.classifier(x)
        return x
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.features_conv(x)
    

cnn = Hooked_CNN()
cnn.eval()

im, lab = next(iter(test_dataloader))
pred = cnn(im)
predicted = pred.argmax(dim= 1)
print(predicted)
print(lab)


import matplotlib.pyplot as plt
m , s = torch.tensor(mean), torch.tensor(std) 
img = im[0]
img = img * s.view(-1,1,1) + m.view(-1,1,1)
img = img.permute(1,2,0).numpy()
img = np.clip(img * 255, 0, 255).astype(np.uint8)
print(type(img[0][0][0]))
print(test_data.classes)

plt.imshow(img)
plt.savefig('Ex3/images/cat.png')

print(pred.shape)
#In this case we take the class activation map of class 3 beacause the class is a cat
pred[:, 3].backward()

# pull the gradients out of the model
gradients = cnn.get_activations_gradient()

# pool the gradients across the channels
pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

# get the activations of the last convolutional layer
activations = cnn.get_activations(im).detach()
print(activations.shape)

# weight the channels by corresponding gradients
for i in range(128):
    activations[:, i, :, :] *= pooled_gradients[i]

# average the channels of the activations
heatmap = torch.mean(activations, dim=1).squeeze()
heatmap = F.relu(heatmap)#np.maximum(heatmap, 0)

# normalize the heatmap
heatmap /= torch.max(heatmap)
plt.matshow(heatmap.squeeze())
plt.savefig('./Ex3/images/heatmap.png')
#plt.show()


import cv2

# Converte l'immagine da RGB (PyTorch) a BGR (OpenCV)
print(np.shape(img))
#img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#cv2.imshow("Image", img)

img = cv2.imread('./Ex3/images/cat.png')
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
cv2.imwrite('./Ex3/images/cat_heatmap.png', superimposed_img)