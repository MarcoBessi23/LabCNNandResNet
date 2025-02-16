import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os


class CNN(nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size= 3, dilation=1, padding = 1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size= 3, dilation=1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size= 3),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.classifier(x)
        return x


class FCN(nn.Module):
    '''
    Fully convolutionalized version of the CNN network above without skip_connections.
    '''
    def __init__(self, NUM_CLASSES = 11):
        super(FCN, self).__init__()

        self.Encoder = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size= 3, dilation=1, padding = 1),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size= 3, dilation=1, padding = 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, kernel_size= 3),
                nn.BatchNorm2d(256),
                nn.ReLU()
                )

        self.Decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, NUM_CLASSES, 1),
                )

    def forward(self, x):
        out = self.Encoder(x)
        out = self.Decoder(out)
        return out
    


class FullyConvPaper(nn.Module):
    '''
    FCN architecture adapted with skip connections as shown in paper https://arxiv.org/abs/1411.4038
    number of channels at each decoder layer was selected with the same criterion as in paper. 
    '''
    def __init__(self, NUM_CLASSES = 11):
        super(FullyConvPaper, self).__init__()

        self.Encoder = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size= 3, dilation=1, padding = 1),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size= 3, dilation=1, padding = 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, kernel_size= 3),
                nn.BatchNorm2d(256),
                nn.ReLU()
                )
        self.conv1x1_layer3 = nn.ConvTranspose2d(64, NUM_CLASSES, 1)
        self.conv1x1_layer7 = nn.ConvTranspose2d(128, NUM_CLASSES, 1)
        self.Deconv1 = nn.ConvTranspose2d(256, 256, 3, 1) 
        self.Deconv2 = nn.ConvTranspose2d(256, NUM_CLASSES, 2, 2)
        self.Deconv3 = nn.ConvTranspose2d(NUM_CLASSES, NUM_CLASSES, 2, 2)
        self.Deconv4 = nn.ConvTranspose2d(NUM_CLASSES, NUM_CLASSES, 1, 1)
        self.Deconv5 = nn.ConvTranspose2d(NUM_CLASSES, NUM_CLASSES, 1)
        self.relu    = nn.ReLU(inplace=True)
        self.bn1     = nn.BatchNorm2d(256)
        self.bn2     = nn.BatchNorm2d(NUM_CLASSES)
        self.bn3     = nn.BatchNorm2d(NUM_CLASSES)
        self.bn4     = nn.BatchNorm2d(NUM_CLASSES)
        
    def forward(self, x):
        out_layer_3 = self.Encoder[:2](x)
        out_layer_7 = self.Encoder[:7](x)
        out = self.Encoder(x)
        out = self.bn1(self.relu(self.Deconv1(out)))
        out = self.Deconv2(out)  #128 channels
        out = self.bn2(self.relu(out) + self.conv1x1_layer7(out_layer_7))
        out = self.Deconv3(out) #64 channels
        out = self.bn3(self.relu(out)+ self.conv1x1_layer3(out_layer_3))
        out = self.bn4(self.relu(self.Deconv4(out)))
        out = self.Deconv5(out)
        return out



def plotting(image, segmented_label, model, index):

    model.eval()
    with torch.no_grad():
        multilogit = F.softmax(model(image), dim=1)
        prediction_mask = torch.max(multilogit, dim=1)[1]
    
    
    fig, axs = plt.subplots(index, 3, figsize=(15, 5))

    axs[0].imshow(image[index].squeeze(0), cmap='gray')
    axs[0].set_title('MNIST imagine')
    axs[0].axis('off')

    axs[1].imshow(segmented_label[index], cmap = 'gray')
    axs[1].set_title('Segmentation Mask')
    axs[1].axis('off')

    axs[2].imshow(prediction_mask[index], cmap='tab20', vmin=0, vmax=10)
    axs[2].set_title('Mask Prediction')
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()

def grid_plotting(image, segmented_label, model, indices, name):
    '''
    Function to plot image, its segmented mask and the predicted mask for a random number of images
    '''
    if name == 'FCN PAPER':
        path = os.path.join(os.getcwd(), 'Ex2/results_ex2', 'grid_res.png')
    else:
        path = os.path.join(os.getcwd(), 'Ex2/results_ex2', 'covolutionalized_grid_res.png')
    #model.eval()
    with torch.no_grad():
        multilogit = F.softmax(model(image), dim=1)
        prediction_mask = torch.max(multilogit, dim=1)[1]
    
    
    fig, axs = plt.subplots(indices, 3, figsize=(15, 15))
    for i in range(indices):

        axs[i,0].imshow(image[i].squeeze(0), cmap='gray')
        axs[i,0].set_title('MNIST imagine')
        axs[i,0].axis('off')

        axs[i,1].imshow(segmented_label[i], cmap = 'gray')
        axs[i,1].set_title('Segmentation Mask')
        axs[i,1].axis('off')

        axs[i,2].imshow(prediction_mask[i], cmap='tab20', vmin=0, vmax=10)
        axs[i,2].set_title('Mask Prediction')
        axs[i,2].axis('off')

    plt.tight_layout()
    plt.savefig(path)




def create_random_grid(images, grid_size=(5, 5), canvas_size=(28, 28)):

    path   = os.path.join(os.getcwd(), )
    canvas = np.zeros((canvas_size[0] * grid_size[0], canvas_size[1] * grid_size[1]))
    
    for image, _ in images:
        
        row = np.random.randint(0, grid_size[0])
        col = np.random.randint(0, grid_size[1])
        start_row = row * canvas_size[0]
        start_col = col * canvas_size[1]

        image_np = np.array(image)
        canvas[start_row:start_row + canvas_size[0], start_col:start_col + canvas_size[1]] = image_np

    return canvas

def test_grid(height, width, model, dataset):

    path_mask = os.path.join(os.getcwd(), 'Ex2/results_ex2', 'multi_digit_mask.png')
    path_pred = os.path.join(os.getcwd(), 'Ex2/results_ex2', 'multi_digit_pred.png')
    sample_images = [dataset[i] for i in np.random.choice(len(dataset), 10, replace=False)]
    grid = create_random_grid(sample_images, grid_size=(height, width), canvas_size=(28,28))
    grid = torch.tensor(grid)
    grid = grid.float()
    grid = grid.unsqueeze(0)
    grid = grid.unsqueeze(0)
    grid = transforms.Normalize((0.1307,), (0.3081,))(grid)
    plt.imshow(grid[0][0], cmap='gray')
    plt.savefig(path_mask)

    logit = F.softmax(model(grid), dim=1)
    predicted_mask = torch.max(logit, dim = 1)[1]
    plt.imshow(predicted_mask[0], cmap='tab20', vmin=0, vmax=10)
    plt.savefig(path_pred)
