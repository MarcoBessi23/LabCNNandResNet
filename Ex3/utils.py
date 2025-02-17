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
import cv2



def denormalize_image(img_tensor, mean, std):
    """Invert normalization in torch.tensor and convert in numpy array"""
    mean, std = torch.tensor(mean), torch.tensor(std)
    img = img_tensor.clone().detach()
    img = img * std.view(-1, 1, 1) + mean.view(-1, 1, 1)
    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img

def save_image(img, path):
    plt.imshow(img)
    #plt.axis('off')
    plt.savefig(path) #, bbox_inches='tight', pad_inches=0)
    plt.close()


def compute_class_activation_map(model, img_tensor, class_idx):
    
    model.eval()
    pred = model(img_tensor)
    pred[:, class_idx].backward()
    
    gradients = model.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    activations = model.get_activations(img_tensor).detach()
    
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap)

    return heatmap.numpy()

def overlay_heatmap(img_path, heatmap, output_path, alpha=0.4):
    """Sovrappone la heatmap all'immagine originale e salva il risultato."""
    img = cv2.imread(img_path)
    heatmap = np.uint8(255 * heatmap)
    print('LA SHAPE DI IMAGE È:')
    print(f'{img.shape[0]}, {img.shape[1]}')
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 1, heatmap, alpha, 0)
    cv2.imwrite(output_path, superimposed_img)
