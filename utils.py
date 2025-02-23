import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import warnings
from skimage.measure import label, regionprops
from scipy import ndimage
import numpy as np
from skimage.morphology import binary_closing, binary_opening
warnings.filterwarnings('ignore') 

# Dice loss for the network
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1e-4

        # Flatten tensors
        pred = F.sigmoid(pred) ## 
        input_flat = pred.view(-1)
        target_flat = target.view(-1).float()

        intersection = torch.sum(input_flat * target_flat)
        union = torch.sum(input_flat) + torch.sum(target_flat)

        dice_coefficient = (2. * intersection + smooth) / (union + smooth)

        return 1 - dice_coefficient


# weighted bce loss to address data imbalancement
class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=None):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, input, target):
        criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        loss = criterion(input, target)
        return loss
    
# post-processing
def postprocess_mask(mask, kernel=np.ones((5,5))):
    # Convert mask to binary image (0s and 1s)
    mask = mask[0]
    mask_binary = (mask > 0.5).cpu().numpy().astype(np.uint8)

    # Apply morphological closing to fill small holes in the mask
    mask_closed = binary_closing(mask_binary, footprint=kernel)

    # Apply morphological opening to remove small noise
    mask_opened = binary_opening(mask_closed, footprint=kernel).astype(np.uint8)

    labeled_mask, num_objects = ndimage.label(mask_opened)

    if num_objects > 1:
        # Find properties of labeled regions
        props = regionprops(labeled_mask)
        # Get the index of the largest region by area
        largest_region_idx = np.argmax([prop.area for prop in props])

        # Extract the largest region from the labeled mask
        largest_region_mask = (labeled_mask == largest_region_idx + 1)
        mask_opened = largest_region_mask.astype(np.uint8)

    return mask_opened
