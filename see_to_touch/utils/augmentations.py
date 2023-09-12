import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T 

from torchvision.transforms.functional import crop

from .constants import *

# Method for tactile augmentations in BYOL
def get_tactile_augmentations(img_means, img_stds, img_size):
    tactile_aug = T.Compose([
        T.RandomApply(
            nn.ModuleList([T.RandomResizedCrop(img_size, scale=(.9, 1))]),
            p = 0.5
        ), 
        T.RandomApply(
            nn.ModuleList([T.GaussianBlur((3, 3), (1.0, 2.0))]), 
            p = 0.5
        ),
        T.Normalize(
            mean = img_means,
            std = img_stds
        )
    ])
    return tactile_aug

def get_vision_augmentations(img_means, img_stds):
    color_aug = T.Compose([
        T.RandomApply(
            nn.ModuleList([T.ColorJitter(.8,.8,.8,.2)]), 
            p = 0.2
        ),
        T.RandomGrayscale(p=0.2), 
        T.RandomApply(
            nn.ModuleList([T.GaussianBlur((3, 3), (1.0, 2.0))]), 
            p = 0.2
        ),
        T.Normalize(
            mean = img_means,
            std =  img_stds
        )
    ])

    return color_aug 

# Official SimCLR augmentations: 
def get_simclr_augmentation(color_jitter_const, mean_tensor, std_tensor):
    return T.Compose([
        T.RandomApply(
            nn.ModuleList([T.ColorJitter(.8 * color_jitter_const, .8 * color_jitter_const, .8 * color_jitter_const, .2 * color_jitter_const)]), 
            p = 0.2
        ),
        T.RandomGrayscale(p = 0.2), 
        T.RandomApply(
            nn.ModuleList([T.GaussianBlur((3, 3), (1.5, 1.5))]), 
            p = 0.2
        ),
        T.Normalize(
            mean = mean_tensor, 
            std = std_tensor
        )
    ])


# Modified version of the official implementation of MoCoV3: https://github.com/facebookresearch/moco-v3/
def get_moco_augmentations(mean_tensor, std_tensor):
    first_augmentation_function = T.Compose([
        T.RandomApply(
            nn.ModuleList([T.ColorJitter(0.4, 0.4, 0.2, 0.1)])  # not strengthened
            , p=0.8
        ),
        T.RandomGrayscale(p = 0.2),
        T.RandomApply(
            torch.nn.ModuleList([T.GaussianBlur((3, 3), (1.0, 2.0))]), 
            p = 0.5
        ),
        T.Normalize(
            mean = mean_tensor,
            std = std_tensor
        )
    ])

    second_augmentation_function = T.Compose([
        T.RandomApply(
            nn.ModuleList([T.ColorJitter(0.4, 0.4, 0.2, 0.1)])  # not strengthened
            , p=0.8
        ),
        T.RandomGrayscale(p = 0.2),
        T.RandomApply(
            torch.nn.ModuleList([T.GaussianBlur((3, 3), (1.0, 2.0))]), 
            p = 0.5
        ),
        T.RandomSolarize(threshold = 0.5, p = 0.2),
        T.Normalize(
            mean = mean_tensor,
            std = std_tensor
        )
    ])

    return first_augmentation_function, second_augmentation_function

# Vision transforms used
def crop_transform(image, camera_view=0, image_size=480): # This is significant to the setup
    if camera_view == 0:
        return crop(image, 0,0, image_size,image_size)
        # return crop(image, 20,20,300,300)
    elif camera_view == 1:
        return crop(image, 0,90, image_size,image_size)
    
    elif camera_view == 2:
        return crop(image, 0, 90, image_size, image_size)
    
def get_inverse_image_norm():
    np_means = np.asarray(VISION_IMAGE_MEANS)
    np_stds = np.asarray(VISION_IMAGE_STDS)

    inv_normalization_transform = T.Compose([
        T.Normalize(mean = [0,0,0], std = 1 / np_stds ), 
        T.Normalize(mean = -np_means, std = [1,1,1])
    ])

    return inv_normalization_transform

# Tactile transforms used
def tactile_scale_transform(image):
    image = (image - TACTILE_PLAY_DATA_CLAMP_MIN) / (TACTILE_PLAY_DATA_CLAMP_MAX - TACTILE_PLAY_DATA_CLAMP_MIN)
    return image

def tactile_clamp_transform(image):
    image = torch.clamp(image, min=TACTILE_PLAY_DATA_CLAMP_MIN, max=TACTILE_PLAY_DATA_CLAMP_MAX)
    return image

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)