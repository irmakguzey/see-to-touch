import torch
import torch.nn as nn

# Taken from https://github.com/irmakguzey/tactile-dexterity
# from .utils import create_fc
from torchvision import models

# Script to return all pretrained models in torchvision.models module
def resnet18(pretrained, out_dim, remove_last_layer=True):
    encoder = models.__dict__['resnet18'](pretrained = pretrained)
    encoder.fc = nn.Identity()

    return encoder

def resnet34(pretrained, out_dim, remove_last_layer=True): # These out_dims are only given for implemntation purposes
    encoder = models.__dict__['resnet34'](pretrained = pretrained)
    encoder.fc = nn.Identity()

    return encoder


def alexnet(pretrained, out_dim, remove_last_layer=False):
    encoder = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=pretrained)
    if remove_last_layer:
        # Remove and recreate the last layer of alexnet - these values are taken from the alexnet implementation
        encoder.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(9216, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, out_dim, bias=True)
        )

    return encoder


def mobilenet_v2(pretrained, out_dim, remove_last_layer=True):
    if pretrained:
        encoder = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained='imagenet')
    else:
        encoder = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2')
    
    if remove_last_layer: # Looked through the encoder and replaced its classifier with empty layers
        encoder.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(1280, out_features=out_dim, bias=True)
        )

    return encoder

def densenet_121(pretrained, out_dim, remove_last_layer=True):
    encoder = models.__dict__['densenet121'](pretrained = True)
    
    if remove_last_layer:
        encoder.classifier = nn.Sequential(
            nn.Linear(1024, out_features=out_dim, bias=True)
        )

    return encoder


def googlenet(pretrained, out_dim, remove_last_layer=True):
    encoder = models.__dict__['googlenet'](pretrained = pretrained)
    
    if remove_last_layer:
        encoder.fc = nn.Sequential(
            nn.Linear(1024, out_features=out_dim, bias=True)
        )

    return encoder

def squeezenet(pretrained, out_dim, remove_last_layer=True):
    encoder = models.__dict__['squeezenet1_0'](pretrained = pretrained)

    if remove_last_layer:
        encoder.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Conv2d(512, out_dim, kernel_size=(1,1), stride=(1,1)),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1,1))
        )

    return encoder