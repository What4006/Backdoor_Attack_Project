import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import random

class IAD_generator(nn.Module):
    def __init__(self):
        super(IAD_generator, self).__init__()

        #encoder 
        # 3 16 32 64 32 16 3
        # 64 64 32 16 32 64 64
        self.encoder=nn.Sequential(
            nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(64,32,3,2,1,1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32,16,3,2,1,1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,3,3,1,1),
            nn.Tanh()
        )

    def forward(self,x):
        x=self.encoder(x)
        x=self.decoder(x)

        return x

        
        