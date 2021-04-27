import torch
import torch.nn as nn
import torch.nn.functional as F

class InpBlock(nn.Module):
    
    def __init__(self, out_channels):
        super().__init__()
        self.inp = nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.leaky = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        x = self.inp(x)
        x = self.leaky(x)
        return x