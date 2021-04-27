import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.leaky = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.leaky(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.leaky(x)

        x = self.pool(x)
        return x