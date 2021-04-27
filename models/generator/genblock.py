import torch
import torch.nn as nn
import torch.nn.functional as F

class GenBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, upsample=True, out=False, inp=False, gen_bias=False):
        super().__init__()
        kern1 = 4 if inp else 3
        pad1 = 0 if inp else 1
        kern2 = 3
        pad2 = 1
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest") if upsample else None
        self.conv1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kern1, stride=1, padding=pad1, bias=gen_bias)
        self.bn1 = nn.InstanceNorm2d(num_features=out_channels)
        self.leaky1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kern2, stride=1, padding=pad2, bias=gen_bias)
        self.bn2 = nn.InstanceNorm2d(num_features=out_channels)
        self.leaky2 = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky2(x)
        return x