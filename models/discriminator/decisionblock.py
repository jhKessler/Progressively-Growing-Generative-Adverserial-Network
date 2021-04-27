import torch
import torch.nn as nn
import torch.nn.functional as F

class DecisionBlock(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.leaky = nn.LeakyReLU(0.2)
        self.conv = nn.Conv2d(in_channels=in_channels+1, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = nn.InstanceNorm2d(in_channels)
        self.pool = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=4, stride=1, bias=False)
        self.flatten = nn.Flatten()
        self.outp = nn.Linear(in_features=in_channels, out_features=1, bias=False)

    def forward(self, x):
        # minibatch stdev
        out_std = torch.sqrt(x.var(0, unbiased=False) + 1e-8)
        mean_std = out_std.mean()
        mean_std = mean_std.expand(x.size(0), 1, 4, 4)
        x = torch.cat([x, mean_std], 1)
        
        x = self.conv(x)
        x = self.norm(x)
        x = self.leaky(x)

        x = self.pool(x)
        x = self.leaky(x)
        x = self.flatten(x)
        x = self.outp(x)
        return x.view(-1)