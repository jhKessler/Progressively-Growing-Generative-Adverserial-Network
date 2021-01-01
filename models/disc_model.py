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

class InpBlock(nn.Module):
    
    def __init__(self, out_channels):
        super().__init__()
        self.inp = nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.leaky = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        x = self.inp(x)
        x = self.leaky(x)
        return x
    
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

class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = DiscBlock(in_channels=16, out_channels=32) # 32x32
        self.conv2 = DiscBlock(in_channels=32, out_channels=64) # 16x16
        self.conv3 = DiscBlock(in_channels=64, out_channels=128) # 8x8
        self.conv4 = DiscBlock(in_channels=128, out_channels=256) # 4x4
        self.outp = DecisionBlock(in_channels=256) # 1x1
        
        self.progression = [
            self.conv1, # -4
            self.conv2, # -3
            self.conv3, # -2
            self.conv4, # -1
        ]
        
        self.fromrgb = nn.ModuleList([
            InpBlock(out_channels=256),
            InpBlock(out_channels=128),
            InpBlock(out_channels=64),
            InpBlock(out_channels=32),
            InpBlock(out_channels=16)])

        self.apply(self.weights_init)
        
    def forward(self, inp, step, alpha):
        x = self.fromrgb[step](inp)

        if step > 0:
            for i in range(step, 0, -1):
                x = self.progression[-i](x)
                
                if i == step:
                    if 0 <= alpha < 1:
                        # slowly fade in the new layer
                        fade_val = F.avg_pool2d(inp, kernel_size=2)
                        fade_val = self.fromrgb[step - 1](fade_val)
                        x = (1 - alpha) * fade_val + alpha * x
        x = self.outp(x)
        return x
    
    def weights_init(self, layer):
            if type(layer) in [nn.Conv2d, nn.ConvTranspose2d]:
                nn.init.kaiming_normal_(layer.weight)
            if type(layer) == nn.BatchNorm2d:
                nn.init.normal_(layer.weight.data, 1.0, 0.02)
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)