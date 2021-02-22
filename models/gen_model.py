import torch
import torch.nn as nn
import torch.nn.functional as F

gen_bias = False

class GenBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, upsample=True, out=False, inp=False):
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
        
    
class Generator(nn.Module):

    def __init__(self, noise_dim):
        assert noise_dim > 0
        super().__init__()
        self.noise_dim = noise_dim
        self.tanh = nn.Tanh()
        self.inp = GenBlock(in_channels=self.noise_dim, out_channels=256, upsample=False, inp=True) # 4x4
        self.conv1 = GenBlock(in_channels=256, out_channels=128) # 8x8
        self.conv2 = GenBlock(in_channels=128, out_channels=128) # 16x16
        self.conv3 = GenBlock(in_channels=128, out_channels=128) # 32x32
        self.conv4 = GenBlock(in_channels=128, out_channels=128) # 64x64
        self.conv5 = GenBlock(in_channels=128, out_channels=128) # 128x128

        self.progression = [
                           self.inp,
                           self.conv1,
                           self.conv2,
                           self.conv3,
                           self.conv4,
                           self.conv5]
        
        self.torgb = nn.ModuleList([
                            nn.ConvTranspose2d(in_channels=256, out_channels=3, kernel_size=1, stride=1, bias=gen_bias), # 4x4
                            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=1, stride=1, bias=gen_bias), # 8x8
                            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=1, stride=1, bias=gen_bias), # 16x16
                            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=1, stride=1, bias=gen_bias), # 32x32
                            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=1, stride=1, bias=gen_bias), # 64x64
                            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=1, stride=1, bias=gen_bias)]) # 128x128

        self.apply(self.weights_init)
        
    def forward(self, x, step, alpha):
        x = x.view(-1, self.noise_dim, 1, 1)
        
        for i in range(step+1):
            if 0 <= alpha < 1 and i == step and step > 0:
                prev_x = x
                prev_img = self.torgb[step-1](prev_x)
                prev_upscaled = F.interpolate(prev_img, scale_factor=2, mode="nearest")
                prev_upscaled = torch.tanh(prev_upscaled)
            x = self.progression[i](x)

        x = self.torgb[step](x)
        x = self.tanh(x)
        if 0 <= alpha < 1 and i == step:
                x = (1 - alpha) * prev_upscaled + alpha * x
        return x

    def weights_init(self, layer):
        if type(layer) in [nn.Conv2d, nn.ConvTranspose2d]:
            nn.init.kaiming_normal_(layer.weight)
        if type(layer) == nn.BatchNorm2d:
            nn.init.normal_(layer.weight.data, 1.0, 0.02)
        if type(layer) == nn.Linear:
            nn.init.xavier_normal_(layer.weight)