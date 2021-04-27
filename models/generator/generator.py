import torch
import torch.nn as nn
import torch.nn.functional as F
from models.generator.genblock import GenBlock

class Generator(nn.Module):

    def __init__(self, noise_dim):
        assert noise_dim > 0
        super().__init__()
        self.noise_dim = noise_dim
        self.inp = GenBlock(in_channels=self.noise_dim, out_channels=256, upsample=False, inp=True) # 4x4
        self.conv1 = GenBlock(in_channels=256, out_channels=256) # 8x8
        self.conv2 = GenBlock(in_channels=256, out_channels=128) # 16x16
        self.conv3 = GenBlock(in_channels=128, out_channels=128) # 32x32
        self.conv4 = GenBlock(in_channels=128, out_channels=128) # 64x64
        self.conv5 = GenBlock(in_channels=128, out_channels=128) # 128x128

        self.progression = [
                           self.conv1,
                           self.conv2,
                           self.conv3,
                           self.conv4,
                           self.conv5]
        
        self.torgb = nn.ModuleList([
                            nn.ConvTranspose2d(in_channels=256, out_channels=3, kernel_size=1, stride=1, bias=False), # 4x4
                            nn.ConvTranspose2d(in_channels=256, out_channels=3, kernel_size=1, stride=1, bias=False), # 8x8
                            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=1, stride=1, bias=False), # 16x16
                            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=1, stride=1, bias=False), # 32x32
                            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=1, stride=1, bias=False), # 64x64
                            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=1, stride=1, bias=False)]) # 128x128

        self.apply(self.weights_init)
        
    def forward(self, x, step, alpha):
        x = x.view(-1, self.noise_dim, 1, 1)
        x = self.inp(x)
        for i in range(step):
            if 0 <= alpha < 1 and i == step and step > 0:
                prev_x = x
                prev_img = self.torgb[step-1](prev_x)
                prev_upscaled = F.interpolate(prev_img, scale_factor=2, mode="nearest")
                
            x = self.progression[i](x)

        x = self.torgb[step](x)
        if 0 <= alpha < 1 and i == step:
                x = (1 - alpha) * prev_upscaled + alpha * x
        return x

    def weights_init(self, layer):
        if type(layer) in [nn.Conv2d, nn.ConvTranspose2d]:
            nn.init.kaiming_normal_(layer.weight)
        if type(layer) == nn.Linear:
            nn.init.xavier_normal_(layer.weight)