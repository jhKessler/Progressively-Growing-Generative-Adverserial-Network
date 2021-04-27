import torch
import torch.nn as nn
import torch.nn.functional as F
from models.discriminator.decisionblock import DecisionBlock
from models.discriminator.inpblock import InpBlock
from models.discriminator.discblock import DiscBlock

class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = DiscBlock(in_channels=128, out_channels=128) # 128x128
        self.conv2 = DiscBlock(in_channels=128, out_channels=128) # 64x64
        self.conv3 = DiscBlock(in_channels=128, out_channels=128) # 32x32
        self.conv4 = DiscBlock(in_channels=128, out_channels=256) # 16x16
        self.conv5 = DiscBlock(in_channels=256, out_channels=256) # 8x8
        self.outp = DecisionBlock(in_channels=256) # 4x4
        
        self.progression = [
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5
        ]
        
        self.fromrgb = nn.ModuleList([
            InpBlock(out_channels=256), # 4x4
            InpBlock(out_channels=256), # 8x8
            InpBlock(out_channels=128), # 16x16
            InpBlock(out_channels=128), # 32x32
            InpBlock(out_channels=128), # 64x64
            InpBlock(out_channels=128) # 128x128
            ])  

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
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)