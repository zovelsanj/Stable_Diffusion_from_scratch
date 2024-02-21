import torch
import torch.nn as nn
from torch.nn import functional as F
from attention import selfAttention

class VAE_attentionBlock(nn.Module):
    '''Variational Autoencoder Attention Block'''

    def __init__(self, channels):
        self.channels = channels
        super().__init__()
        self.group_norm = nn.GroupNorm(32, self.channels)
        self.attention = selfAttention(1, self.channels)
        
    def forward(self, x):
        residue = x
        num_batch, channles, height, width = x.shape
        x = x.view(num_batch, channles, height*width)
        x = x.transpose(-1, -2)
        x = self.attention(x)
        x = x.transpose(-1, -2)
        x = x.view((num_batch, channles, height, width))
        
        return x + residue
    
class VAE_residualBlock(nn.Module):
    '''Variational Autoencoder Residual Block'''
    def __init__(self, in_channels, out_channels):
        self.in_channles = in_channels
        self.out_channles = out_channels
        super().__init__()
        
        self.group_norm_1 = nn.GroupNorm(32, self.in_channels)
        self.conv_1 = nn.Conv2d(self.in_channles, self.out_channles, kernel_size=3, padding=1)
        
        self.group_norm_2 = nn.GroupNorm(32, self.in_channels)
        self.conv_2 = nn.Conv2d(self.in_channles, self.out_channles, kernel_size=3, padding=1)
        
        if self.in_channels == self.out_channles:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(self.in_channles, self.out_channles, kernel_size=1, padding=0)
            
    def forwad(self, x):
        '''
        `x` shape: (batch_size, in_channels, height, width)
        '''
        residue = x
        x = self.group_norm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.group_norm_2(x)
        
        return x + self.residual_layer(residue)

class VAE_decoder(nn.Sequential):
    '''Variational Autoencoder Decoder Block'''

    def __init__(self):
        '''
        input shape: (batch_size, 512, height/8, width/8)
        Upsample_1 output: (batch_size, 512, height/4, width/4)
        Upsample_2 output: (batch_size, 512, height/2, width/2)
        VAE_residualBlock (3rd group) output: (batch_size, 256, height/2, width/2)
        Upsample_3 output: (batch_size, 256, height, width)
        VAE_residualBlock (4th group) output: (batch_size, 128, height/2, width/2)
        GroupNorm: divides 128 channels to 32 groups
        Conv2d_last output:(batch_size, 3, height, width)
        '''
        super.__init__(
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=1, padding=0),
            nn.Conv2d(in_channels=4, out_channels=512, kernel_size=3, padding=1),
            
            VAE_residualBlock(in_channels=512, out_channels=512),
            VAE_attentionBlock(channels=512),
            VAE_residualBlock(in_channels=512, out_channels=512),
            VAE_residualBlock(in_channels=512, out_channels=512),
            VAE_residualBlock(in_channels=512, out_channels=512),
            VAE_residualBlock(in_channels=512, out_channels=512),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            VAE_residualBlock(in_channels=512, out_channels=512),
            VAE_residualBlock(in_channels=512, out_channels=512),
            VAE_residualBlock(in_channels=512, out_channels=512),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            VAE_residualBlock(in_channels=512, out_channels=256),
            VAE_residualBlock(in_channels=256, out_channels=256),
            VAE_residualBlock(in_channels=256, out_channels=256),
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            VAE_residualBlock(in_channels=256, out_channels=128),
            VAE_residualBlock(in_channels=128, out_channels=128),
            VAE_residualBlock(in_channels=128, out_channels=128),

            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.SiLU(),

            nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        '''
        `x` shape: (batch_size, 512, height/8, width/8)
        '''
        x /= 0.18215    #compensate the scaling at encoder end
        for module in self:
            x = module(x)
        
        return x
