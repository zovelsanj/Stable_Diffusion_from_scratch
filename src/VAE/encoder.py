import torch
from torch import nn
from torch.nn import functional as F

from decoder import VAE_residualBlock, VAE_attentionBlock

class VAE_Encoder(nn.Sequential):
    '''Variational Autoencoder Encoder Block'''

    def __init__(self, in_channels=3, out_channels=128, kernel_size=3, stride=2, padding=1):
        '''
        input shape: (batch_size, in_channels=3, height, width)
        Conv2d_1 output: (batch_size, out_channels=128, height, width)
        VAE_residualBlock_1 output: (batch_size, out_channels=128, height, width)
        VAE_residualBlock_2 output: (batch_size, out_channels=128, height, width)
        
        Conv2d_2 output: (batch_size, out_channels=128, height/2, width/2)
        VAE_residualBlock_3 output: (batch_size, out_channels=256, height/2, width/2)
        VAE_residualBlock_4 output: (batch_size, out_channels=256, height/2, width/2)
        
        and so on.
        
        VAE_attentionBlock output: (batch_size, out_channels=256, height/4, width/4)
        
        '''
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        super().__init__(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, padding=self.padding),
            VAE_residualBlock(self.out_channels, self.out_channels),
            VAE_residualBlock(self.out_channels, self.out_channels),
            
            nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=0),
            VAE_residualBlock(self.out_channels, 256),
            VAE_residualBlock(256, 256),
            
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=self.kernel_size, stride=self.stride, padding=0),
            VAE_residualBlock(256, 512),
            VAE_residualBlock(512, 512),
            
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=self.kernel_size, stride=self.stride, padding=0),
            VAE_residualBlock(512, 512),
            VAE_residualBlock(512, 512),
            VAE_residualBlock(512, 512),
            
            VAE_attentionBlock(512),
            VAE_residualBlock(512, 512),
            nn.GroupNorm(num_groups=32, num_channels=512),  #divides 512 channels to 32 groups
            nn.SiLU(),
            
            nn.Conv2d(in_channels=512, out_channels=8, kernel_size=self.kernel_size, padding=1),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1, padding=1)
        )
    
    def forward(self, x, noise):
        '''
        `x` shape: (batch_size, in_channels, height, width)
        `noise` shape: (batch_size, out_channels, height, width)
        '''
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))  #(pad_left=0, pad_right=1, pad_up=0, pad_down=1) => asymmetrical padding
            x = module(x)
        
        mean, log_var = torch.chunk(x, 2, dim=1)    #(batch_size, 8, height/8, width/8) => two tensor of shape (batch_size, 4, height/8, width/8)
        log_var = torch.clamp(log_var, -30, 20)
        variance = log_var.exp()
        sd = variance.sqrt()
        
        x = mean + sd * noise   #transform N(0, 1) to N(mean, variance)
        return x*0.18215    #return scaled output (constant as per the paper)
        