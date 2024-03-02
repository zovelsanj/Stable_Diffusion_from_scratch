import torch
import torch.nn as nn
from torch.nn import functional as F
from VAE.attention import selfAttention, crossAttention

class timeEmbedding(nn.Module):
    '''Time Embedding (Positional encoding) Block'''

    def __init__(self, num_embed):
        self.num_embed = num_embed
        super().__init__()

        self.linear_1 = nn.Linear(in_features=self.num_embed, out_features=self.num_embed*4)
        self.linear_2 = nn.Linear(in_features=self.num_embed*4, out_features=self.num_embed*4)

    def forward(self, x):
        '''
        `x` shape: (1, 320)
        '''
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)

        return x


class Upsample(nn.Module):
    '''Upsample Layer'''
    
    def __init__(self, channels):
        self.channels = channels
        super().__init__()

        self.conv_layer =  nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv_layer(x)
        return x
    

class UNetResidualBlock(nn.Module):
    '''U-Net Residual Block
    - Combines latents (Encoder output + CLIP output) with Time embeddings to feed to U-Net
    so that the output depends on these combination and not just on any one of them.'''

    def __init__(self, in_channels, out_channels, n_time=1280):
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__()

        self.group_norm_feature = nn.GroupNorm(num_groups=32, num_channels=self.in_channels)
        self.conv_feature = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(in_features=n_time, out_features=self.out_channels)

        self.group_norm_merged = nn.GroupNorm(num_groups=32, num_channels=self.out_channels)
        self.conv_merged = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, padding=1)

        if self.in_channels == self.out_channels:
            self.residual_layer =  nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, padding=0)

    def forward(self, features, time):
        '''
        `features` (latent) shape: (batch_size, 320, height/8, width/8)
        `time` shape: (1, 1280)
        '''
        residue = features
        features = self.group_norm_feature(features)
        features = F.silu(features)
        features = self.conv_feature(features)

        time = F.silu(time)
        time = self.linear_time(time)

        merged = features + time.unsqueeze(-1).unsqueeze(-1)    #time doesn't have batch_size and channels emeddings, so add with unsqueeze
        merged = self.group_norm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residue)
    

class UNetAttentionBlock(nn.Module):
    '''U-Net Attention Block'''

    def __init__(self, num_head, embedding_size, d_context=768):
        self.num_head = num_head
        self.embedding_size = embedding_size
        self.channels = self.num_head * self.embedding_size
        super().__init__()

        self.group_norm = nn.GroupNorm(num_groups=32, num_channels=self.channels, eps=1e-6)
        self.conv_input = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, padding=0)
        
        self.layernorm_1 = nn.LayerNorm(normalized_shape=self.channels)
        self.self_attention = selfAttention(num_heads=self.num_head, d_embed=self.embedding_size)

        self.layernorm_2 = nn.LayerNorm(normalized_shape=self.channels)
        self.cross_attention = crossAttention(num_heads=self.num_head, channels=self.channels, d_context=d_context, in_proj_bias=False)

        self.layernorm_3 = nn.LayerNorm(normalized_shape=self.channels)
        self.linear_gelu_1 = nn.Linear(in_features=self.channels, out_features=4*self.channels*2)
        self.linear_gelu_2 = nn.Linear(in_features=4*self.channels, out_features=self.channels)
        
        self.conv_ouput = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, padding=0)

    def forward(self, x, context):
        '''
        `x` (latent) shape: (batch_size, features, height, width)
        `context` shape: (batch_size, seq_len, d_embed)
        '''
        residue_long = x
        x = self.group_norm(x)
        x = self.conv_input(x)
        batch_size, features, height, width = x.shape
        x = x.view(batch_size, features, height*width)  #(batch_size, features, height, width) -> (batch_size, features, height*width)
        x = x.transpose(-1, -2) #(batch_size, features, height*width) -> (batch_size, height*width, features)

        # Normalization + Self-attention with skip connection
        residue_short = x
        x = self.layernorm_1(x)
        x = self.self_attention(x)
        x += residue_short

        # Normalization + Cross-attention with skip connection
        x = self.layernorm_2(x)
        x = self.cross_attention(x, context)
        x += residue_short
        residue_short = x

        x = self.layernorm_3(x)
        x, gate = self.linear_gelu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)

        x = self.linear_gelu_2(x)
        x += residue_short

        x = x.transpose(-1, -2) # (batch_size, height*width, features) -> (batch_size, features, height*width)
        x = x.view((batch_size, features, height, width))   #(batch_size, features, height*width) -> (batch_size, features, height, width) 

        x = self.conv_ouput(x)
        return x + residue_long


class UNetOutputlayer(nn.Module):
    '''U-Net Output Layer'''

    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__()

        self.group_norm = nn.GroupNorm(num_groups=32, num_channels=self.in_channels)
        self.conv_layer = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        '''
        `x` shape: (batch_size, 320, height/8, width/8)
        `output` shape: (batch_size, 4, height/8, width/8)
        '''
        x = self.group_norm(x)
        x = F.silu(x)
        x = self.conv_layer(x)
        return x
    

class switchSequential(nn.Sequential):
    '''switch sequential block is just like sequential block but can recognize the layer parameters'''
    def forward(self, latent, context, time):
        for layer in self:
            if isinstance(layer, UNetAttentionBlock):
                x = layer(latent, context, time)
            elif isinstance(layer, UNetResidualBlock):
                x = layer(latent, context)
            else:
                x = layer(latent)

        return x


class UNet(nn.Module):
    '''U-Net Block:

        - At Encoder:
          - Keep Reducing the size of image while increasing the features.
          - conv2d_1 output: (batch_size, 4, height/8, width/8)
          - conv2d_2 output: (batch_size, 320, height/16, width/16)
          - conv2d_3 output: (batch_size, 640, height/32, width/32)
        and so on.

        - At Decode:
          - Reverse of Encoder.
          - NOTE: due to skip connection, number of in_channels in decoder end is double of the corresponding encoder layer's out_channels
    '''

    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Module(
            [switchSequential(nn.Conv2d(in_channels=4, out_channels=320, kernel_size=3, padding=1)),
             switchSequential(UNetResidualBlock(in_channels=320, out_channels=320), UNetAttentionBlock(num_head=8, embedding_size=40)),
             switchSequential(UNetResidualBlock(in_channels=320, out_channels=320), UNetAttentionBlock(num_head=8, embedding_size=40)),

             switchSequential(nn.Conv2d(in_channels=320, out_channels=320, kernel_size=3, stride=2, padding=1)),
             switchSequential(UNetResidualBlock(in_channels=320, out_channels=640), UNetAttentionBlock(num_head=8, embedding_size=80)),
             switchSequential(UNetResidualBlock(in_channels=640, out_channels=640), UNetAttentionBlock(num_head=8, embedding_size=80)),

             switchSequential(nn.Conv2d(in_channels=640, out_channels=640, kernel_size=3, stride=2, padding=1)),
             switchSequential(UNetResidualBlock(in_channels=640, out_channels=1280), UNetAttentionBlock(num_head=8, embedding_size=160)),
             switchSequential(UNetResidualBlock(in_channels=1280, out_channels=1280), UNetAttentionBlock(num_head=8, embedding_size=160)),

             switchSequential(nn.Conv2d(in_channels=1280, out_channels=1280, kernel_size=3, stride=2, padding=1)),
             switchSequential(UNetResidualBlock(in_channels=1280, out_channels=1280)),
             switchSequential(UNetResidualBlock(in_channels=1280, out_channels=1280))
            ]
        )

        self.bottle_neck = switchSequential(
             switchSequential(UNetResidualBlock(in_channels=1280, out_channels=1280)),
             switchSequential(UNetAttentionBlock(num_head=8, embedding_size=160)),
             switchSequential(UNetResidualBlock(in_channels=1280, out_channels=1280))
        )

        self.decoder = nn.Module(
            [switchSequential(UNetResidualBlock(in_channels=2560, out_channels=1280)),    #Skip connection doubles in-channels in decoder end
             switchSequential(UNetResidualBlock(in_channels=2560, out_channels=1280)),
             switchSequential(UNetResidualBlock(in_channels=2560, out_channels=1280), Upsample(1280)),

             switchSequential(UNetResidualBlock(in_channels=2560, out_channels=1280), UNetAttentionBlock(num_head=8, embedding_size=160)),
             switchSequential(UNetResidualBlock(in_channels=2560, out_channels=1280), UNetAttentionBlock(num_head=8, embedding_size=160)),
             switchSequential(UNetResidualBlock(in_channels=1920, out_channels=1280), UNetAttentionBlock(num_head=8, embedding_size=160), Upsample(1280)),

             switchSequential(UNetResidualBlock(in_channels=1920, out_channels=640), UNetAttentionBlock(num_head=8, embedding_size=80)),
             switchSequential(UNetResidualBlock(in_channels=1280, out_channels=640), UNetAttentionBlock(num_head=8, embedding_size=80)),
             switchSequential(UNetResidualBlock(in_channels=960, out_channels=640), UNetAttentionBlock(num_head=8, embedding_size=80), Upsample(640)),

             switchSequential(UNetResidualBlock(in_channels=960, out_channels=320), UNetAttentionBlock(num_head=8, embedding_size=40)),
             switchSequential(UNetResidualBlock(in_channels=640, out_channels=320), UNetAttentionBlock(num_head=8, embedding_size=40)),
             switchSequential(UNetResidualBlock(in_channels=640, out_channels=320), UNetAttentionBlock(num_head=8, embedding_size=40))
            ]
            )


class Diffusion(nn.Module):
    '''Diffusion Model (basically U-Net)'''

    def __init__(self):
        self.time_embedding = timeEmbedding(320)   #U-Net also requires timestamps in addition to noises
        self.unet = UNet()
        self.final = UNetOutputlayer(320, 4)
    
    def forward(self, latent, context, time):
        '''
        `latent` size: (batch_size, 4, height/8, width/8)
        `context` size: (batch_size, seq_len, d_embed)
        `time` size: (embedding_num=1, embedding_size=320) => Positional Encoding
        '''
        time = self.time_embedding(time)    #(1, 320) -> (1, 1280)
        output = self.unet(latent, context, time)  #(batch_size, 4, height/8, width/8) -> (batch_size, 320, height/8, width/8)
        output = self.final(output) #(batch_size, 8, height/8, width/8) -> #(batch_size, 4, height/8, width/8)
        return output
    