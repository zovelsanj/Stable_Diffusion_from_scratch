import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class selfAttention(nn.Module):
    '''Self Attention Block'''

    def __init__(self, num_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        self.num_heads = num_heads
        self.d_embed = d_embed
        self.in_proj_bias = in_proj_bias
        self.out_proj_bias = out_proj_bias
        super().__init__()
        
        self.in_proj = nn.Linear(self.d_embed, 3*self.d_embed, bias=self.in_proj_bias) #For Q, K and V
        self.out_proj = nn.Linear(self.d_embed, self.d_embed, bias=self.out_proj_bias)
        self.d_head = self.d_embed // self.num_heads
    
    def forward(self, x, causal_mask=False):
        '''
        `x` (latent) shape: (batch_size, seq_len, d_embed)
        '''
        batch_size, seq_len, d_embed = x.shape
        intermediate_shape = (batch_size, seq_len, self.num_heads, self.d_head)
        Q, K, V = self.in_proj(x).chunk(3, dim=-1)
        
        #(batch_size, seq_len, d_embed) -> (batch_size, seq_len, H, d_embed/H) -> (batch_size, H, seq_len, d_embed/H)
        Q = Q.view(intermediate_shape).transpose(1, 2)
        K = K.view(intermediate_shape).transpose(1, 2)
        V = V.view(intermediate_shape).transpose(1, 2)
        
        self.weight = Q @ K.transpose(-1, -2)
        
        if causal_mask:
            #mask where upper triangle is 1
            mask = torch.ones_like(self.weight, dtype=torch.bool).triu(1)
            self.weight.masked_fill_(mask, - torch.inf)

        self.weight /= math.sqrt(self.d_head)
        self.weight =  F.softmax(self.weight, dim=1)
        
        output = self.weight @ V
        output = output.transpose(1, 2)
        output = output.reshape(x.shape)
        output = self.out_proj(output)
        
        return output
    

class crossAttention(nn.Module):
    '''Cross Attention Block'''

    def __init__(self, num_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        self.num_heads = num_heads
        self.d_embed = d_embed
        self.d_cross = d_cross
        self.in_proj_bias = in_proj_bias
        self.out_proj_bias = out_proj_bias
        super().__init__()

        self.Q_proj = nn.Linear(self.d_embed, self.d_embed, bias=self.in_proj_bias) #For Q
        self.K_proj = nn.Linear(self.d_embed, self.d_embed, bias=self.in_proj_bias) #For K
        self.V_proj = nn.Linear(self.d_embed, self.d_embed, bias=self.in_proj_bias) #For V
        self.out_proj = nn.Linear(self.d_embed, self.d_embed, bias=self.out_proj_bias)
        self.d_head = self.d_embed // self.num_heads
        
    def forward(self, x, y):
        '''
        `x` (latent) shape: (batch_size, seq_len_Q, dim_Q)
        `y` (context) shape: (batch_size, seq_len_KV, dim_KV) = (batch_size, 77, 768)
        '''
        batch_size, seq_len, d_embed = x.shape
        intermediate_shape = (batch_size, seq_len, self.num_heads, self.d_head)
        Q, K, V = self.in_proj(x).chunk(3, dim=-1)
        
        #(batch_size, seq_len, d_embed) -> (batch_size, seq_len, H, d_embed/H) -> (batch_size, H, seq_len, d_embed/H)
        Q = self.Q_proj(x).view(intermediate_shape).transpose(1, 2)
        K = self.K_proj(y).view(intermediate_shape).transpose(1, 2)
        V = self.V_proj(y).view(intermediate_shape).transpose(1, 2)

        self.weight = Q @ K.transpose(-1, -2)

        self.weight /= math.sqrt(self.d_head)
        self.weight =  F.softmax(self.weight, dim=1)
        
        output = self.weight @ V
        output = output.transpose(1, 2).continuous()
        output = output.reshape(x.shape)
        output = self.out_proj(output)
        
        return output
