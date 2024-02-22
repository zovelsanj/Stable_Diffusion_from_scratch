import torch
import torch.nn as nn
from torch.nn import functional as F
from VAE.attention import selfAttention

class CLIP_embedding(nn.Module):
    '''CLIP Embedding Block for Text Prompt Embeddings'''
    
    def __init__(self, num_vocab, num_embed, num_tokens):
        '''
        `num_vocab` => vocabulary size
        `num_embed` => embedding size
        `num_tokens` => tokens size = maximum sequence length
        '''
        self.num_vocab = num_vocab
        self.num_embed = num_embed
        self.num_tokens = num_tokens
        super().__init__()

        self.token_embedding = nn.Embedding(num_embeddings=self.num_vocab, embedding_dim=self.num_embed)
        self.positional_embedding = nn.Parameter(data=torch.zeros(num_tokens, num_embed))

    def forward(self, tokens):
        x = self.token_embedding(tokens)
        x += self.positional_embedding
        return x

class CLIP_layer(nn.Module):
    '''CLIP Layer Block'''

    def __init__(self, num_head, num_embed):
        self.num_head = num_head
        self.num_embed = num_embed
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(normalized_shape=self.num_embed)
        self.attention = selfAttention(num_heads=self.num_head, d_embed=self.num_embed)

        self.layernorm_2 = nn.LayerNorm(normalized_shape=self.num_embed)

        self.linear_1 = nn.Linear(in_features=self.num_embed, out_features=self.num_embed*4)
        self.linear_2 = nn.Linear(in_features=self.num_embed*4, out_features=self.num_embed)

    def forward(self, x):
        '''
        Self Attention Layer: LayerNormalization + Multi-Head Attention + Residual Connection
        Feed Forward Layer: LayerNormalization + Linear Layers (GeLU) + Residual Connection
        '''
        #self attention layer
        residue = x
        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)
        x += residue 

        #feed forward layer
        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = F.gelu(x)
        x = self.linear_2(x)
        x += residue

        return x

class CLIP(nn.Module):
    '''CLIP Encoder Block'''

    def __init__(self):
        self.embedding = CLIP_embedding(num_vocab=49408, num_embed=768, num_tokens=77) #for pre-trained model
        self.clip_layers = nn.Module(
            [CLIP_layer(num_attention_heads=12, embedding_size=768) for i in range(12)]
        )
        self.layer_norm = nn.LayerNorm(normalized_shape=768)
    
    def forward(self, tokens):
        tokens = tokens.type(torch.long)
        state = self.embedding(tokens)  #convert tokens to embeddings

        for layer in self.clip_layers:
            state = layer(state)
        
        output = self.layer_norm(state)
        return output
