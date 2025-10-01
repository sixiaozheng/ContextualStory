import torch
import torch.nn as nn
from diffusers.models.attention_processor import Attention

def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module

class TextAdapter(nn.Module):
    def __init__(self, zero_initialize=True):
        super(TextAdapter, self).__init__()
        
        # Define the layers of the CNN
        self.norm2 = nn.LayerNorm(1024)
        self.attn2 = Attention(
            query_dim=1024,
            cross_attention_dim=1024,
            heads=8,
            dim_head=128,
            dropout=0.0,
            bias=False,
            upcast_attention=False,
            out_bias=True,
        )  # is self-attn if encoder_hidden_states is none

        if zero_initialize:
            self.attn2.to_out = zero_module(self.attn2.to_out)
        
    def forward(self, x, encoder_hidden_states, attention_mask=None):
        # Forward pass through the layers
        norm_x = self.norm2(x)
        attn_output = self.attn2(
            norm_x, # [30, 1024, 320] 1024=H*W=32*32
            encoder_hidden_states=encoder_hidden_states, # [30, 77, 1024]
            attention_mask=attention_mask, # None
        )
        x = attn_output + x
        return x
    
