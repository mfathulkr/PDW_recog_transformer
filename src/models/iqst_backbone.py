#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Defines the IQ Signal Transformer (IQST) Backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# No external local imports needed here

logger = logging.getLogger(__name__)

class IQSTBackbone(nn.Module):
    """Implementation of the IQ Signal Transformer backbone (IQST-S / IQST-L).
    
    Takes 2x512 IQ input, performs patch embedding, adds positional and shared
    embeddings, passes through a Transformer Encoder, and returns the processed
    shared embedding suitable for task heads.
    """
    # Pass config parameters directly
    def __init__(self, variant='S', input_channels=2, input_length=512,
                 embedding_dim=768, num_patches=8, 
                 s_encoder_layers=3, s_attention_heads=3, 
                 l_encoder_layers=6, l_attention_heads=9):
        super().__init__()
        self.variant = variant
        self.embedding_dim = embedding_dim
        self.num_patches = num_patches
        self.input_length = input_length # Original signal length per channel
        self.input_channels = input_channels  # I and Q

        # Adjust parameters for Large variant
        if variant == 'L':
            num_encoder_layers = l_encoder_layers
            num_heads = l_attention_heads
            logger.info(f"Initializing IQST-L: {num_encoder_layers} layers, {num_heads} heads.")
        else:
            num_encoder_layers = s_encoder_layers
            num_heads = s_attention_heads
            logger.info(f"Initializing IQST-S: {num_encoder_layers} layers, {num_heads} heads.")

        # 1. Patch Embedding Calculation
        # Input (B, 2, 512) -> Flatten to (B, 1024)
        total_features = self.input_channels * self.input_length # 1024
        if total_features % num_patches != 0:
             raise ValueError(f"Total features ({total_features}) not divisible by num_patches ({num_patches})")
        self.patch_size = total_features // num_patches # 1024 / 8 = 128
        logger.info(f"IQST Patch size: {self.patch_size}")
        self.patch_embedding = nn.Linear(self.patch_size, embedding_dim)

        # 2. Positional and Shared Embeddings
        # +1 for the shared (class-like) token
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embedding_dim))
        # Learnable token prepended to the sequence
        self.shared_embedding_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        # 3. Transformer Encoder
        # Standard Transformer Encoder Layer configuration
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim,
                                                 nhead=num_heads,
                                                 dim_feedforward=embedding_dim * 4, # Common heuristic
                                                 dropout=0.1, # Standard dropout rate
                                                 activation=F.gelu, # GELU activation as per paper
                                                 batch_first=True, # Expect (Batch, Seq, Feature)
                                                 norm_first=False) # Standard Post-LN transformer
        encoder_norm = nn.LayerNorm(embedding_dim) # Final layer norm
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 
                                                       num_layers=num_encoder_layers,
                                                       norm=encoder_norm)

        # Output shape for heads: Channels=embedding_dim, Length=1
        self.output_channels = self.embedding_dim
        self.output_length = 1

    def forward(self, x):
        # Input x shape: (batch_size, 2, 512)
        if x.shape[1] != self.input_channels or x.shape[2] != self.input_length:
            raise ValueError(f"IQSTBackbone expects input shape (B, {self.input_channels}, {self.input_length}), got {x.shape}")
        batch_size = x.shape[0]

        # Flatten I and Q channels: (B, 2, 512) -> (B, 1024)
        x = x.view(batch_size, -1)
        
        # Create patches: (B, 1024) -> (B, num_patches, patch_size)
        # unfold(dimension, size, step)
        x = x.unfold(dimension=1, size=self.patch_size, step=self.patch_size)
        
        # Apply patch embedding (linear projection): (B, num_patches, patch_size) -> (B, num_patches, embedding_dim)
        x = self.patch_embedding(x)

        # Prepend the shared embedding token: (B, 1, embedding_dim)
        # expand creates a view, not a copy
        shared_token = self.shared_embedding_token.expand(batch_size, -1, -1)
        x = torch.cat((shared_token, x), dim=1) # Shape: (B, num_patches + 1, embedding_dim)

        # Add positional embeddings (learned)
        # Positional embedding shape: (1, num_patches + 1, embedding_dim) - broadcasts across batch
        x = x + self.positional_embedding

        # Pass through Transformer Encoder
        transformer_output = self.transformer_encoder(x) # Shape: (B, num_patches + 1, embedding_dim)

        # Extract the output corresponding to the *processed* shared embedding token (first token)
        shared_output = transformer_output[:, 0] # Shape: (B, embedding_dim)

        # Reshape for TaskHead (expects Conv1d input: B, C, L)
        # Treat the embedding dim as channels (C), add a dummy length (L=1) dimension
        shared_output_reshaped = shared_output.unsqueeze(-1) # Shape: (B, embedding_dim, 1)
        
        return shared_output_reshaped 