#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Defines the CNN-based Backbone Architectures (CNN1D, CNN2D).
"""

import torch
import torch.nn as nn
import logging
import numpy as np

# Import helper from sibling module
from .task_heads import calculate_conv_output_length

logger = logging.getLogger(__name__)

# --- Shared Backbones --- #

class CNN1DBackbone(nn.Module):
    """Implementation of the CNN1D backbone (Section 2.3).
    
    Consists of Conv1d(k=3, s=1, p=1) -> ReLU -> MaxPool1d(k=2, s=2) -> Dropout(0.25)
    """
    def __init__(self, input_channels=2, input_length=512, num_filters=8, kernel_size=3, pool_kernel_size=2):
        super().__init__()
        self.input_channels = input_channels
        self.input_length = input_length
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2 # Maintain length for conv, e.g., k=3, p=1
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_kernel_size # Standard max pool stride
        
        # Layers
        self.conv1 = nn.Conv1d(in_channels=self.input_channels, out_channels=self.num_filters, 
                               kernel_size=self.kernel_size, padding=self.padding)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=self.pool_kernel_size, stride=self.pool_stride)
        self.dropout = nn.Dropout(0.25)

        # Calculate output length after pooling
        # Length after conv (stride=1, padding=same -> length = input_length)
        conv_out_len = self.input_length
        # Length after pool (stride=pool_kernel_size)
        self.output_length = calculate_conv_output_length(
            input_length=conv_out_len, 
            kernel_size=self.pool_kernel_size, 
            stride=self.pool_stride, 
            padding=0 # MaxPool default padding
        )
        self.output_channels = self.num_filters
        logger.info(f"CNN1D backbone calculated output shape: (B, {self.output_channels}, {self.output_length})")

    def forward(self, x):
        # Input x shape: (batch_size, 2, 512)
        if x.shape[1] != self.input_channels or x.shape[2] != self.input_length:
            raise ValueError(f"CNN1DBackbone expects input shape (B, {self.input_channels}, {self.input_length}), got {x.shape}")
            
        x = self.conv1(x)       # (B, 8, 512)
        x = self.relu(x)
        x = self.pool(x)        # (B, 8, 256) assuming pool_kernel_size=2
        x = self.dropout(x)
        # Output shape: (batch_size, self.output_channels, self.output_length)
        return x

class CNN2DBackbone(nn.Module):
    """Implementation of the CNN2D backbone (Section 2.3).
    
    Input requires reshaping 2x512 -> 1x32x32 first.
    Consists of Conv2d(k=2, s=1, p=0) -> ReLU -> MaxPool2d(k=2, s=2) -> Dropout(0.25)
    """
    def __init__(self, input_channels=1, input_size=32, num_filters=8, kernel_size=2, pool_kernel_size=2):
        super().__init__()
        self.input_channels = input_channels
        self.input_size = input_size # Should be 32
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = 1
        self.padding = 0
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_kernel_size

        self.conv1 = nn.Conv2d(in_channels=self.input_channels, out_channels=self.num_filters, 
                               kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=self.pool_kernel_size, stride=self.pool_stride)
        self.dropout = nn.Dropout(0.25)

        # Calculate output shape (needed if we flatten later)
        # Input H, W = 32, 32
        conv_out_h = calculate_conv_output_length(self.input_size, self.kernel_size, self.stride, self.padding)
        conv_out_w = calculate_conv_output_length(self.input_size, self.kernel_size, self.stride, self.padding)
        # Pool output H, W
        self.output_h = calculate_conv_output_length(conv_out_h, self.pool_kernel_size, self.pool_stride, 0)
        self.output_w = calculate_conv_output_length(conv_out_w, self.pool_kernel_size, self.pool_stride, 0)
        self.output_channels = self.num_filters
        # Example output for k=2,p=0,s=1 conv -> 31x31. k=2,s=2 pool -> 15x15. Output shape (B, 8, 15, 15)
        logger.info(f"CNN2D backbone calculated output shape (before flatten): (B, {self.output_channels}, {self.output_h}, {self.output_w})")


    def forward(self, x_reshaped):
        # Input x_reshaped shape: (batch_size, 1, 32, 32)
        if x_reshaped.shape[1] != self.input_channels or x_reshaped.shape[2] != self.input_size or x_reshaped.shape[3] != self.input_size:
             raise ValueError(f"CNN2DBackbone expects input shape (B, {self.input_channels}, {self.input_size}, {self.input_size}), got {x_reshaped.shape}")
             
        x = self.conv1(x_reshaped) # (B, 8, 31, 31)
        x = self.relu(x)
        x = self.pool(x)        # (B, 8, 15, 15)
        x = self.dropout(x)
        # Output shape: (batch_size, self.output_channels, self.output_h, self.output_w)
        return x 