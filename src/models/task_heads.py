#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Defines the Task-Specific Head for the MTL Model.
"""

import torch
import torch.nn as nn
import numpy as np
import logging

logger = logging.getLogger(__name__)

# --- Helper function to calculate Conv output size --- #
def calculate_conv_output_length(input_length, kernel_size, stride=1, padding=0, dilation=1):
    """Calculates the output length of a 1D convolution or pooling layer."""
    # Formula from PyTorch Conv1d documentation
    if input_length <= 0:
        return 0 # Avoid issues with invalid input length
    return int(np.floor(((input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1))

# --- Task-Specific Head --- #
class TaskHead(nn.Module):
    """A generic task-specific head for the MTL framework.

    Consists of Conv1d -> BatchNorm1d -> ReLU -> Dropout -> Flatten -> Linear -> Dropout.
    Output activation (Softmax) applied only if is_classification=True.
    """
    # Added backbone_output_length parameter to calculate Linear input size
    def __init__(self, input_channels, backbone_output_length, num_filters, output_dim, is_classification=False):
        super().__init__()
        self.is_classification = is_classification

        # Convolutional Layer (Using Kernel Size 3, Padding 1 as per paper's 3x3 hint adapted to 1D)
        self.conv1_kernel_size = 3
        self.conv1_padding = 1
        self.conv = nn.Conv1d(in_channels=input_channels, out_channels=num_filters, 
                              kernel_size=self.conv1_kernel_size, padding=self.conv1_padding)
        
        # Batch Normalization
        self.bn = nn.BatchNorm1d(num_filters)
        
        # Activation
        self.relu = nn.ReLU()
        
        # Dropout after Conv
        self.dropout_conv = nn.Dropout(0.25)
        
        # Flatten layer
        self.flatten = nn.Flatten()

        # Calculate the output length after Conv1d
        # Assuming stride=1, dilation=1 for the Conv1d layer
        conv_output_length = calculate_conv_output_length(
            input_length=backbone_output_length, 
            kernel_size=self.conv1_kernel_size, 
            padding=self.conv1_padding
        )
        
        # Calculate the input size to the dense layer
        dense_input_size = num_filters * conv_output_length
        if dense_input_size <= 0:
            # Log the problematic values
            logger.error(f"Calculated dense_input_size is non-positive ({dense_input_size}). "
                         f"Input channels: {input_channels}, Backbone output length: {backbone_output_length}, "
                         f"Num filters: {num_filters}, Conv output length: {conv_output_length}")
            raise ValueError(f"Calculated dense_input_size is non-positive ({dense_input_size}). Check backbone_output_length ({backbone_output_length}) and head parameters.")

        # Dense Layer (replace LazyLinear with explicit size)
        self.dense = nn.Linear(dense_input_size, output_dim)
        
        # Dropout after Dense
        self.dropout_dense = nn.Dropout(0.5)
        
        # Optional Softmax for classification
        if self.is_classification:
            # Use LogSoftmax + NLLLoss later for better numerical stability if needed,
            # but paper implies Softmax here and CrossEntropyLoss later.
            self.output_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # Input x shape expected: (batch_size, input_channels, sequence_length)
        if len(x.shape) != 3:
             raise ValueError(f"TaskHead expects input shape (B, C, L), but got {x.shape}")

        x = self.conv(x)      # (B, num_filters, conv_output_length)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout_conv(x)
        x = self.flatten(x)   # (B, num_filters * conv_output_length)
        x = self.dense(x)     # (B, output_dim)
        x = self.dropout_dense(x)
        
        if self.is_classification:
            x = self.output_activation(x)
            
        # Output for regression is (B, 1), for classification (B, num_classes)
        return x 