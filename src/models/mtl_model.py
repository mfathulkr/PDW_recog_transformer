# src/models/mtl_model.py

import torch
import torch.nn as nn
import logging

# Import necessary components from sibling modules
from .cnn_backbones import CNN1DBackbone, CNN2DBackbone
from .iqst_backbone import IQSTBackbone
from .task_heads import TaskHead

logger = logging.getLogger(__name__)

# --- Overall MTL Model --- #
class MTLModel(nn.Module):
    """Multi-Task Learning model combining a backbone and task-specific heads.

    Handles instantiation of the chosen backbone and the five task heads,
    and manages the forward pass including necessary reshaping.
    """
    # Pass config dictionaries for cleaner initialization
    def __init__(self, model_config, data_config, training_config):
        super().__init__()
        self.model_name = model_config['name']
        self.num_classes = model_config['num_classes']
        self.task_head_filters = model_config['task_head']['num_filters']
        logger.info(f"Initializing MTL Model with backbone: {self.model_name}")

        # --- Instantiate Backbone --- #
        if self.model_name == 'CNN1D':
            # Use backbone params from config if provided, else defaults
            cnn1d_params = model_config.get('cnn1d', {})
            self.backbone = CNN1DBackbone(
                input_channels=cnn1d_params.get('input_channels', 2),
                num_filters=cnn1d_params.get('num_filters', 8),
                kernel_size=cnn1d_params.get('kernel_size', 3),
                pool_kernel_size=cnn1d_params.get('pool_kernel_size', 2)
            )
            # output_channels and output_length are now attributes of the backbone
        elif self.model_name == 'CNN2D':
            # Use backbone params from config if provided, else defaults
            cnn2d_params = model_config.get('cnn2d', {})
            self.backbone = CNN2DBackbone(
                input_channels=cnn2d_params.get('input_channels', 1),
                num_filters=cnn2d_params.get('num_filters', 8),
                kernel_size=cnn2d_params.get('kernel_size', 2),
                pool_kernel_size=cnn2d_params.get('pool_kernel_size', 2)
            )
            # Output is (B, C, H, W), will need flattening before heads
        elif self.model_name == 'IQST-S':
            # Pass relevant IQST params from config
            iqst_params = model_config.get('iqst', {}) # Get IQST specific config
            self.backbone = IQSTBackbone(
                variant='S',
                embedding_dim=iqst_params.get('embedding_dim', 768), # Provide defaults
                num_patches=iqst_params.get('num_patches', 8),
                s_encoder_layers=iqst_params.get('s_encoder_layers', 3),
                s_attention_heads=iqst_params.get('s_attention_heads', 3)
                # L params are not needed here but could be included with defaults if structure was shared
            )
        elif self.model_name == 'IQST-L':
            iqst_params = model_config.get('iqst', {}) # Get IQST specific config
            self.backbone = IQSTBackbone(
                variant='L',
                embedding_dim=iqst_params.get('embedding_dim', 768), # Provide defaults
                num_patches=iqst_params.get('num_patches', 8),
                l_encoder_layers=iqst_params.get('l_encoder_layers', 6),
                # Correctly get 'l_num_mha_heads' from config, fall back to 9 if not present
                l_attention_heads=iqst_params.get('l_num_mha_heads', iqst_params.get('l_attention_heads', 12)) # Defaulted to 12 from config
                # S params are not needed here
            )
        else:
            raise ValueError(f"Unknown backbone name in config: {self.model_name}")

        # --- Optional: Projection layer for IQST to match paper's 128-dim head input ---
        self.projection_to_128_dim = None
        if self.model_name in ['IQST-S', 'IQST-L']:
            backbone_output_actual_dim = self.backbone.output_channels # Should be 768
            if backbone_output_actual_dim == 768: # Only add projection if backbone outputs 768
                self.projection_to_128_dim = nn.Linear(backbone_output_actual_dim, 128)
                logger.info(f"Added projection layer for {self.model_name} from {backbone_output_actual_dim} to 128 dimensions for task heads.")
                backbone_output_channels_for_heads = 128
                backbone_output_length_for_heads = 1 # Length remains 1 after projection and reshape
            else: # If IQST output is not 768 for some reason, use its direct output dim
                logger.warning(f"{self.model_name} output channels ({backbone_output_actual_dim}) not 768. Using direct output for heads.")
                backbone_output_channels_for_heads = backbone_output_actual_dim
                backbone_output_length_for_heads = self.backbone.output_length
            self.needs_flattening = False # IQST output is already (B,C,L)-like after backbone
        # --- Determine Input Shape for Heads --- #
        # Heads expect (B, C, L) input for Conv1d
        elif self.model_name == 'CNN2D':
            # Output is (B, C, H, W) -> Flatten C*H*W -> Add Length dim L=1
            backbone_output_channels_for_heads = self.backbone.output_channels * self.backbone.output_h * self.backbone.output_w
            backbone_output_length_for_heads = 1 # Treat flattened features as having length 1
            self.needs_flattening = True # Flag to flatten CNN2D output in forward pass
        else: # CNN1D, IQST-S, IQST-L
            # Output is already (B, C, L) or reshaped to it in backbone's forward
            backbone_output_channels_for_heads = self.backbone.output_channels
            backbone_output_length_for_heads = self.backbone.output_length
            self.needs_flattening = False

        logger.info(f"Input for Task Heads (Channels, Length): ({backbone_output_channels_for_heads}, {backbone_output_length_for_heads})")

        # --- Instantiate Task Heads --- #
        head_input_channels = backbone_output_channels_for_heads

        # Regression Heads (output_dim=1)
        self.head_np = TaskHead(head_input_channels, backbone_output_length_for_heads, self.task_head_filters, 1, is_classification=False)
        self.head_pw = TaskHead(head_input_channels, backbone_output_length_for_heads, self.task_head_filters, 1, is_classification=False)
        self.head_pri = TaskHead(head_input_channels, backbone_output_length_for_heads, self.task_head_filters, 1, is_classification=False)
        self.head_td = TaskHead(head_input_channels, backbone_output_length_for_heads, self.task_head_filters, 1, is_classification=False)

        # Classification Head (output_dim = num_classes)
        self.head_class = TaskHead(head_input_channels, backbone_output_length_for_heads, self.task_head_filters, self.num_classes, is_classification=True)

        # --- Initialize Weights --- #
        # Apply LeCun initialization as specified in paper (Section 3.1)
        logger.info("Applying LeCun Normal weight initialization...")
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Applies LeCun Normal initialization to Conv and Linear layers."""
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            # LeCun Normal initialization: N(0, std^2) with std = sqrt(1 / fan_in)
            # kaiming_normal_ with a=0 and nonlinearity='relu' is equivalent for fan_in mode
            # Using 'relu' nonlinearity as it's common and works similarly for a=0
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
             # Initialize LayerNorm parameters
             nn.init.constant_(module.bias, 0)
             nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
            # Initialize BatchNorm parameters
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Parameter):
             # Heuristic: Initialize standalone Parameter tensors if they exist (like positional embeddings)
             # Use a standard normal distribution for these, unless specific initialization is known
             if module.dim() > 1: # Avoid initializing scalar parameters if any
                 nn.init.normal_(module, std=0.02) # Common initialization for embeddings


    def forward(self, x):
        """Performs the forward pass through the backbone and heads.

        Args:
            x (torch.Tensor): Input tensor. Shape: (batch, 2, 512) or (batch, 1, 32, 32).
                              Shape depends on the dataset preprocessing for the chosen model.

        Returns:
            dict: A dictionary containing the output tensors from each task head.
                  Keys: 'class', 'np', 'pw', 'pri', 'td'. Values are raw outputs (logits for class).
        """
        batch_size = x.shape[0]

        # --- Input Reshaping (for CNN2D) --- #
        # Assume input 'x' is already in the correct format required by the backbone
        # (B, 2, 512) for CNN1D, IQST
        # (B, 1, 32, 32) for CNN2D (preprocessing should handle this)
        if self.model_name == 'CNN2D':
            if x.shape != (batch_size, 1, 32, 32):
                 logger.warning(f"Input shape {x.shape} for CNN2D might be unexpected. Expected (B, 1, 32, 32). Ensure data preprocessing matches.")
                 # Attempt to reshape if it looks like the raw (B, 2, 512) format
                 if x.shape == (batch_size, 2, 512):
                     logger.info("Reshaping input from (B, 2, 512) to (B, 1, 32, 32) inside CNN2D forward path.")
                     if x.numel() // batch_size != 1024:
                         raise ValueError(f"Cannot reshape input of size {x.shape} to (-, 1, 32, 32) for CNN2D")
                     x = x.view(batch_size, -1).view(batch_size, 1, 32, 32)
                 # else: # If it's neither expected shape, raise error after logging warning
                 #    raise ValueError(f"Unexpected input shape for CNN2D: {x.shape}. Expected (B, 1, 32, 32) or (B, 2, 512).")

        # --- Pass through Backbone --- #
        features = self.backbone(x) # CNN1D/IQST output (B, C, L), CNN2D output (B, C, H, W)

        # --- Apply projection if defined (for IQST models) ---
        if self.projection_to_128_dim is not None and self.model_name in ['IQST-S', 'IQST-L']:
            # features from IQST backbone are (B, 768, 1)
            squeezed_features = features.squeeze(-1) # (B, 768)
            projected_features = self.projection_to_128_dim(squeezed_features) # (B, 128)
            features_for_heads = projected_features.unsqueeze(-1) # (B, 128, 1)
        # --- Flatten features if needed (for CNN2D before heads) --- #
        elif self.needs_flattening:
            # Input features are (B, C, H, W)
            # Flatten C, H, W dims, keep Batch dim
            features_for_heads = features.view(features.size(0), -1) # (B, C*H*W)
            # Add a dummy Length dimension for Conv1d heads: (B, C*H*W, 1)
            features_for_heads = features_for_heads.unsqueeze(-1)
            # features_for_heads now shape (B, backbone_output_channels_for_heads, 1)
        else: # For CNN1D or IQST if projection wasn't applied
            features_for_heads = features

        # --- Pass features through Task Heads --- #
        # Heads expect input shape (B, Channels, Length)
        # features_for_heads shape: (B, head_input_channels, backbone_output_length_for_heads)
        output_class = self.head_class(features_for_heads) # Output (B, num_classes)
        output_np = self.head_np(features_for_heads)       # Output (B, 1)
        output_pw = self.head_pw(features_for_heads)       # Output (B, 1)
        output_pri = self.head_pri(features_for_heads)     # Output (B, 1)
        output_td = self.head_td(features_for_heads)       # Output (B, 1)

        # Return dictionary of outputs (remove trailing dim from regression outputs)
        return {
            'class': output_class,
            'np': output_np.squeeze(-1),
            'pw': output_pw.squeeze(-1),
            'pri': output_pri.squeeze(-1),
            'td': output_td.squeeze(-1)
        } 