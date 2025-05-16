#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Handles the training process for the MTL model.

Includes functions for:
- Setting up the model, optimizer, and loss functions.
- Running the main training loop over epochs and batches.
- Calculating the combined multi-task loss.
- Performing backpropagation and optimizer steps.
- Running validation loops.
- Logging metrics to TensorBoard.
- Saving model checkpoints.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import time
import logging
from tqdm import tqdm # For progress bars
import yaml
import numpy as np # Added for MAE calculation

# Import necessary components from other modules
from src.data_handling.dataset import get_dataloaders, NUM_CLASSES
from src.models import MTLModel # Updated import
from src.utils.helpers import denormalize # For potential validation metric calculation

logger = logging.getLogger(__name__)

# Define regression keys here or import if available elsewhere
REGRESSION_KEYS = ['np', 'pw', 'pri', 'td']

def setup_optimizer(model, config):
    """Sets up the optimizer based on the configuration.

    Args:
        model (nn.Module): The model to optimize.
        config (dict): The training configuration dictionary.

    Returns:
        torch.optim.Optimizer: The configured optimizer.
    """
    optimizer_config = config['training']['optimizer']
    optimizer_type = optimizer_config['type'].lower()
    lr = optimizer_config['learning_rate']

    if optimizer_type == 'adam':
        # Add other Adam parameters (betas, eps, weight_decay) if specified in config
        optimizer = optim.Adam(model.parameters(), lr=lr)
        logger.info(f"Using Adam optimizer with learning rate: {lr}")
    # Add other optimizers like SGD if needed
    # elif optimizer_type == 'sgd':
    #     momentum = optimizer_config.get('momentum', 0.9)
    #     optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    #     logger.info(f"Using SGD optimizer with learning rate: {lr}, momentum: {momentum}")
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_config['type']}")

    return optimizer

def setup_loss_functions(config):
    """Sets up the individual loss functions and weights.

    Args:
        config (dict): The training configuration dictionary.

    Returns:
        tuple: (classification_criterion, regression_criterion, loss_weights)
               - classification_criterion: Loss function for classification.
               - regression_criterion: Loss function for regression.
               - loss_weights: Dictionary of weights for each task loss.
    """
    # Classification Loss (Categorical Cross-Entropy)
    # Note: CrossEntropyLoss expects raw logits from the model
    classification_criterion = nn.CrossEntropyLoss()
    logger.info("Using CrossEntropyLoss for classification.")

    # Regression Loss (L1 / MAE)
    regression_criterion = nn.L1Loss()
    logger.info("Using L1Loss (MAE) for regression.")

    # Loss Weights
    # Ensure all expected keys exist in the config or provide defaults
    loss_weights_config = config['training']['loss_weights']
    loss_weights = {
        'class': loss_weights_config.get('classification', 0.1), # Default if missing
        'np': loss_weights_config.get('regression_np', 0.225),
        'pw': loss_weights_config.get('regression_pw', 0.225),
        'pri': loss_weights_config.get('regression_pri', 0.225),
        'td': loss_weights_config.get('regression_td', 0.225)
    }
    logger.info(f"Using loss weights: {loss_weights}")
    # Verify weights sum approximately to 1 (as per paper's strategy)
    total_weight = sum(loss_weights.values())
    if not abs(total_weight - 1.0) < 1e-6:
         logger.warning(f"Sum of loss weights ({total_weight}) is not 1.0.")

    return classification_criterion, regression_criterion, loss_weights

def calculate_mtl_loss(outputs, targets, class_criterion, reg_criterion, weights):
    """Calculates the combined Multi-Task Learning loss.

    Args:
        outputs (dict): Dictionary of model outputs for each task.
        targets (dict): Dictionary of ground truth labels for each task (regression labels are normalized).
        class_criterion: The classification loss function.
        reg_criterion: The regression loss function.
        weights (dict): Dictionary of weights for each task loss.

    Returns:
        tuple: (total_loss, individual_losses)
               - total_loss (torch.Tensor): The weighted combined loss.
               - individual_losses (dict): Dictionary of unweighted losses for each task.
    """
    # Classification loss
    loss_class = class_criterion(outputs['class'], targets['class']) # Target should be integer class index

    # Regression losses (targets are normalized, outputs are predictions in [0,1] range)
    loss_np = reg_criterion(outputs['np'], targets['np'])
    loss_pw = reg_criterion(outputs['pw'], targets['pw'])
    loss_pri = reg_criterion(outputs['pri'], targets['pri'])
    loss_td = reg_criterion(outputs['td'], targets['td'])

    # Combine losses with weights
    total_loss = (weights['class'] * loss_class) + \
                 (weights['np'] * loss_np) + \
                 (weights['pw'] * loss_pw) + \
                 (weights['pri'] * loss_pri) + \
                 (weights['td'] * loss_td)

    individual_losses = {
        'class': loss_class.item(),
        'np': loss_np.item(),
        'pw': loss_pw.item(),
        'pri': loss_pri.item(),
        'td': loss_td.item(),
        'total': total_loss.item()
    }

    return total_loss, individual_losses

def train_one_epoch(model, dataloader, optimizer, class_criterion, reg_criterion, loss_weights, device):
    """Runs a single training epoch.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): DataLoader for the training set.
        optimizer (torch.optim.Optimizer): The optimizer.
        class_criterion: Classification loss function.
        reg_criterion: Regression loss function.
        loss_weights (dict): Weights for task losses.
        device (torch.device): The device to train on ('cuda' or 'cpu').

    Returns:
        dict: Dictionary containing average losses for the epoch.
    """
    model.train() # Set model to training mode
    # Initialize accumulators for all expected keys from loss_weights + 'total'
    task_keys = list(loss_weights.keys()) # Explicitly convert to list
    all_keys_for_accumulation = task_keys + ['total'] # Concatenate lists
    epoch_losses = {key: 0.0 for key in all_keys_for_accumulation} # Create the dictionary

    # Wrap dataloader with tqdm for a progress bar
    pbar = tqdm(dataloader, desc="Training Epoch", leave=False)
    for i, (iq_batch, label_batch) in enumerate(pbar):
        # Move data to the target device
        iq_batch = iq_batch.to(device)
        # Move each label tensor in the dictionary to the device
        label_batch = {k: v.to(device) for k, v in label_batch.items() if isinstance(v, torch.Tensor)}
        # Keep non-tensor labels (like SNR) on CPU if they exist
        label_batch.update({k: v for k, v in label_batch.items() if not isinstance(v, torch.Tensor)})

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(iq_batch)

        # Calculate loss
        total_loss, individual_losses = calculate_mtl_loss(
            outputs, label_batch, class_criterion, reg_criterion, loss_weights
        )

        # Backward pass
        total_loss.backward()

        # Optimizer step
        optimizer.step()

        # Accumulate losses for epoch average
        for key, loss_val in individual_losses.items():
            epoch_losses[key] += loss_val

        # Update progress bar description
        pbar.set_postfix(loss=individual_losses['total'])

    # Calculate average losses for the epoch
    num_batches = len(dataloader)
    avg_losses = {key: val / num_batches for key, val in epoch_losses.items()}

    return avg_losses

def validate_one_epoch(model, dataloader, class_criterion, reg_criterion, loss_weights, device, dataset_stats):
    """Runs a single validation epoch.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for the validation set.
        class_criterion: Classification loss function.
        reg_criterion: Regression loss function.
        loss_weights (dict): Weights for task losses.
        device (torch.device): The device to evaluate on ('cuda' or 'cpu').
        dataset_stats (dict): Statistics from the training set (for de-normalization).

    Returns:
        dict: Dictionary containing average validation losses, accuracy, and de-normalized MAEs.
    """
    model.eval() # Set model to evaluation mode
    # Initialize accumulators for all expected keys from loss_weights + 'total'
    # Ensure keys for individual losses, 'total' loss, and 'accuracy' are present
    val_metric_keys = list(loss_weights.keys()) + ['total', 'accuracy']
    val_epoch_metrics = {key: 0.0 for key in val_metric_keys}
    # Add accumulators for de-normalized MAEs
    for reg_key in REGRESSION_KEYS:
        val_epoch_metrics[f'mae_denorm_{reg_key}'] = 0.0

    correct_classifications = 0
    total_samples = 0

    pbar = tqdm(dataloader, desc="Validation Epoch", leave=False)
    with torch.no_grad(): # Disable gradient calculations
        for i, (iq_batch, label_batch) in enumerate(pbar):
            # Move data to the target device
            iq_batch = iq_batch.to(device)
            label_batch_device = {k: v.to(device) for k, v in label_batch.items() if isinstance(v, torch.Tensor)}
            # label_batch_device.update({k: v for k, v in label_batch.items() if not isinstance(v, torch.Tensor)}) # SNR etc.

            # Forward pass
            outputs = model(iq_batch)

            # Calculate loss (these are normalized losses)
            batch_total_loss, batch_individual_losses = calculate_mtl_loss(
                outputs, label_batch_device, class_criterion, reg_criterion, loss_weights
            )

            # Accumulate normalized losses
            for key, loss_val in batch_individual_losses.items():
                val_epoch_metrics[key] += loss_val # loss_val is already .item()

            # Calculate classification accuracy
            _, predicted_classes = torch.max(outputs['class'], 1) # Get the index of the max log-probability
            correct_classifications += (predicted_classes == label_batch_device['class']).sum().item()
            batch_total_samples = iq_batch.size(0)
            total_samples += batch_total_samples

            # Calculate and accumulate de-normalized MAE for regression tasks
            for reg_key in REGRESSION_KEYS:
                pred_norm = outputs[reg_key].detach().cpu().numpy()
                target_norm = label_batch_device[reg_key].detach().cpu().numpy()

                min_val = dataset_stats['label_min_max'][reg_key]['min']
                max_val = dataset_stats['label_min_max'][reg_key]['max']

                # Ensure denormalize handles arrays if pred_norm/target_norm are arrays
                pred_denorm = denormalize(pred_norm, min_val, max_val)
                target_denorm = denormalize(target_norm, min_val, max_val)
                
                # Sum of absolute errors for the batch
                sum_abs_error_denorm = np.sum(np.abs(pred_denorm - target_denorm))
                val_epoch_metrics[f'mae_denorm_{reg_key}'] += sum_abs_error_denorm

            # Update progress bar
            pbar.set_postfix(loss=batch_individual_losses['total'])

    # Calculate average losses and accuracy for the epoch
    num_batches = len(dataloader)
    avg_metrics = {key: val / num_batches for key, val in val_epoch_metrics.items() if key not in ['accuracy'] + [f'mae_denorm_{reg_key}' for reg_key in REGRESSION_KEYS]}
    avg_metrics['accuracy'] = correct_classifications / total_samples if total_samples > 0 else 0.0
    
    # Calculate average de-normalized MAEs
    for reg_key in REGRESSION_KEYS:
        avg_metrics[f'mae_denorm_{reg_key}'] = val_epoch_metrics[f'mae_denorm_{reg_key}'] / total_samples if total_samples > 0 else 0.0
        # Convert to microseconds (µs) if original values were in seconds for pw, pri, td
        if reg_key in ['pw', 'pri', 'td']:
            avg_metrics[f'mae_denorm_{reg_key}'] *= 1e6

    return avg_metrics

def train_model(config: dict, device: torch.device):
    """Orchestrates the entire model training process."""
    logger.info("Starting model training...")

    # --- Configuration Extraction ---
    # Extract relevant configuration sections
    data_config = config['data']
    model_config = config['model']
    training_config = config['training']
    logging_config = config['logging']

    # --- Setup TensorBoard --- #
    log_dir = logging_config['log_dir']
    # Create a specific subdirectory for this run (e.g., based on timestamp or model name)
    writer = SummaryWriter(log_dir=log_dir)
    logger.info(f"TensorBoard logs will be saved to: {log_dir}")

    # --- Setup Checkpoint Directory --- #
    # Create a unique run name including model name and timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{model_config['name']}_{timestamp}"
    checkpoint_dir = os.path.join(log_dir, run_name, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info(f"Run name: {run_name}")
    logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")

    # --- Data Loaders --- #
    logger.info("Setting up data loaders...")
    try:
        # Get batch size from training config, num_workers from data config
        batch_size = training_config['batch_size']
        num_workers = data_config.get('num_workers', 0)
        data_format = model_config.get('input_format', '2x512') # Get format from model_config or default

        # Get individual split ratios from config
        train_split = data_config.get('train_split', 0.7) # Default from dataset.py if not in data_config
        val_split = data_config.get('val_split', 0.15)
        test_split = data_config.get('test_split', 0.15)

        # Validate splits sum approximately to 1
        if not abs(train_split + val_split + test_split - 1.0) < 1e-6:
            logger.warning(f"Config split ratios ({train_split}, {val_split}, {test_split}) do not sum to 1.0. Using defaults 0.7, 0.15, 0.15.")
            val_split = 0.15
            test_split = 0.15

        train_loader, val_loader, _, dataset_stats = get_dataloaders(
            hdf5_path=data_config['path'],
            batch_size=batch_size,
            num_workers=num_workers,
            output_format=data_format, # Use determined data_format
            # Pass individual splits
            test_split=test_split,
            val_split=val_split,
            seed=data_config.get('seed', 42) # Pass seed as well
        )
        logger.info(f"Data loaders created. Train: {len(train_loader)} batches, Val: {len(val_loader)} batches.")
        logger.info(f"Dataset stats used for normalization: {dataset_stats}")
    except Exception as e:
        logger.error(f"Failed to create data loaders: {e}", exc_info=True)
        # sys.exit(1) # Don't exit from here, let main handle it if necessary
        raise # Re-raise the exception

    # --- Model Initialization --- #
    logger.info(f"Initializing model: {model_config['name']}")
    try:
        model = MTLModel(model_config, data_config, training_config)
        model.to(device)
        logger.info("Model initialized and moved to device.")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}", exc_info=True)
        raise

    # --- Optimizer --- #
    optimizer = setup_optimizer(model, config)

    # --- Loss Functions and Weights --- #
    class_criterion, reg_criterion, loss_weights = setup_loss_functions(config)

    # --- Training Loop --- #
    num_epochs = training_config['epochs']
    checkpoint_freq = training_config.get('checkpoint_freq', 10) # Save every 10 epochs by default
    best_val_loss = float('inf')

    logger.info(f"Starting training for {num_epochs} epochs...")
    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        logger.info(f"--- Epoch {epoch}/{num_epochs} ---")

        # Training Step
        train_losses = train_one_epoch(model, train_loader, optimizer, class_criterion, reg_criterion, loss_weights, device)
        logger.info(f"Epoch {epoch} Training Avg Losses: {train_losses}")

        # Log training losses to TensorBoard
        for key, loss_val in train_losses.items(): # train_losses should contain 'total' and individual task losses
            writer.add_scalar(f'Loss/train/{key}', loss_val, epoch)

        # Validation Step
        val_results = validate_one_epoch(model, val_loader, class_criterion, reg_criterion, loss_weights, device, dataset_stats) # Pass dataset_stats
        logger.info(f"Epoch {epoch} Validation Results: {val_results}")

        # Log validation losses and metrics to TensorBoard
        for key, value in val_results.items():
            if key == 'accuracy':
                writer.add_scalar('Accuracy/val', value, epoch)
            elif key.startswith('mae_denorm_'):
                 writer.add_scalar(f'MAE_denorm/val/{key.replace("mae_denorm_", "")}', value, epoch)
            else: # These are the normalized losses
                writer.add_scalar(f'Loss/val/{key}', value, epoch)
        
        epoch_duration = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch} completed in {epoch_duration:.2f} seconds.")

        # --- Checkpoint Saving --- #
        current_val_loss = val_results['total']

        # Save periodically
        if epoch % checkpoint_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_losses['total'],
                'val_loss': current_val_loss,
                'val_accuracy': val_results['accuracy']
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Save the best model based on validation loss
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'val_accuracy': val_results.get('accuracy', 0.0), # Use .get for safety
                'config': config # Save config with the best model
            }, best_checkpoint_path)
            logger.info(f"Saved new best model checkpoint (Val Loss: {best_val_loss:.4f}) to {best_checkpoint_path}")

    # --- End of Training --- #
    logger.info("Training finished.")
    writer.close() # Close the TensorBoard writer

    # Add config to TensorBoard
    writer.add_text('config', yaml.dump(config), 0) 