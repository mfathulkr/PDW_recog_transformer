#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Handles the evaluation of a trained Multi-Task Learning model.
Loads a checkpoint, runs inference on the test set, calculates metrics
(accuracy, MAE), and logs the results. Includes denormalization for
interpretable regression errors and optional per-SNR reporting.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import logging
from tqdm import tqdm
import os
from collections import defaultdict

# Import necessary components from other project modules
from src.models import MTLModel # Assuming __init__.py makes it available
from src.data_handling.dataset import RadCharDataset # Needed for type hinting and potentially accessing stats
from src.utils.helpers import denormalize # Function to reverse normalization

logger = logging.getLogger(__name__)

# --- Helper Functions ---

def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device):
    """Loads model weights from a checkpoint file.

    Args:
        model (nn.Module): The model instance to load weights into.
        checkpoint_path (str): Path to the checkpoint file (.pth or .pt).
        device (torch.device): The device to load the checkpoint onto.

    Returns:
        nn.Module: The model with loaded weights.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        Exception: For other loading errors.
    """
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    try:
        # Load checkpoint onto the specified device
        # map_location ensures tensors are loaded onto the correct device,
        # especially if the checkpoint was saved on a different device type (e.g., GPU -> CPU).
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Check if the checkpoint contains 'model_state_dict'
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model state_dict from {checkpoint_path}")
            # Optionally, log other info like epoch or validation loss if available
            if 'epoch' in checkpoint:
                logger.info(f"Checkpoint was saved at epoch {checkpoint['epoch']}")
            if 'best_val_loss' in checkpoint:
                 logger.info(f"Checkpoint corresponds to validation loss: {checkpoint['best_val_loss']:.4f}")
        else:
            # Assume the checkpoint *is* the state_dict directly
            model.load_state_dict(checkpoint)
            logger.warning(f"Loaded state_dict directly from {checkpoint_path} (no extra info found).")

        model.to(device) # Ensure model is on the correct device
        return model

    except Exception as e:
        logger.error(f"Error loading checkpoint from {checkpoint_path}: {e}", exc_info=True)
        raise e

def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculates classification accuracy.

    Args:
        predictions (torch.Tensor): Raw logits or probabilities from the model (Batch, Num_Classes).
        targets (torch.Tensor): Ground truth class indices (Batch,).

    Returns:
        float: Accuracy value (0.0 to 1.0).
    """
    # Get the index of the max log-probability/probability
    predicted_classes = torch.argmax(predictions, dim=1)
    correct = (predicted_classes == targets).sum().item()
    total = targets.size(0)
    accuracy = correct / total if total > 0 else 0.0
    return accuracy

def calculate_mae(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculates Mean Absolute Error (MAE).

    Args:
        predictions (torch.Tensor): Model's regression predictions (Batch,).
        targets (torch.Tensor): Ground truth regression values (Batch,).

    Returns:
        float: MAE value.
    """
    # Ensure tensors are flat (Batch,)
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    mae = torch.abs(predictions - targets).mean().item()
    return mae

# --- Main Evaluation Function ---

def evaluate_model(config: dict, model: nn.Module, test_loader: DataLoader, device: torch.device, dataset_stats: dict):
    """Evaluates the trained MTL model on the test set.

    Args:
        config (dict): The configuration dictionary (for evaluation settings).
        model (nn.Module): The trained model instance (weights should be loaded).
        test_loader (DataLoader): DataLoader for the test set.
        device (torch.device): The device to run evaluation on (e.g., 'cuda' or 'cpu').
        dataset_stats (dict): Dictionary containing normalization statistics ('mean', 'std')
                              for each regression task, used for denormalization.
                              Keys should match the regression task names (e.g., 'np', 'pw', 'pri', 'td').
    """
    logger.info("Starting evaluation on the test set...")
    model.eval() # Set the model to evaluation mode (disables dropout, batchnorm uses running stats)

    # Store all predictions and targets to calculate overall metrics at the end
    all_preds = defaultdict(list)
    all_targets = defaultdict(list)
    # Optional: Store per SNR if available
    snr_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list))) # snr_results[snr_key][preds/targets][task_key]

    # Check if dataset provides SNR information
    has_snr_info = False
    try:
        # Peek at the first batch to check for 'snr' key in targets
        logger.info("Checking first batch for SNR information...")
        first_batch_data, first_batch_targets = next(iter(test_loader))
        if isinstance(first_batch_targets, dict) and 'snr' in first_batch_targets:
            has_snr_info = True
            logger.info("SNR information detected in dataset targets. Will report per-SNR metrics.")
        else:
             logger.info("SNR key not found in first batch targets. Proceeding without per-SNR analysis.")
        # Restore iterator state if possible or just reload data (simpler for now)
        # This check modifies the iterator, so the first batch will be skipped
        # unless the dataloader is recreated or reset. For simplicity in evaluation,
        # this check might be acceptable, or dataloader could be recreated after check.
        # Alternatively, check dataset object attribute if available.
    except StopIteration:
        logger.warning("Test loader is empty. Cannot check for SNR or evaluate.")
        return {} # Return empty results if no data
    except Exception as e:
         logger.warning(f"Could not determine if SNR info is available from first batch: {e}. Proceeding without per-SNR analysis.")
         has_snr_info = False # Assume no SNR info if check fails


    with torch.no_grad(): # Disable gradient calculations for efficiency
        for batch_idx, (data, targets) in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Move data and targets to the configured device
            # Handle both input formats (2x512 or 1x32x32) based on model type in config
            model_name = config.get('model', {}).get('name', 'Unknown')
            if model_name == 'CNN2D':
                # CNN2D expects (B, 1, 32, 32)
                # Assume dataset provides this format if model is CNN2D
                 if data.shape[1:] != (1, 32, 32):
                      # Attempt reshape if necessary, log warning
                      if data.shape[1:] == (2, 512):
                          logger.warning(f"Batch {batch_idx}: Reshaping input from {data.shape} to (-1, 1, 32, 32) for CNN2D evaluation.")
                          batch_size = data.shape[0]
                          data = data.view(batch_size, -1).view(batch_size, 1, 32, 32)
                      else:
                          logger.error(f"Batch {batch_idx}: Unexpected input shape {data.shape} for CNN2D. Cannot evaluate.")
                          continue # Skip batch if shape is wrong
            elif data.shape[1:] != (2, 512):
                 logger.error(f"Batch {batch_idx}: Unexpected input shape {data.shape} for {model_name}. Expected (B, 2, 512). Cannot evaluate.")
                 continue # Skip batch

            data = data.to(device)

            # Move all target tensors to the device
            targets_device = {}
            for key, value in targets.items():
                # We expect 'class', 'np', 'pw', 'pri', 'td', potentially 'snr'
                if isinstance(value, torch.Tensor):
                    targets_device[key] = value.to(device)
                # else: # Keep non-tensor info like SNR as is for now
                #    targets_device[key] = value

            # --- Forward Pass ---
            predictions = model(data) # Get dictionary of predictions

            # --- Collect Predictions and Targets ---
            for task_key in predictions.keys(): # 'class', 'np', 'pw', 'pri', 'td'
                if task_key in targets_device:
                    # Store predictions (move to CPU to avoid GPU memory buildup)
                    all_preds[task_key].append(predictions[task_key].detach().cpu())
                    # Store corresponding targets
                    all_targets[task_key].append(targets_device[task_key].detach().cpu())
                else:
                    logger.warning(f"Prediction key '{task_key}' not found in targets for batch {batch_idx}.")

            # --- Collect per SNR --- #
            if has_snr_info and 'snr' in targets: # Check if 'snr' is actually in the current batch targets
                snr_values = targets['snr'] # This should be a list or tensor of SNRs for the batch
                # Ensure snr_values is iterable and matches batch size
                if isinstance(snr_values, torch.Tensor):
                     snr_values = snr_values.tolist() # Convert tensor to list for easier iteration
                elif not isinstance(snr_values, (list, np.ndarray)):
                     logger.warning(f"Unexpected type for SNR values in batch {batch_idx}: {type(snr_values)}. Skipping per-SNR collection for this batch.")
                     continue # Skip per-SNR for this batch
                
                if len(snr_values) != data.size(0):
                    logger.warning(f"Mismatch between batch size ({data.size(0)}) and number of SNR values ({len(snr_values)}) in batch {batch_idx}. Skipping per-SNR collection.")
                    continue # Skip per-SNR for this batch

                for i in range(data.size(0)): # Iterate through items in the batch
                    snr_val = snr_values[i] # Get SNR for this specific sample (should be an int)
                    snr_key = f"SNR_{snr_val}" # Create a dictionary key like SNR_10, SNR_-2
                    for task_key in predictions.keys():
                        if task_key in targets_device:
                            # Store individual prediction and target tensor slices indexed by SNR
                            # Move to CPU immediately to save GPU memory
                            snr_results[snr_key]['preds'][task_key].append(predictions[task_key][i].detach().cpu())
                            snr_results[snr_key]['targets'][task_key].append(targets_device[task_key][i].detach().cpu())
            elif has_snr_info:
                 logger.warning(f"SNR key 'snr' expected but not found in targets for batch {batch_idx}.")


    # --- Consolidate Results ---
    logger.info("Evaluation loop finished. Consolidating results...")
    final_preds = {}
    final_targets = {}
    for task_key in all_preds.keys():
        if all_preds[task_key]: # Check if list is not empty
             final_preds[task_key] = torch.cat(all_preds[task_key], dim=0)
             final_targets[task_key] = torch.cat(all_targets[task_key], dim=0)
        else:
            logger.warning(f"No predictions collected for task: {task_key}. Skipping metric calculation.")


    # --- Calculate Overall Metrics ---
    results = {}
    logger.info("--- Overall Test Set Metrics ---")

    # Classification Accuracy
    if 'class' in final_preds and 'class' in final_targets:
        class_accuracy = calculate_accuracy(final_preds['class'], final_targets['class'])
        results['class_accuracy'] = class_accuracy
        logger.info(f"Classification Accuracy: {class_accuracy:.4f}")
    else:
        logger.warning("Classification results not available.")

    # Regression MAE (De-normalized)
    regression_tasks = ['np', 'pw', 'pri', 'td']
    for task in regression_tasks:
        if task in final_preds and task in final_targets:
            # Get predictions and targets
            preds = final_preds[task]
            targs = final_targets[task]

            # De-normalize using provided statistics
            if 'label_min_max' in dataset_stats and task in dataset_stats['label_min_max']:
                min_val = dataset_stats['label_min_max'][task]['min']
                max_val = dataset_stats['label_min_max'][task]['max']

                preds_denorm = denormalize(preds, min_val, max_val)
                targs_denorm = denormalize(targs, min_val, max_val)

                mae_denorm = calculate_mae(preds_denorm, targs_denorm)
                results[f'{task}_mae_denormalized'] = mae_denorm
                logger.info(f"Regression MAE ({task}, De-normalized): {mae_denorm:.4f}")

                 # Also calculate MAE on normalized values for reference
                mae_norm = calculate_mae(preds, targs)
                results[f'{task}_mae_normalized'] = mae_norm
                logger.debug(f"Regression MAE ({task}, Normalized): {mae_norm:.4f}") # Use debug level

            else:
                logger.warning(f"Normalization stats for task '{task}' not found in dataset_stats. Cannot calculate de-normalized MAE.")
                # Calculate MAE on (presumably) normalized values
                mae_norm = calculate_mae(preds, targs)
                results[f'{task}_mae_normalized'] = mae_norm
                logger.info(f"Regression MAE ({task}, Normalized): {mae_norm:.4f}")
        else:
             logger.warning(f"Results for regression task '{task}' not available.")

    # --- Calculate Per-SNR Metrics (Optional) ---
    if has_snr_info and snr_results:
        logger.info("--- Per-SNR Test Set Metrics ---")
        snr_metric_results = defaultdict(dict)
        # Sort SNR keys numerically based on the integer value after 'SNR_'
        sorted_snrs = sorted(snr_results.keys(), key=lambda x: int(x.split('_')[1]))

        for snr_key in sorted_snrs:
            snr = snr_key.split('_')[1]
            logger.info(f"--- SNR: {snr} dB ---")
            snr_preds = defaultdict(list)
            snr_targets = defaultdict(list)
            valid_snr_data = True

            # Consolidate predictions and targets for this specific SNR
            # Each item in snr_results[snr_key]['preds'][task_key] is a tensor for ONE sample
            # We need to stack them into a batch-like tensor for metric calculation
            for task_key in snr_results[snr_key]['preds']:
                 if snr_results[snr_key]['preds'][task_key]: # Check if list is not empty
                     try:
                         snr_preds[task_key] = torch.stack(snr_results[snr_key]['preds'][task_key], dim=0)
                         snr_targets[task_key] = torch.stack(snr_results[snr_key]['targets'][task_key], dim=0)
                     except Exception as stack_err:
                          logger.error(f"Error stacking tensors for task {task_key} at SNR {snr}: {stack_err}")
                          valid_snr_data = False
                          break # Stop processing this SNR if stacking fails
                 else:
                     logger.warning(f"No predictions collected for task {task_key} at SNR {snr}.")
                     valid_snr_data = False # Mark as invalid if any task is missing data

            if not valid_snr_data:
                 logger.warning(f"Skipping metric calculation for SNR {snr} due to data issues.")
                 continue # Skip to the next SNR level

            # Calculate metrics for this SNR
            # Classification Accuracy
            if 'class' in snr_preds and 'class' in snr_targets:
                acc_snr = calculate_accuracy(snr_preds['class'], snr_targets['class'])
                snr_metric_results[snr_key]['class_accuracy'] = acc_snr
                logger.info(f"  Classification Accuracy: {acc_snr:.4f}")
            else:
                logger.warning(f"Classification results not available for SNR {snr}.")

            # Regression MAE (De-normalized)
            for task in regression_tasks:
                 if task in snr_preds and task in snr_targets:
                    preds_snr = snr_preds[task]
                    targs_snr = snr_targets[task]
                    # Need dataset_stats for denormalization
                    if 'label_min_max' in dataset_stats and task in dataset_stats['label_min_max']:
                         min_val = dataset_stats['label_min_max'][task]['min']
                         max_val = dataset_stats['label_min_max'][task]['max']

                         preds_denorm_snr = denormalize(preds_snr, min_val, max_val)
                         targs_denorm_snr = denormalize(targs_snr, min_val, max_val)
                         mae_denorm_snr = calculate_mae(preds_denorm_snr, targs_denorm_snr)
                         snr_metric_results[snr_key][f'{task}_mae_denormalized'] = mae_denorm_snr
                         logger.info(f"  Regression MAE ({task}, De-normalized): {mae_denorm_snr:.4f}")
                    else:
                        logger.warning(f"Normalization stats for task '{task}' not found for SNR {snr}. Cannot calculate de-normalized MAE.")
                        # Calculate MAE on normalized values as a fallback
                        mae_norm_snr = calculate_mae(preds_snr, targs_snr)
                        snr_metric_results[snr_key][f'{task}_mae_normalized'] = mae_norm_snr
                        logger.info(f"  Regression MAE ({task}, Normalized): {mae_norm_snr:.4f}")
                 else:
                     logger.warning(f"Results for regression task '{task}' not available for SNR {snr}.")

        # Add per-SNR results to the main results dict
        results['per_snr_metrics'] = snr_metric_results
    elif has_snr_info:
        logger.warning("SNR info was detected, but no per-SNR results were collected during evaluation loop. Check data loading and collection logic.")

    logger.info("Evaluation finished.")
    return results # Return the calculated metrics 