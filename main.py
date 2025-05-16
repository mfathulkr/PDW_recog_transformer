#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Main entry point for the PDW Recognition Transformer project.

This script handles command-line argument parsing to orchestrate
different modes of operation, such as training and evaluation.
It loads configurations and delegates tasks to the appropriate
modules within the src/ directory.
"""

import argparse
import logging
import os
import sys # Import sys for potential path manipulation if needed
import torch # Need torch for device setup
import yaml # Need yaml for exception handling during config load

# Add import for the config loader and logging setup
from src.utils.helpers import load_config, setup_logging
# Import the training function
from src.train import train_model
# Import the evaluation function and checkpoint loader
from src.evaluate import evaluate_model, load_checkpoint
# Import data loading function
from src.data_handling.dataset import get_dataloaders
# Import the main model class
from src.models import MTLModel

# --- Basic Logging Setup (using setup_logging) --- #
# Setup basic configuration first, will be enhanced if log file specified
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parses command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Multi-Task Learning for Radar Signal Characterisation")

    parser.add_argument("--mode", type=str, required=True, choices=['train', 'evaluate'],
                        help="Operating mode: train or evaluate")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the configuration file (e.g., config/base_config.yaml)")
    # Model choice is now mainly driven by config, but can be overridden (optional)
    parser.add_argument("--model", type=str, default=None, choices=['CNN1D', 'CNN2D', 'IQST-S', 'IQST-L', None],
                        help="Override the model architecture specified in the config file.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a model checkpoint file (required for evaluation, optional for resuming training)")
    # Data path override
    parser.add_argument("--data_path", type=str, default=None,
                        help="Override the path to the RadChar HDF5 dataset file specified in the config.")
    # Output directory override
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override the directory specified in the config to save logs, checkpoints, etc.")
    # Device override
    parser.add_argument("--device", type=str, default=None, choices=['cuda', 'cpu', None],
                        help="Override the device ('cuda' or 'cpu') specified in the config.")
    # Optional argument for log file
    parser.add_argument("--log_file", type=str, default=None,
                        help="Optional path to a file for saving logs.")
    # Add batch size override argument
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override the batch size specified in the config file for training and evaluation.")

    args = parser.parse_args()

    # --- Argument Validation --- #
    if args.mode == 'evaluate' and not args.checkpoint:
        parser.error("--checkpoint is required for evaluate mode.")
    # Check for config file existence before trying to load
    if not os.path.exists(args.config):
        parser.error(f"Configuration file not found at: {args.config}")
    # Data path validation will happen after config loading if args.data_path is None

    return args

def main():
    """Main execution function.

    Parses arguments, loads configuration, sets up logging,
    and calls the appropriate training or evaluation function.
    """
    args = parse_arguments()

    # --- Load Configuration --- #
    try:
        config = load_config(args.config)
        # logger.info(f"Configuration loaded successfully from {args.config}") # Logging setup happens next
    except (FileNotFoundError, yaml.YAMLError) as e:
        # Use basic logger if setup hasn't happened yet
        logging.error(f"Failed to load or parse configuration: {e}", exc_info=True)
        sys.exit(1) # Exit if config is invalid

    # --- Configuration & Argument Merging/Override --- #
    # Command-line arguments override config file settings if provided
    if args.model is not None:
        config['model']['name'] = args.model
    if args.data_path is not None:
        config['data']['path'] = args.data_path
    if args.device is not None:
        config['training']['device'] = args.device # Assume device is under 'training' section
        config['evaluation']['device'] = args.device # Also set for evaluation consistency
    if args.output_dir is not None:
        config['logging']['log_dir'] = args.output_dir
        config['training']['checkpoint_dir'] = os.path.join(args.output_dir, 'checkpoints') # Update checkpoint dir too
    # Override batch size if provided
    if args.batch_size is not None:
        if args.batch_size <= 0:
            logger.warning(f"Invalid batch size provided ({args.batch_size}). Using config value instead.")
        else:
            config['training']['batch_size'] = args.batch_size
            config['evaluation']['batch_size'] = args.batch_size
            logger.info(f"Overriding batch size with command-line value: {args.batch_size}")

    # --- Setup Enhanced Logging (using info from config/args) --- #
    log_dir = config.get('logging', {}).get('log_dir', 'runs')
    log_level_str = config.get('logging', {}).get('level', 'INFO').upper()
    log_file_path = args.log_file # Use CLI arg for log file path

    # Ensure base log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Construct log file path if specified via CLI or config (CLI takes precedence)
    if log_file_path:
        # Allow absolute path or relative to CWD
        if not os.path.isabs(log_file_path):
             log_file_path = os.path.join(os.getcwd(), log_file_path) # Or maybe relative to log_dir? Let's use CWD.
    else:
        # Default log file name within log_dir if not specified
        log_file_name = f"{config.get('model',{}).get('name', 'model')}_{args.mode}.log"
        log_file_path = os.path.join(log_dir, log_file_name)

    # Configure logging using the helper function
    setup_logging(log_level=log_level_str, log_file=log_file_path)

    # Now that logging is configured, log the startup info
    logger.info(f"Logging configured. Level: {log_level_str}, File: {log_file_path}")
    logger.info(f"Starting script in {args.mode} mode.")
    logger.info(f"Arguments parsed: {args}")
    logger.info(f"Initial configuration loaded from: {args.config}")
    logger.info(f"Final configuration being used: {config}") # Log the final config

    # --- Validate Essential Config Paths --- #
    if not os.path.exists(config['data']['path']):
         logger.error(f"Data file specified in config not found at: {config['data']['path']}")
         sys.exit(1)

    # --- Setup Device --- #
    device_str = config.get('training', {}).get('device', 'cpu') # Default to CPU if not found
    if device_str == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA device requested but not available. Falling back to CPU.")
        device_str = 'cpu'
    elif device_str == 'cuda':
         # Optionally log which GPU is being used
         gpu_index = torch.cuda.current_device()
         gpu_name = torch.cuda.get_device_name(gpu_index)
         logger.info(f"Using CUDA device {gpu_index}: {gpu_name}")
    else:
         logger.info("Using CPU device.")
    device = torch.device(device_str)

    # --- Mode Dispatch --- #
    if args.mode == 'train':
        logger.info("Initiating training process...")
        # Call the actual training function
        train_model(config, device) # Pass config and device
        # train_model(args, config) # Old way passing args
    elif args.mode == 'evaluate':
        logger.info("Initiating evaluation process...")

        # --- Get Data Loader for Test Set ---
        # Need the test loader and the normalization stats
        try:
            # Ensure get_dataloaders uses the correct path from the merged config
            # It needs the config to know batch size, workers, split ratios etc.
            # It returns train, val, test loaders AND stats dict
            _, _, test_loader, dataset_stats = get_dataloaders(
                hdf5_path=config['data']['path'], 
                batch_size=config.get('evaluation', {}).get('batch_size', config.get('training', {}).get('batch_size', 64)), # Use eval batch size or fallback
                num_workers=config.get('data', {}).get('num_workers', 0),
                output_format=config.get('data', {}).get('format', '2x512'), # Use output_format, key in config is 'format'
                # Pass individual splits from config
                test_split=config.get('data', {}).get('test_split', 0.15),
                val_split=config.get('data', {}).get('val_split', 0.15),
                seed=config.get('data', {}).get('seed', 42) # Pass seed as well
            )
            logger.info(f"Test data loader created successfully. Dataset stats: {dataset_stats}")
        except Exception as e:
             logger.error(f"Failed to create data loaders: {e}", exc_info=True)
             sys.exit(1)

        # --- Instantiate Model ---
        try:
            # Pass the relevant parts of the config to the model constructor
            model = MTLModel(
                model_config=config['model'],
                data_config=config['data'],
                training_config=config['training'] # Some model aspects might depend on training settings? Unlikely but pass for now.
            )
            logger.info(f"Model {config['model']['name']} instantiated successfully.")
        except Exception as e:
            logger.error(f"Failed to instantiate model: {e}", exc_info=True)
            sys.exit(1)

        # --- Load Checkpoint ---
        # Checkpoint path comes from args.checkpoint (validated earlier)
        try:
            model = load_checkpoint(model, args.checkpoint, device)
            # Model is moved to device inside load_checkpoint
            logger.info(f"Checkpoint loaded successfully from: {args.checkpoint}")
        except Exception as e:
            # Error already logged in load_checkpoint
            logger.error(f"Failed to load checkpoint. Exiting.")
            sys.exit(1)

        # --- Run Evaluation ---
        try:
            results = evaluate_model(
                config=config,
                model=model,
                test_loader=test_loader,
                device=device,
                dataset_stats=dataset_stats # Pass the stats for denormalization
            )
            logger.info(f"Evaluation complete. Results: {results}")

            # Optionally save results to a file in the output directory
            results_file = os.path.join(log_dir, f"evaluation_results_{config.get('model',{}).get('name', 'model')}.yaml")
            try:
                with open(results_file, 'w') as f:
                     yaml.dump(results, f, indent=4)
                logger.info(f"Evaluation results saved to: {results_file}")
            except Exception as e:
                logger.error(f"Failed to save evaluation results: {e}")

        except Exception as e:
             logger.error(f"An error occurred during evaluation: {e}", exc_info=True)
             sys.exit(1)
    else:
        # This case should be prevented by argparse choices, but added for safety
        logger.error(f"Invalid mode specified: {args.mode}")
        raise ValueError(f"Invalid mode: {args.mode}")

    logger.info(f"Script finished in {args.mode} mode.")


if __name__ == "__main__":
    main() 