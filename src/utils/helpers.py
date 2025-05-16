#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Contains utility functions for the PDW Recognition project.

This includes helper functions for:
- Metric calculations (e.g., de-normalized MAE)
- Normalization/De-normalization of labels
- Potentially custom weight initializations (if needed beyond framework defaults)
- Any other reusable code snippets.
"""

import numpy as np
import logging
import yaml
import os

logger = logging.getLogger(__name__) # Use logger from main script or configure here

def denormalize(value, min_val, max_val):
    """De-normalizes a value from [0, 1] range back to its original scale.

    Args:
        value (float or np.ndarray): The normalized value(s).
        min_val (float): The minimum value used for normalization.
        max_val (float): The maximum value used for normalization.

    Returns:
        float or np.ndarray: The value(s) in the original scale.
    """
    if max_val == min_val: # Avoid division by zero if max and min are the same
        # In this case, all original values were the same, so return that value
        return min_val
    return value * (max_val - min_val) + min_val

def load_config(config_path):
    """Loads configuration settings from a YAML file.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the configuration settings.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If the configuration file is invalid YAML.
    """
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found at: {config_path}")
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded configuration from: {config_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file: {config_path}", exc_info=True)
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading config: {config_path}", exc_info=True)
        raise e

def setup_logging(log_level=logging.INFO, log_file=None):
    """Configures logging for the application.

    Sets the basic configuration for the root logger.

    Args:
        log_level (int): The logging level (e.g., logging.INFO, logging.DEBUG).
        log_file (str, optional): Path to a file to save logs. If None, logs only go to console.
    """
    log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    handlers = [logging.StreamHandler()] # Log to console by default

    if log_file:
        # Ensure the directory for the log file exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a') # Append mode
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
        print(f"Logging to console and file: {log_file}") # Add print statement for confirmation
    else:
        print("Logging to console only.") # Add print statement for confirmation

    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)
    logger.info("Logging configured.")

# Add other utility functions as needed, e.g.:
# - calculate_mae(predictions, targets, scaler_info) # Handles de-normalization
# - setup_logging()
# - save_checkpoint(model, optimizer, epoch, path)
# - load_checkpoint(model, optimizer, path) 