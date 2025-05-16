#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Handles dataset loading, parsing, preprocessing, and splitting for RadChar.

Implements the PyTorch Dataset class for RadChar, handling:
- Reading data from HDF5 files.
- Extracting IQ samples and labels.
- Applying necessary preprocessing steps:
    - IQ data standardization (based on training set statistics).
    - Regression label normalization to [0, 1] (based on training set statistics).
- Providing data in the format required by the models (e.g., 2x512 or 1x32x32).

Also includes functions for splitting the dataset indices into train, validation,
and test sets according to the paper's 70-15-15 ratio.
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import h5py
import numpy as np
import logging
from sklearn.model_selection import train_test_split
import time # Import time for timing the stats calculation

logger = logging.getLogger(__name__)

# --- Constants for Label Keys --- #
# These should match the dtype names in the HDF5 file (see docs/radcharRM.md)
LABEL_KEYS = {
    'index': 'index',
    'signal_type': 'signal_type',
    'np': 'number_of_pulses',
    'pw': 'pulse_width',
    'td': 'time_delay',
    'pri': 'pulse_repetition_interval',
    'snr': 'signal_to_noise_ratio'
}
REGRESSION_KEYS = ['np', 'pw', 'td', 'pri']
CLASSIFICATION_KEY = 'signal_type'
SNR_KEY = 'snr'

# Mapping from integer in dataset to string name (optional, for clarity)
SIGNAL_TYPE_MAP = {
    0: 'coherent_pulse_train',
    1: 'barker_code',
    2: 'polyphase_barker_code',
    3: 'frank_code',
    4: 'linear_frequency_modulated'
}
NUM_CLASSES = len(SIGNAL_TYPE_MAP)

class RadCharDataset(Dataset):
    """PyTorch Dataset class for the RadChar HDF5 dataset.

    Handles data loading, preprocessing (standardization/normalization),
    and provides samples for the DataLoader.
    """
    def __init__(self, hdf5_path, indices, stats=None, output_format='2x512'):
        """Initializes the dataset.

        Args:
            hdf5_path (str): Path to the HDF5 dataset file.
            indices (list or np.ndarray): List of indices to include in this dataset subset (train/val/test).
            stats (dict, optional): Dictionary containing mean/std for IQ data and min/max
                                    for regression labels, calculated from the training set.
                                    Required for val/test sets, calculated if None (for train set).
            output_format (str): The desired output format for IQ data ('2x512' or '1x32x32').
        """
        self.hdf5_path = hdf5_path
        self.indices = indices
        self.stats = stats
        self.output_format = output_format
        self.num_samples = len(indices)

        # Preprocessing parameters (mean, std, min, max)
        self.iq_mean = None
        self.iq_std = None
        self.label_min_max = {} # To store min/max for each regression label

        # Load data pointers (don't load all data into memory)
        # Using 'with' ensures the file is closed properly if an error occurs
        try:
            self._h5_file = h5py.File(self.hdf5_path, 'r')
            self._h5_iqs = self._h5_file['iq']
            self._h5_labels = self._h5_file['labels']
        except Exception as e:
            logger.error(f"Failed to open or read HDF5 file: {self.hdf5_path}", exc_info=True)
            # Ensure file handle is closed if partially opened
            if hasattr(self, '_h5_file') and self._h5_file:
                self._h5_file.close()
            raise e # Re-raise the exception

        # Calculate or apply stats
        if self.stats is None:
            logger.info("Calculating statistics from the provided indices (assuming this is the training set)...")
            self._calculate_stats()
            logger.info("Statistics calculation complete.")
        else:
            logger.info("Applying provided statistics (assuming this is a validation or test set)...")
            self.iq_mean = self.stats['iq_mean']
            self.iq_std = self.stats['iq_std']
            self.label_min_max = self.stats['label_min_max']
            if self.iq_mean is None or self.iq_std is None or not self.label_min_max:
                logger.error("Provided stats dictionary is incomplete.")
                self._h5_file.close()
                raise ValueError("Provided stats dictionary is incomplete.")
            logger.info("Statistics applied.")

        # Check output format validity
        if self.output_format not in ['2x512', '1x32x32']:
            logger.error(f"Invalid output_format: {self.output_format}")
            self._h5_file.close()
            raise ValueError(f"Invalid output_format specified: {self.output_format}")


    def _calculate_stats(self):
        """Calculates mean/std for IQ data and min/max for regression labels.

        Iterates through the specified indices of the HDF5 dataset IN CHUNKS
        to compute statistics accurately without loading everything into memory.
        This should ONLY be called on the training dataset.
        """
        logger.info(f"Starting accurate statistics calculation for {self.num_samples} training samples...")
        start_time = time.time()

        # --- 1. Calculate Label Min/Max --- #
        # This is less memory intensive and can be done first.
        min_labels = {key: float('inf') for key in REGRESSION_KEYS}
        max_labels = {key: float('-inf') for key in REGRESSION_KEYS}

        # Sort indices before reading labels for fancy indexing
        sorted_indices_for_labels = np.sort(self.indices)

        # Load necessary label columns for the training indices directly
        # This might still be large, but usually feasible compared to IQ data
        try:
            labels_subset = self._h5_labels[list(sorted_indices_for_labels)] # Use sorted indices
            for key in REGRESSION_KEYS:
                label_data = labels_subset[LABEL_KEYS[key]]
                min_labels[key] = np.min(label_data)
                max_labels[key] = np.max(label_data)
            self.label_min_max = {key: {'min': min_labels[key], 'max': max_labels[key]} for key in REGRESSION_KEYS}
            del labels_subset # Free memory
            logger.info("Label min/max calculation complete.")
        except Exception as e:
             logger.error("Error calculating label min/max.", exc_info=True)
             self._h5_file.close()
             raise e

        # --- 2. Calculate IQ Mean/Std using Chunking --- #
        # Determine a reasonable chunk size based on available memory
        # Example: Process 10,000 samples at a time
        chunk_size = 10000
        num_chunks = int(np.ceil(self.num_samples / chunk_size))

        sum_i = np.float64(0.0)
        sum_q = np.float64(0.0)
        sum_sq_i = np.float64(0.0)
        sum_sq_q = np.float64(0.0)
        total_elements = 0

        logger.info(f"Calculating IQ mean/std over {num_chunks} chunks (chunk size: {chunk_size})...")

        # Sort indices for potentially faster sequential access in HDF5
        sorted_indices = np.sort(self.indices)

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, self.num_samples)
            chunk_indices = sorted_indices[start_idx:end_idx]

            if len(chunk_indices) == 0:
                continue

            try:
                # Load chunk of complex IQ data
                iq_chunk_complex = self._h5_iqs[list(chunk_indices)]

                # Separate I and Q, cast to float32 for processing
                iq_chunk_real = np.real(iq_chunk_complex).astype(np.float32)
                iq_chunk_imag = np.imag(iq_chunk_complex).astype(np.float32)
                del iq_chunk_complex # Free memory

                # Update sums for mean calculation (convert to float64 for precision)
                sum_i += np.sum(iq_chunk_real, dtype=np.float64)
                sum_q += np.sum(iq_chunk_imag, dtype=np.float64)

                # Update sums of squares for std calculation
                sum_sq_i += np.sum(np.square(iq_chunk_real, dtype=np.float64))
                sum_sq_q += np.sum(np.square(iq_chunk_imag, dtype=np.float64))

                # Keep track of total number of elements processed (N * signal_length)
                total_elements += iq_chunk_real.size # Size includes all elements in the chunk

                if (i + 1) % max(1, num_chunks // 10) == 0: # Log progress periodically
                     logger.info(f"  Processed chunk {i+1}/{num_chunks}")

            except Exception as e:
                logger.error(f"Error processing chunk {i+1} for IQ stats.", exc_info=True)
                self._h5_file.close()
                raise e

        if total_elements == 0:
            logger.error("No elements processed for IQ statistics calculation.")
            self._h5_file.close()
            raise ValueError("Cannot calculate statistics with zero elements.")

        # Calculate mean
        mean_i = sum_i / total_elements
        mean_q = sum_q / total_elements

        # Calculate standard deviation (using E[X^2] - (E[X])^2)
        var_i = (sum_sq_i / total_elements) - np.square(mean_i)
        var_q = (sum_sq_q / total_elements) - np.square(mean_q)

        # Clamp variance to avoid issues with floating point precision near zero
        std_i = np.sqrt(max(var_i, 1e-12)) # Use max(var, eps) instead of adding eps inside sqrt
        std_q = np.sqrt(max(var_q, 1e-12))

        self.iq_mean = np.array([mean_i, mean_q], dtype=np.float32)
        self.iq_std = np.array([std_i, std_q], dtype=np.float32)

        # --- Store Calculated Stats --- #
        self.stats = {
            'iq_mean': self.iq_mean,
            'iq_std': self.iq_std,
            'label_min_max': self.label_min_max
        }

        end_time = time.time()
        logger.info(f"Accurate statistics calculation finished in {end_time - start_time:.2f} seconds.")
        logger.info(f"  IQ Mean (I, Q): {self.iq_mean}")
        logger.info(f"  IQ Std Dev (I, Q): {self.iq_std}")
        logger.info(f"  Label Min/Max: {self.label_min_max}")

    def __len__(self):
        """Returns the number of samples in the dataset subset."""
        return self.num_samples

    def __getitem__(self, idx):
        """Retrieves and preprocesses a single sample from the dataset.

        Args:
            idx (int): The index within the self.indices list.

        Returns:
            tuple: (iq_data, labels_dict)
                   - iq_data (torch.Tensor): Preprocessed IQ data in the specified output_format.
                   - labels_dict (dict): Dictionary containing the classification label (int)
                                         and normalized regression labels (float tensors).
                                         Also includes the original SNR.
                                         Keys: 'class', 'np', 'pw', 'pri', 'td', 'snr'
        """
        # Get the actual HDF5 index from our subset indices
        h5_index = self.indices[idx]

        try:
            # --- Load Raw Data --- #
            # Load complex IQ data for the specific index
            iq_complex = self._h5_iqs[h5_index]
            # Load the structured label for the specific index
            label_struct = self._h5_labels[h5_index]

            # --- Preprocess IQ Data --- #
            # Separate I and Q channels
            iq_real = np.real(iq_complex).astype(np.float32)
            iq_imag = np.imag(iq_complex).astype(np.float32)

            # Stack them into (2, 512)
            iq_stacked = np.stack((iq_real, iq_imag), axis=0) # Shape: (2, 512)

            # Standardize IQ data (using pre-calculated or provided stats)
            # Ensure stats are available
            if self.iq_mean is None or self.iq_std is None:
                logger.error("IQ normalization stats are missing. Cannot preprocess data.")
                raise RuntimeError("IQ normalization statistics are not available.")
            # Apply standardization: (x - mean) / std
            # Need to reshape mean/std for broadcasting: (2,) -> (2, 1)
            iq_standardized = (iq_stacked - self.iq_mean[:, np.newaxis]) / self.iq_std[:, np.newaxis]

            # --- Format IQ Data --- #
            if self.output_format == '1x32x32':
                # Reshape (2, 512) -> (1024,) -> (1, 32, 32)
                if iq_standardized.size != 1024:
                     logger.warning(f"Sample at h5_index {h5_index} has unexpected size {iq_standardized.size} before reshaping to 1x32x32.")
                     # Handle potential error or return None/skip sample?
                     # For now, proceed but log warning. Shape errors might occur later.
                # Flatten first, then reshape. Ensure C-contiguous order.
                iq_data = torch.from_numpy(iq_standardized.flatten()).view(1, 32, 32)
            else: # Default to '2x512'
                iq_data = torch.from_numpy(iq_standardized) # Shape (2, 512)

            # --- Extract and Preprocess Labels --- #
            targets = {}

            # Classification label (convert to LongTensor)
            class_label_key = LABEL_KEYS[CLASSIFICATION_KEY]
            targets['class'] = torch.tensor(label_struct[class_label_key], dtype=torch.long)

            # Regression labels (normalize to [0, 1])
            for key in REGRESSION_KEYS:
                label_key = LABEL_KEYS[key]
                raw_value = label_struct[label_key]
                # Get min/max stats for this label
                stats = self.label_min_max.get(key)
                if stats is None:
                    logger.error(f"Normalization stats missing for label '{key}'. Cannot preprocess.")
                    raise RuntimeError(f"Normalization statistics not available for label '{key}'.")
                min_val = stats['min']
                max_val = stats['max']
                # Normalize: (value - min) / (max - min)
                range_val = max_val - min_val
                if range_val == 0: # Avoid division by zero if min == max
                    normalized_value = 0.5 # Assign a neutral value or handle as error?
                    logger.warning(f"Label '{key}' has zero range (min=max={min_val}). Setting normalized value to 0.5.")
                else:
                    normalized_value = (raw_value - min_val) / range_val
                # Clamp values to [0, 1] just in case of floating point issues
                normalized_value = np.clip(normalized_value, 0.0, 1.0)
                targets[key] = torch.tensor(normalized_value, dtype=torch.float32)

            # Extract SNR (store as integer, no normalization needed)
            snr_label_key = LABEL_KEYS[SNR_KEY]
            targets['snr'] = int(label_struct[snr_label_key]) # Store as Python int
            # Alternatively, could store as torch.tensor(..., dtype=torch.long) if preferred
            # targets['snr'] = torch.tensor(label_struct[snr_label_key], dtype=torch.long)

            return iq_data, targets

        except Exception as e:
            # Log the HDF5 index and the dataset index that caused the error
            logger.error(f"Error processing sample at dataset index {idx} (HDF5 index {h5_index}): {e}", exc_info=True)
            # Depending on dataloader behavior, returning None might be skipped, or cause issues.
            # It might be safer to raise the exception or return dummy data.
            # Let's re-raise for now to make errors explicit.
            raise e

    def get_stats(self):
        """Returns the calculated or stored statistics."""
        return self.stats

    def close(self):
        """Closes the HDF5 file handle."""
        if hasattr(self, '_h5_file') and self._h5_file:
            try:
                self._h5_file.close()
                logger.info(f"Closed HDF5 file: {self.hdf5_path}")
            except Exception as e:
                 logger.error(f"Error closing HDF5 file: {self.hdf5_path}", exc_info=True)

    def __del__(self):
        """Ensures the HDF5 file is closed when the object is garbage collected."""
        self.close()

def get_dataloaders(hdf5_path, batch_size, test_split=0.15, val_split=0.15, output_format='2x512', num_workers=0, seed=42):
    """Creates train, validation, and test DataLoaders for the RadChar dataset.

    Args:
        hdf5_path (str): Path to the HDF5 dataset file.
        batch_size (int): Batch size for the DataLoaders.
        test_split (float): Proportion of the dataset to use for the test set (e.g., 0.15 for 15%).
        val_split (float): Proportion of the dataset to use for the validation set (e.g., 0.15 for 15%).
        output_format (str): The desired IQ data format ('2x512' or '1x32x32').
        num_workers (int): Number of subprocesses to use for data loading.
        seed (int): Random seed for splitting data reproducibly.

    Returns:
        tuple: (train_loader, val_loader, test_loader, train_stats)
               - train_loader (DataLoader): DataLoader for the training set.
               - val_loader (DataLoader): DataLoader for the validation set.
               - test_loader (DataLoader): DataLoader for the test set.
               - train_stats (dict): Statistics calculated from the training set.
    """
    logger.info(f"Creating dataloaders from: {hdf5_path}")
    logger.info(f"Splits: Train={1.0-test_split-val_split:.2f}, Val={val_split:.2f}, Test={test_split:.2f}")
    logger.info(f"Output format: {output_format}")

    # Get total number of samples and indices
    try:
        with h5py.File(hdf5_path, 'r') as f:
            total_samples = len(f['iq'])
            all_indices = list(range(total_samples))
            # Load all labels once for stratified split if desired (memory intensive for very large datasets)
            # For now, split indices directly
            # all_labels_signal_type = f['labels'][LABEL_KEYS[CLASSIFICATION_KEY]][:]
    except Exception as e:
        logger.error(f"Failed to get sample count from HDF5 file: {hdf5_path}", exc_info=True)
        raise e

    # Split indices: First into train and temp (val+test)
    train_indices, temp_indices = train_test_split(
        all_indices,
        test_size=(test_split + val_split), # Combine val and test sizes for the first split
        random_state=seed,
        # stratify=all_labels_signal_type # Optional: Stratify by signal type
    )

    # Split temp indices into val and test
    # Adjust split ratio relative to the size of the temp set
    relative_test_split = test_split / (test_split + val_split)
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=relative_test_split,
        random_state=seed,
        # stratify=all_labels_signal_type[temp_indices] if stratify else None # Optional: Stratify
    )

    logger.info(f"Dataset sizes: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

    # Create Datasets
    # Train dataset (calculates stats)
    train_dataset = RadCharDataset(hdf5_path, train_indices, stats=None, output_format=output_format)
    train_stats = train_dataset.get_stats()

    # Validation dataset (applies train stats)
    val_dataset = RadCharDataset(hdf5_path, val_indices, stats=train_stats, output_format=output_format)

    # Test dataset (applies train stats)
    test_dataset = RadCharDataset(hdf5_path, test_indices, stats=train_stats, output_format=output_format)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    logger.info("DataLoaders created successfully.")

    # Note: RadCharDataset instances hold open HDF5 file handles.
    # Consider closing them explicitly after training/evaluation if memory is a concern,
    # although __del__ provides some safety.

    return train_loader, val_loader, test_loader, train_stats

# --- Example Usage (for testing this module directly) --- #
if __name__ == '__main__':
    # Configure logging for direct script execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    # --- Configuration --- #
    # Create a dummy HDF5 file for testing if one doesn't exist
    dummy_h5_path = 'data/dummy_radchar.h5'
    num_dummy_samples = 1000
    signal_length = 512

    if not os.path.exists(dummy_h5_path):
        logger.info(f"Creating dummy HDF5 file at: {dummy_h5_path}")
        os.makedirs('data', exist_ok=True)
        with h5py.File(dummy_h5_path, 'w') as f:
            # Dummy IQ data (complex numbers)
            dummy_iq = np.random.randn(num_dummy_samples, signal_length) + 1j * np.random.randn(num_dummy_samples, signal_length)
            f.create_dataset('iq', data=dummy_iq.astype(np.complex64))

            # Dummy labels (structured array)
            label_dtype = [('index', '<i8'), ('signal_type', '<i8'), ('number_of_pulses', '<i8'),
                           ('pulse_width', '<f8'), ('time_delay', '<f8'), ('pulse_repetition_interval', '<f8'),
                           ('signal_to_noise_ratio', '<i8')]
            dummy_labels = np.zeros(num_dummy_samples, dtype=label_dtype)
            dummy_labels['index'] = np.arange(num_dummy_samples)
            dummy_labels['signal_type'] = np.random.randint(0, NUM_CLASSES, num_dummy_samples)
            dummy_labels['number_of_pulses'] = np.random.randint(2, 7, num_dummy_samples)
            dummy_labels['pulse_width'] = np.random.uniform(10e-6, 16e-6, num_dummy_samples)
            dummy_labels['time_delay'] = np.random.uniform(1e-6, 10e-6, num_dummy_samples)
            dummy_labels['pulse_repetition_interval'] = np.random.uniform(17e-6, 23e-6, num_dummy_samples)
            dummy_labels['signal_to_noise_ratio'] = np.random.randint(-20, 21, num_dummy_samples)
            f.create_dataset('labels', data=dummy_labels)
        logger.info("Dummy HDF5 file created.")
    else:
        logger.info(f"Using existing dummy HDF5 file: {dummy_h5_path}")

    # --- Test DataLoader Creation --- #
    try:
        train_loader, val_loader, test_loader, train_stats = get_dataloaders(
            hdf5_path=dummy_h5_path,
            batch_size=32,
            output_format='2x512' # Test with '2x512'
        )

        logger.info(f"Train stats: {train_stats}")

        # --- Test Iteration --- #
        logger.info("Iterating through one batch of train_loader...")
        for i, (iq_batch, label_batch) in enumerate(train_loader):
            logger.info(f"Batch {i+1}:")
            logger.info(f"  IQ data shape: {iq_batch.shape}, dtype: {iq_batch.dtype}")
            logger.info(f"  Labels type: {type(label_batch)}")
            logger.info(f"  Label keys: {label_batch.keys()}")
            logger.info(f"  Class label shape: {label_batch['class'].shape}")
            logger.info(f"  'np' label shape: {label_batch['np'].shape}, dtype: {label_batch['np'].dtype}")
            logger.info(f"  'snr' label shape: {label_batch['snr'].shape}")
            logger.info(f"  First 'np' label (normalized): {label_batch['np'][0].item()}")
            logger.info(f"  First class label: {label_batch['class'][0].item()}")
            logger.info(f"  First SNR label: {label_batch['snr'][0].item()}")

            # --- Test De-normalization --- #
            np_min = train_stats['label_min_max']['np']['min']
            np_max = train_stats['label_min_max']['np']['max']
            from src.utils.helpers import denormalize # Assuming helpers.py is accessible
            denorm_np = denormalize(label_batch['np'][0].item(), np_min, np_max)
            logger.info(f"  First 'np' label (de-normalized): {denorm_np}")

            if i >= 0: # Only process one batch for testing
                break

        logger.info("Testing RadCharDataset with '1x32x32' format...")
        train_loader_32, _, _, _ = get_dataloaders(
            hdf5_path=dummy_h5_path,
            batch_size=32,
            output_format='1x32x32' # Test with '1x32x32'
        )
        iq_batch_32, _ = next(iter(train_loader_32))
        logger.info(f"IQ data shape for '1x32x32' format: {iq_batch_32.shape}")

        # Explicitly close datasets (good practice)
        # Get dataset objects from dataloaders
        train_loader.dataset.close()
        val_loader.dataset.close()
        test_loader.dataset.close()
        train_loader_32.dataset.close()

    except Exception as e:
        logger.error("An error occurred during dataset testing.", exc_info=True)
    finally:
        # Optional: Clean up dummy file
        # if os.path.exists(dummy_h5_path):
        #     os.remove(dummy_h5_path)
        #     logger.info(f"Removed dummy HDF5 file: {dummy_h5_path}")
        pass 