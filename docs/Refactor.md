# Refactoring and Review Notes

This document outlines areas identified during the code review that may require further attention, fixes, or corrections.

## 1. `src/evaluate.py` - Evaluation Script

### 1.1. Critical: Incorrect De-normalization for Regression MAE
*   **File:** `src/evaluate.py`
*   **Function:** `evaluate_model`
*   **Lines (approximate, based on review):** Around lines 260-280 for overall MAE, and 320-340 for per-SNR MAE.
*   **Issue:** The de-normalization of regression predictions and targets currently uses `dataset_stats['mean'][task]` and `dataset_stats['std'][task]`. This is incorrect. The `dataset_stats` dictionary (derived from `RadCharDataset.get_stats()`) stores regression normalization parameters as `label_min_max[task]['min']` and `label_min_max[task]['max']`.
*   **Required Fix:**
    *   Modify the de-normalization calls to use the correct min/max values.
    *   Example for overall MAE section (similar logic for per-SNR):
        ```python
        # Incorrect current logic (conceptual)
        # preds_denorm = denormalize(preds, dataset_stats['mean'][task], dataset_stats['std'][task])
        # targs_denorm = denormalize(targs, dataset_stats['mean'][task], dataset_stats['std'][task])

        # Corrected logic (conceptual)
        min_val = dataset_stats['label_min_max'][task]['min']
        max_val = dataset_stats['label_min_max'][task]['max']
        preds_denorm = denormalize(preds.cpu().numpy(), min_val, max_val) # Ensure denormalize handles numpy arrays
        targs_denorm = denormalize(targs.cpu().numpy(), min_val, max_val) # Ensure denormalize handles numpy arrays
        # Convert back to tensors if calculate_mae expects tensors, or adapt calculate_mae
        preds_denorm_tensor = torch.tensor(preds_denorm, device=preds.device)
        targs_denorm_tensor = torch.tensor(targs_denorm, device=targs.device)
        mae_denorm = calculate_mae(preds_denorm_tensor, targs_denorm_tensor)
        ```
    *   Ensure the `denormalize` function in `utils/helpers.py` can handle tensor/NumPy array inputs if `preds` and `targs` are not single values (it currently seems to expect float or ndarray, so `preds.cpu().numpy()` is appropriate).

### 1.2. Consistency: MAE Unit Conversion to Microseconds (µs)
*   **File:** `src/evaluate.py`
*   **Function:** `evaluate_model`
*   **Issue:** The conversion of MAE values for pulse width (`pw`), PRI (`pri`), and time delay (`td`) to microseconds (by multiplying by `1e6`) is present in `src/train.py` within `validate_one_epoch` but appears to be missing in `evaluate_model` in `src/evaluate.py` for the final reported overall and per-SNR MAEs.
*   **Required Fix:** After calculating the de-normalized MAE for `pw`, `pri`, and `td` in `evaluate_model`, multiply the result by `1e6` to convert it to microseconds, ensuring consistency with the paper's Table 1 and validation metrics during training.
    ```python
    # Example for overall MAE section (similar logic for per-SNR)
    # ... after mae_denorm is calculated ...
    if task in ['pw', 'pri', 'td']:
        mae_denorm *= 1e6
    results[f'{task}_mae_denormalized'] = mae_denorm
    logger.info(f"Regression MAE ({task}, De-normalized, µs for relevant): {mae_denorm:.4f}")
    ```

### 1.3. Minor: SNR Info Check in `evaluate_model`
*   **File:** `src/evaluate.py`
*   **Function:** `evaluate_model`
*   **Issue:** The check for SNR information (`has_snr_info`) involves `next(iter(test_loader))`, which consumes the first batch. The comment notes this.
*   **Suggestion:** While not critical for correctness (evaluation is usually on a static dataset), for very small test sets or more robust handling, consider:
    1.  Checking an attribute of the `test_loader.dataset` object if it indicates SNR availability (e.g., `test_loader.dataset.has_snr_info` if such an attribute were added to `RadCharDataset`).
    2.  Re-instantiating the `test_loader` after the check if it's crucial not to skip the first batch and the dataset is small. (Current approach is likely fine for typical test set sizes).

## 2. `src/models/iqst_backbone.py` - IQST Output vs. Paper Description //DONE

*   **File:** `src/models/iqst_backbone.py` (`IQSTBackbone` class)
*   **File (Paper):** `docs/paper.md` (Section 2.3)
*   **Issue:** The paper states: "We feed the outputs from the shared embedding as a 1×128 feature map into each task-specific head". The implemented `IQSTBackbone` outputs a `(B, 768, 1)` feature map (embedding_dim is 768).
*   **Current Handling:** The `TaskHead` implementation correctly accepts `in_channels=768` and `backbone_output_length=1`. Its internal Conv1D layer then processes these 768 channels (e.g., down to 32 filters as per `config.task_head.num_filters`).
*   **Review Point:** This is not a bug in the current code, as the components are wired to work together. However, it's a slight deviation from the paper's literal description of a "1x128 feature map" entering the heads.
*   **Action:** No immediate code change needed unless strict adherence to a 128-dim interface *before* the head's Conv1d is desired. If so, an additional `nn.Linear(768, 128)` projection would need to be added in `MTLModel` between the IQST backbone output and the task heads, and the heads' `input_channels` adjusted. The current implementation (768 channels into head's Conv1d) is functional. This is more of a documentation/consistency check against the paper.

## 3. General Code and Configuration

### 3.1. Stratified Splitting in `src/data_handling/dataset.py`
*   **File:** `src/data_handling/dataset.py`
*   **Function:** `get_dataloaders`
*   **Issue:** Stratified splitting logic is commented out: `# stratify=all_labels_signal_type`.
*   **Recommendation:** Consider enabling stratified splitting by `signal_type` if not already tested. This ensures that class proportions are maintained across train/validation/test splits, which is generally good practice, especially if some signal types are less frequent. This would require loading the `signal_type` column for all samples before splitting.

### 3.2. LeCun vs. Kaiming Initialization in `src/models/mtl_model.py`
*   **File:** `src/models/mtl_model.py`
*   **Function:** `_init_weights`
*   **Issue:** The paper specifies "LeCun initialisation". The code uses `nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')`.
*   **Clarification:** Kaiming Normal with `a=0` (default for `nonlinearity='relu'`) and `mode='fan_in'` is mathematically equivalent to LeCun Normal (`sqrt(1/fan_in)`) when using ReLU-like activations.
*   **Action:** This is acceptable and a common practical choice. Could add a comment explicitly stating this equivalence for clarity if desired. No code change strictly needed.

### 3.3. Checkpoint Loading in `src/evaluate.py` - Main Block
*   **File:** `src/evaluate.py`
*   **Function:** `if __name__ == '__main__':` (expected structure, not fully visible in review)
*   **Recommendation:** Ensure the main execution block in `evaluate.py`:
    1.  Loads the full configuration dictionary that was saved with the `best_model.pth` checkpoint (the `train.py` script saves `config` in the checkpoint).
    2.  Uses this loaded training-time config to instantiate the `MTLModel` (to ensure architecture matches).
    3.  Uses this loaded training-time config to call `get_dataloaders` to retrieve the correct `dataset_stats` (mean/std for IQ, and **min/max for regression labels**) that were used during the training of the loaded checkpoint. This is critical for correct de-normalization.
    4.  The `test_loader` used for the actual evaluation loop can then be created based on parameters in an evaluation-specific config or the loaded training config.

This list should provide a good starting point for refactoring and ensuring the implementation is robust and aligns closely with the paper's intent where necessary. 