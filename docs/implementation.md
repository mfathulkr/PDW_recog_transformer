```markdown
# Implementation Report: Multi-Task Learning for Radar Signal Characterisation

**Document Version:** 1.0
**Date:** May 12, 2025
**Based on Paper:** Huang, Z., Pemasiri, A., Denman, S., Fookes, C., & Martin, T. (2024). *MULTI-TASK LEARNING FOR RADAR SIGNAL CHARACTERISATION*. arXiv:2306.13105v2.

## 1. Introduction

This document provides a comprehensive guide for implementing the experimental setup described in the research paper "MULTI-TASK LEARNING FOR RADAR SIGNAL CHARACTERISATION." The objective is to enable the replication of the training procedures, model evaluations, and key findings presented in Section 3 ("EXPERIMENTS") of the paper. This report assumes the use of Python and a common deep learning framework such as PyTorch or TensorFlow.

The paper introduces a Multi-Task Learning (MTL) framework for Radar Signal Characterisation (RSC), proposing several model architectures, including the IQ Signal Transformer (IQST), and a new synthetic dataset called RadChar. This report will focus on the practical steps to reproduce their experimental results.

## 2. Prerequisites

### 2.1. Software Stack
*   **Python:** Version 3.7 or higher is recommended.
*   **Deep Learning Framework:**
    *   PyTorch (e.g., version 1.10+) or
    *   TensorFlow (e.g., version 2.5+)
    *   The choice will influence specific API calls, but the core logic remains the same.
*   **Core Libraries:**
    *   **NumPy:** For efficient numerical computations and array manipulations, crucial for handling IQ data.
    *   **scikit-learn:** Useful for data splitting and potentially for some metric calculations if not readily available in the chosen deep learning framework.
*   **Version Control:**
    *   **Git:** Required for cloning the RadChar dataset repository from GitHub.

### 2.2. Hardware Requirements
*   **GPU:** An NVIDIA GPU with CUDA support is highly recommended to achieve reasonable training times, as the paper utilizes an Nvidia Tesla A100. A GPU with at least 16GB of VRAM is advisable, though this may vary based on the final model complexity and batch size.
*   **CPU:** A modern multi-core processor.
*   **RAM:** A minimum of 32GB of system RAM is suggested for handling data loading and preprocessing, especially with a dataset of 1 million waveforms.

### 2.3. Dataset
*   **RadChar Dataset:** This synthetic radar signals dataset is central to the experiments.
    *   **Source:** Downloadable from the GitHub repository linked in the paper: `https://github.com/abcxyzi/RadChar`
    *   **Contents:** Comprises 1 million unique radar waveforms. Each waveform includes:
        *   512 baseband IQ samples (⃗x<sub>i</sub> + j⃗x<sub>q</sub>).
        *   Labels for 5 radar signal types (for classification).
        *   Labels for 4 signal parameters (for regression):
            *   Pulse Width (PW, t<sub>pw</sub>)
            *   Pulse Repetition Interval (PRI, t<sub>pri</sub>)
            *   Number of pulses (n<sub>p</sub>)
            *   Pulse time delay (t<sub>d</sub>)
    *   **SNR Range:** Signals are generated with varying SNRs from -20 dB to 20 dB.

## 3. Dataset Preparation and Preprocessing

Accurate data preparation is crucial for replicating the paper's results.

### 3.1. Data Acquisition and Initial Setup
1.  **Clone the Dataset:**
    ```bash
    git clone https://github.com/abcxyzi/RadChar.git
    ```
2.  **Understand Data Structure:** Familiarize yourself with how the IQ data and labels are stored. The paper specifies 512 baseband IQ samples. For models like CNN1D and IQST, this is treated as a 2x512 tensor (separate I and Q channels). For CNN2D, it's reshaped into a 32x32 tensor.

### 3.2. Data Splitting
The dataset must be divided into training, validation, and test sets according to the paper's 70-15-15% ratio.

*   **Training Set:** 70% of the data.
*   **Validation Set:** 15% of the data.
*   **Test Set:** 15% of the data.

It's good practice to use a fixed random seed for splitting to ensure reproducibility. Stratification by signal type during splitting can also be beneficial.

```python
# Conceptual Python code for data splitting (using scikit-learn)
from sklearn.model_selection import train_test_split

# Assuming 'all_iq_data' and 'all_labels' are loaded
# all_labels should be a structured array or dictionary containing all 5 task labels

# First split: 70% train, 30% temporary (for validation + test)
train_iq, temp_iq, train_labels, temp_labels = train_test_split(
    all_iq_data, all_labels, test_size=0.30, random_state=42 # Use a consistent random_state
)

# Second split: 50% of temporary data for validation, 50% for test
# (0.15 / 0.30 = 0.5)
val_iq, test_iq, val_labels, test_labels = train_test_split(
    temp_iq, temp_labels, test_size=0.50, random_state=42
)

print(f"Training samples: {len(train_iq)}")
print(f"Validation samples: {len(val_iq)}")
print(f"Test samples: {len(test_iq)}")
```

### 3.3. Preprocessing Steps (as per Section 3.1 of the paper)

1.  **Standardize Raw IQ Samples:**
    *   Calculate the mean and standard deviation of the IQ samples. **Crucially, these statistics must be computed *only* from the training set.**
    *   Apply the Z-score standardization ( (value - mean) / std_dev ) to the I and Q channels of the training, validation, and test sets using the training set's statistics.

    ```python
    # Conceptual Python for IQ standardization (NumPy)
    # Assuming train_iq is a NumPy array of shape (num_samples, 2, 512)

    # Calculate mean and std for I and Q channels from the training set
    mean_i_train = np.mean(train_iq[:, 0, :])
    std_i_train = np.std(train_iq[:, 0, :])
    mean_q_train = np.mean(train_iq[:, 1, :])
    std_q_train = np.std(train_iq[:, 1, :])

    def standardize_iq_data(iq_data, mean_i, std_i, mean_q, std_q):
        processed_iq = np.copy(iq_data)
        processed_iq[:, 0, :] = (iq_data[:, 0, :] - mean_i) / std_i
        processed_iq[:, 1, :] = (iq_data[:, 1, :] - mean_q) / std_q
        return processed_iq

    train_iq_std = standardize_iq_data(train_iq, mean_i_train, std_i_train, mean_q_train, std_q_train)
    val_iq_std = standardize_iq_data(val_iq, mean_i_train, std_i_train, mean_q_train, std_q_train)
    test_iq_std = standardize_iq_data(test_iq, mean_i_train, std_i_train, mean_q_train, std_q_train)
    ```

2.  **Normalize Regression Labels:**
    *   Normalize the four regression target labels (t<sub>pw</sub>, t<sub>pri</sub>, n<sub>p</sub>, t<sub>d</sub>) to a range of **[0, 1]**.
    *   This is typically done using min-max scaling. Calculate the minimum and maximum values for each regression target **from the training set labels only.**
    *   Apply this scaling to the labels in the training, validation, and test sets. Store these min/max values, as they are needed to de-normalize predictions for interpretable MAE reporting.

    ```python
    # Conceptual Python for regression label normalization
    # Assuming train_labels is a dictionary or structured array
    # Example for one label, e.g., 'pw' (Pulse Width)

    min_max_scalers = {} # To store scalers for de-normalization

    for label_name in ['pw', 'pri', 'np', 'td']: # Assuming these are keys in your label structure
        min_val = np.min(train_labels[label_name])
        max_val = np.max(train_labels[label_name])
        min_max_scalers[label_name] = {'min': min_val, 'max': max_val}

        train_labels[f'{label_name}_norm'] = (train_labels[label_name] - min_val) / (max_val - min_val)
        val_labels[f'{label_name}_norm'] = (val_labels[label_name] - min_val) / (max_val - min_val)
        test_labels[f'{label_name}_norm'] = (test_labels[label_name] - min_val) / (max_val - min_val)
    ```

## 4. Model Architecture Implementation

The paper proposes a Multi-Task Learning (MTL) framework with hard parameter sharing, where a common backbone feeds into multiple task-specific heads.

### 4.1. Shared Feature Extraction Backbones (Section 2.3)

Implement the following backbone architectures:

*   **CNN2D:**
    *   Input: Single-channel 32x32 tensor (reshaped from raw IQ data). The 512 complex IQ samples (1024 real values) are reshaped.
    *   Architecture: "a single convolution layer with 8 filters using a kernal size of 2×2 followed by a 2×2 max pooling operation."
    *   Activation: ReLU.
    *   Dropout: 0.25.
*   **CNN1D:**
    *   Input: Two separate I and Q channels of shape 2x512.
    *   Architecture: "uses 1D convolutional and max pooling operations instead, while maintaining the same number of filters as CNN2D (8 filters)." Kernel size for 1D conv/pool needs to be chosen (e.g., kernel size 3 or similar for 1D).
    *   Activation: ReLU.
    *   Dropout: 0.25.
*   **IQ Signal Transformer (IQST-S - Standard):**
    *   Input: 2x512 tensor (I and Q channels).
    *   Patch Embedding: Dual-channel IQ data flattened to form 8x1x128 blocks (tokens). Dense linear projection to 8 learnable patch embeddings (embedding dimension 768).
    *   Positional Embeddings: Standard positional embeddings added.
    *   Input to Transformer Encoder: 128x8.
    *   Learnable Embedding: Additional learnable embedding for common feature sharing (similar to class token in ViT).
    *   Transformer Encoder: Standard architecture from Vaswani et al. [13].
        *   Activation: GELU.
        *   Layers: 3 encoder layers.
        *   Attention: 3 multi-head attention blocks.
    *   Output to Task Heads: Output from the shared learnable embedding (1x128 feature map).
*   **IQST-L (Large):**
    *   Same as IQST-S but with increased capacity:
        *   Layers: 6 encoder layers.
        *   Attention: 9 multi-head attention blocks.

### 4.2. Multi-Task Learning Framework (Section 2.2)

*   **Hard Parameter Sharing:** All tasks share one of the backbones described above.
*   **Task-Specific Heads:** Five parallel heads branch out from the shared backbone's output.
    *   1 Classification Head: For signal type identification.
    *   4 Regression Heads: For t<sub>pw</sub>, t<sub>pri</sub>, n<sub>p</sub>, t<sub>d</sub> estimation.

### 4.3. Structure of Task-Specific Heads (Section 2.2)

Each of the 5 task-specific heads has an identical lightweight structure:
1.  **Convolutional Layer:**
    *   A single convolutional layer. The paper mentions "a kernal size of 3×3". If the backbone output is a 1D feature map (e.g., 1x128 from IQST), this should be interpreted as a 1D convolution (e.g., kernel size 3).
    *   Number of filters: "driven by the output dimension of the shared backbone." This needs careful interpretation – likely a fixed number of filters (e.g., 32, 64) that then leads to the dense layer.
2.  **Batch Normalization:** Applied *before* the activation function.
3.  **Activation Function:** ReLU.
4.  **Dropout:** Rate of 0.25 applied after the convolutional layer (or its activation).
5.  **Dense Layer (Fully Connected):**
6.  **Dropout:** Rate of 0.5 applied after the dense layer.
7.  **Output Layer:**
    *   **Classification Head:** Softmax activation to output probabilities for signal classes.
    *   **Regression Heads:** Linear activation (i.e., direct output from the dense layer) for parameter predictions.

## 5. Training Procedure (Section 3.1)

The models are trained end-to-end by optimizing a compound multi-task loss.

### 5.1. Optimizer
*   **Adam Optimizer:** Use the standard implementation from PyTorch (`torch.optim.Adam`) or TensorFlow (`tf.keras.optimizers.Adam`).

### 5.2. Parameter Initialization
*   **LeCun Initialization:** (Also known as LeCun Normal). Weights are drawn from a normal distribution N(0, σ<sup>2</sup>) where σ = sqrt(1/fan_in).
    *   PyTorch: Can be implemented using `torch.nn.init.normal_` after calculating the correct std, or `torch.nn.init.kaiming_normal_` with `a=0, mode='fan_in', nonlinearity='leaky_relu'` (where leaky_relu with 0 slope is ReLU) might be close if `nonlinearity='relu'` is not directly supported for variance scaling in this way. Check PyTorch docs for `Linear` and `Conv` layer default initializations or apply manually.
    *   TensorFlow/Keras: `tf.keras.initializers.LecunNormal()`.

### 5.3. Training Hyperparameters
*   **Learning Rate:** 5e-4 (0.0005).
*   **Batch Size:** 64.
*   **Number of Epochs:** 100.

### 5.4. Loss Function (Section 2.2 & 3.3)

A compound multi-task loss (L<sub>mtl</sub>) is minimized:
L<sub>mtl</sub>(θ<sub>sh</sub>, θ<sub>1</sub>, ..., θ<sub>5</sub>) = Σ<sup>5</sup><sub>i=1</sub> w<sub>i</sub>L<sub>i</sub>(θ<sub>sh</sub>, θ<sub>i</sub>)

*   **Individual Task Losses (L<sub>i</sub>):**
    *   **Classification Task (L<sub>class</sub>):** Categorical Cross-Entropy Loss.
        *   PyTorch: `torch.nn.CrossEntropyLoss()`
        *   TensorFlow/Keras: `tf.keras.losses.CategoricalCrossentropy(from_logits=False)` (if model outputs probabilities via softmax) or `(from_logits=True)` (if model outputs raw scores).
    *   **Regression Tasks (L<sub>pw</sub>, L<sub>pri</sub>, L<sub>np</sub>, L<sub>td</sub>):** L1 Loss (Mean Absolute Error).
        *   PyTorch: `torch.nn.L1Loss()`
        *   TensorFlow/Keras: `tf.keras.losses.MeanAbsoluteError()`

*   **Task Weights (w<sub>i</sub>):** Based on the ablation study (Section 3.3):
    *   Weight for classification loss (w<sub>class</sub>): **0.1**
    *   Weight for each of the 4 regression losses (w<sub>reg</sub>): **0.225**
    *   (Total weight sum: 0.1 + 4 * 0.225 = 1.0)

```python
# Conceptual loss calculation in a training loop (PyTorch-like)
# model_outputs = model(input_batch) # Assuming model returns a tuple/dict of outputs
# target_labels = ... # Corresponding ground truth labels (classification + normalized regression)

loss_class = classification_criterion(model_outputs['class'], target_labels['class'])
loss_pw = regression_criterion(model_outputs['pw'], target_labels['pw_norm'])
loss_pri = regression_criterion(model_outputs['pri'], target_labels['pri_norm'])
loss_np = regression_criterion(model_outputs['np'], target_labels['np_norm'])
loss_td = regression_criterion(model_outputs['td'], target_labels['td_norm'])

w_classification = 0.1
w_regression = 0.225

total_loss = (w_classification * loss_class) + \
             (w_regression * loss_pw) + \
             (w_regression * loss_pri) + \
             (w_regression * loss_np) + \
             (w_regression * loss_td)

# --- Backpropagation ---
# optimizer.zero_grad()
# total_loss.backward()
# optimizer.step()
```

## 6. Evaluation Protocol (Section 3.2)

Model performance is evaluated on the held-out **test set**.

### 6.1. Evaluation Metrics

*   **Classification Task:**
    *   **Metric:** Classification Accuracy.
    *   Formula: (Number of correctly classified samples) / (Total number of test samples).
*   **Regression Tasks (t<sub>pw</sub>, t<sub>pri</sub>, n<sub>p</sub>, t<sub>d</sub>):**
    *   **Metric:** Mean Absolute Error (MAE).
    *   Calculated independently for each of the four regression tasks.
    *   **Important:** MAE should be reported in the original units of the parameters (e.g., µs for t<sub>pw</sub>, t<sub>pri</sub>, t<sub>d</sub>). This requires de-normalizing the model's predictions (which are in the [0, 1] range) back to their original scale using the `min_val` and `max_val` stored from the training set label normalization step.

    ```python
    # Conceptual de-normalization for MAE calculation
    def denormalize_prediction(norm_pred, min_val, max_val):
        return norm_pred * (max_val - min_val) + min_val

    # Example for PW
    # predicted_pw_normalized = model_output_for_pw_on_test_set
    # true_pw_original_scale = test_labels['pw'] # Original scale labels
    # min_pw_train, max_pw_train = min_max_scalers['pw']['min'], min_max_scalers['pw']['max']
    
    # predicted_pw_original_scale = denormalize_prediction(predicted_pw_normalized, min_pw_train, max_pw_train)
    # mae_pw = np.mean(np.abs(predicted_pw_original_scale - true_pw_original_scale))
    ```

### 6.2. SNR-Dependent Evaluation

*   The paper evaluates performance across a range of SNRs from **-20 dB to 20 dB**.
*   Table 1 specifically reports results at **-10 dB, 0 dB, and 10 dB SNR**.
*   The test set evaluation should be structured to filter samples by their SNR levels to reproduce these specific results and the plots in Figure 3. The RadChar dataset should contain SNR information for each sample.

## 7. Ablation Study Considerations (Section 3.3)

These insights from the paper can guide implementation and verify behavior:

*   **Depth of Convolutional Layers in Heads:** The paper found that deeper convolutional networks in the task-specific heads negatively impacted regression tasks (higher errors), possibly due to reduced temporal resolution. The specified single convolutional layer design in heads should be followed.
*   **Task Weighting Strategy:** The chosen weights (w<sub>class</sub>=0.1, w<sub>reg</sub>=0.225) were found to provide stable task performance. The paper notes these weights aim for a "relatively even distribution of w<sub>i</sub>L<sub>i</sub> during model initialisation." This could be an interesting diagnostic to check during the first few training steps. Simply increasing a task's weight did not necessarily improve its individual test performance.

## 8. Conceptual Code Structure

A modular organization of the codebase is recommended for clarity and maintainability:

*   `dataset.py` (or `data_loader.py`):
    *   Contains the PyTorch `Dataset` or TensorFlow `tf.data.Dataset` class for RadChar.
    *   Handles loading, parsing IQ data and labels.
    *   Implements preprocessing: standardization of IQ, normalization of regression labels.
    *   Provides data loaders/iterators for training, validation, and test sets.
*   `models.py`:
    *   Definitions for the shared backbone architectures (CNN1D, CNN2D, IQST_S, IQST_L).
    *   Implementation of the generic task-specific head module.
    *   The main MTL model class that combines a chosen backbone with the five task heads.
*   `train.py`:
    *   Main script for initiating and managing the training process.
    *   Handles model instantiation, optimizer setup, loss function definition.
    *   Implements the training loop (epochs, batches), including forward pass, loss calculation, backward pass, and optimizer step.
    *   Includes a validation loop for monitoring performance on the validation set.
    *   Manages model checkpointing (saving best models).
    *   Integrates logging of metrics (e.g., using TensorBoard, Weights & Biases, or simple print statements).
*   `evaluate.py`:
    *   Script for loading a trained model checkpoint.
    *   Performs evaluation on the test set.
    *   Calculates and reports classification accuracy and de-normalized MAE for regression tasks.
    *   Includes logic to filter test data by SNR and report metrics per SNR level.
*   `config.py` (or argument parsing):
    *   Centralized location for hyperparameters (learning rate, batch size, epochs, task weights, model choices, dataset paths, etc.).
*   `utils.py`:
    *   Contains utility functions, e.g., custom initialization functions (if LeCun is not directly available), metric calculation helpers, de-normalization functions, etc.

## 9. Implementation Checklist

Ensure the following key aspects are correctly implemented:

*   [ ] **Dataset:** RadChar downloaded and correctly parsed.
*   [ ] **Data Split:** 70-15-15% for train-validation-test, consistently.
*   [ ] **IQ Standardization:** Using *training set* mean/std, applied to all splits.
*   [ ] **Regression Label Normalization:** To [0,1] using *training set* min/max, applied to all splits. Store scalers.
*   [ ] **Backbone Architectures:** CNN1D, CNN2D, IQST-S, IQST-L implemented as per paper's description.
*   [ ] **Task-Specific Heads:** Correct structure (1 conv + 1 dense, ReLU, BatchNorm, specified dropout rates).
*   [ ] **Optimizer:** Adam.
*   [ ] **Initialization:** LeCun initialization for model parameters.
*   [ ] **Hyperparameters:** Learning rate (5e-4), batch size (64), epochs (100).
*   [ ] **Loss Function:** Compound MTL loss with Categorical Cross-Entropy (classification) and L1 Loss (regression).
*   [ ] **Task Weights:** w<sub>class</sub>=0.1, w<sub>reg</sub>=0.225 for each regression task.
*   [ ] **Evaluation Metrics:** Classification Accuracy; MAE for regression (predictions de-normalized to original scale).
*   [ ] **SNR-Specific Evaluation:** Capability to evaluate at -10, 0, and 10 dB SNR on the test set.

## 10. Conclusion

This implementation report provides a detailed roadmap for reproducing the experiments presented in "MULTI-TASK LEARNING FOR RADAR SIGNAL CHARACTERISATION." By adhering to the specified data handling, model architectures, training regimen, and evaluation protocols, researchers should be able to validate the paper's findings and build upon its contributions to the field of radar signal characterisation using deep learning. Careful attention to each step, especially the nuances of preprocessing and loss configuration, is paramount for a successful replication.
```