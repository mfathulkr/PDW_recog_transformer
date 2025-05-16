# PDW Recognition Transformer

This project implements the research paper "MULTI-TASK LEARNING FOR RADAR SIGNAL CHARACTERISATION" (arXiv:2306.13105v2).

## Project Goal

Replicate the experiments from the paper, focusing on multi-task learning (MTL) for radar signal classification and parameter regression using various deep learning architectures, including the proposed IQ Signal Transformer (IQST).

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd PDW_recog_transformer
    ```
2.  **Create Conda Environment:**
    ```bash
    conda env create -f environment.yml
    ```
3.  **Activate Environment:**
    ```bash
    conda activate pdw_transformer
    ```
4.  **Download Data:** Obtain the RadChar dataset (.h5 file, e.g., `RadChar-Baseline.h5`) from the link provided in `docs/radcharRM.md` or the paper's repository (`https://github.com/abcxyzi/RadChar`) and place it in the `data/` directory.

## Usage

Use the main entry point `main.py`.

*   **Train a model:**
    ```bash
    python main.py --mode train --config <path_to_config_file> --model <model_name> 
    ```
*   **Evaluate a model:**
    ```bash
    python main.py --mode evaluate --config <path_to_config_file> --model <model_name> --checkpoint <path_to_checkpoint>
    ```

(Detailed command-line arguments will be defined later.)

## Project Structure

```
PDW_recog_transformer/
├── config/                 # Configuration files
├── data/                   # RadChar dataset files (.h5)
├── docs/                   # Paper, Readme, Figures, Implementation Guide
├── src/                    # Source code
│   ├── data_handling/      # Data loading & preprocessing
│   ├── models/             # Model architectures
│   └── utils/                # Utility functions
├── .gitignore
├── environment.yml         # Conda environment file
├── main.py                 # Main entry point
└── README.md               # This file
``` 