## Description
This repository contains the code used to train, validate, and evaluate the ML models trained to produce point-wise predictions of stress-strain curves. This was originally applied to predicting the compressive behaviour of syntactic foams for varying material parameters, however, it may be applied effectively to other materials and material parameters with different stress-strain responses.

## Setup

1. Setup your Python environment using venv (Windows):
    ```sh
    py -m venv .venv
    ```
2. Activate the environment in the terminal:
    ```sh
    .\.venv\Scripts\activate
    ```
3. Install dependencies:
    ```sh
    py -m pip install -r requirements.txt
    ```
    If you don't have an NVIDIA GPU, just install PyTorch for CPU:
    ```sh
    py -m pip install torch
    ```    
    Otherwise, you should install PyTorch for GPU (make sure you have the necessary drivers installed for CUDA):
    ```sh
    py -m pip install torch --index-url https://download.pytorch.org/whl/cu124
    ```

## Usage

```main_notebook.ipynb``` is the main Jupyter notebook which calls the functions included in the 'my_utils' folder. This includes data pre-processing, model selection and configuration, model training, validation, testing, saving and loading.
