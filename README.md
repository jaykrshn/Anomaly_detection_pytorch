Here’s an updated version of your README file with the CUDA and PyTorch instructions included:

---

# ResNet-Style VAE with Adjustable Perceptual Loss  

This repository provides a **ResNet-Style Variational Autoencoder (VAE)** with an adjustable **perceptual loss** using a pre-trained **VGG19** network.  

---

## Other Features  

1. **Flexible Autoencoder Modes**  
   - **Unsupervised Autoencoder**  
   - **Supervised Autoencoder**  

2. **Hyperparameter Tuning**  
   - Integration with the **Optuna** package for automated and efficient hyperparameter optimization.  

3. **Experiment Tracking**  
   - Supports **MLflow** for tracking experiments, logging parameters, metrics, and models.  

---
## Installation & Setup

### 1. Create a Python Environment

Create a new Python environment (recommended using `venv` or `conda`) and install the required dependencies. 
*Note: tested in python 3.9.19*

```bash
# Create a new virtual environment
python3 -m venv .venv

# Activate the virtual environment

# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate

# Install required libraries
pip install -r requirements.txt
```

1. **CUDA Support**  
   - This code requires a CUDA-compatible GPU to run smoothly.  

2. **PyTorch Installation**  
   - Install the appropriate PyTorch version based on your CUDA version. To determine your CUDA version, run:  

     ```bash
     nvcc --version
     ```

   - Then, install PyTorch using the [PyTorch Installation Guide](https://pytorch.org/get-started/locally/).

---
### 2. Start MLflow Server

MLflow is used for experiment tracking, including the hyperparameter optimization of the SVC model.

To start the MLflow server, run:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 8000
```

---

### 3. Training the Autoencoder  

You can train the VAE in either **supervised** or **unsupervised** mode:  

#### a) Supervised Training  

Run the following command:  

```bash
python train_vae_supervised.py -mn model_name 
```

- **Arguments**:  
   - `-mn`: Model name to save after training.  

#### b) Unsupervised Training  

Run the following command:  

```bash
python train_vae.py --mn model_name --dataset_root #path to dataset root#
```

- **Arguments**:  
   - `--mn`: Model name to save after training.  
   - `--dataset_root`: Path to the dataset root directory.  

---

#### c)  Hyperparameter Tuning  

To tune the model's hyperparameters using **Optuna**, run the following command:  

```bash
python ae_tuning.py --dataset_root #path to dataset root#
```

- **Arguments**:  
   - `--dataset_root`: Path to the dataset root directory.  

--- 



## Acknowledgements  

This project builds on concepts and code adapted from [CNN-VAE by Luke Ditria](https://github.com/LukeDitria/CNN-VAE). 
Refer the repo for more details
---

