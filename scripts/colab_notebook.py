# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Change to your project directory
%cd /content/drive/MyDrive/ms_medvision

# Check if all required files are present
import os
required_files = ['train_all.py', 'model.py', 'dataset.py', 'visualization.py']
for file in required_files:
    if not os.path.exists(file):
        raise FileNotFoundError(f"Missing required file: {file}")
print("All required files found!")

# Install Miniconda
!wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
!bash Miniconda3-latest-Linux-x86_64.sh -b -f -p /usr/local/miniconda3
!rm Miniconda3-latest-Linux-x86_64.sh

# Initialize conda in shell
import sys
sys.path.append('/usr/local/miniconda3/bin')
!conda init bash

# Create and activate conda environment
!conda create -n medvision python=3.10 -y
!source activate medvision

# Install required packages in conda environment
!conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
!pip install scikit-learn medmnist seaborn tqdm matplotlib numpy scipy plotly opencv-python

# Verify GPU is available and show info
import torch
print("\nGPU Information:")
!nvidia-smi
print(f"\nUsing device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Set PyTorch to use GPU memory more efficiently
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    # Set memory allocation to be more efficient
    torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
    torch.backends.cuda.max_split_size_mb = 512  # Limit memory splits

print("\nStarting training...")
# Run training script with debug mode and appropriate batch size
!PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 python train_all.py --debug --batch-size 32

print("\nTraining complete! Check the results directory for outputs.") 