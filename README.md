# SCALE: Scale-aware Cross-modal Attention Learning for Medical Vision

> 🔬 **Work in Progress**: This repository contains the implementation of SCALE (Scale-aware Cross-modal Attention Learning), a framework for multi-scale medical image analysis. The codebase is actively being developed and improved.

## Overview

SCALE is a PyTorch-based framework that introduces a novel approach to medical image analysis by combining:
1. Scale-aware attention mechanisms for handling multi-resolution features
2. Cross-modal fusion between 2D and 3D medical imaging data
3. Interpretable deep learning for medical applications

The framework is evaluated on publicly available MedMNIST datasets, which include both 2D and 3D medical imaging data across various modalities:
- 2D datasets: PathMNIST (pathology), ChestMNIST (chest X-ray), BloodMNIST (blood cell)
- 3D datasets: OrganMNIST3D (abdominal CT), NoduleMNIST3D (chest CT)

## Features

- Multi-modal processing of 2D and 3D medical images
- Advanced attention mechanisms:
  - Scale-aware attention for multi-resolution feature learning
  - Cross-modal attention for 2D-3D feature fusion
- Support for multiple medical imaging tasks:
  - Binary classification (e.g., nodule detection)
  - Multi-class classification (e.g., organ segmentation)
  - Multi-label classification (e.g., pathology classification)
- Built-in interpretability tools
- Google Colab integration for easy experimentation
- Comprehensive evaluation metrics

## Quick Start

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/salilp42/SCALE.git
cd SCALE
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Google Colab

We provide a Colab notebook for quick experimentation without local setup. See `colab_notebook.py` for details.

## Usage

Basic training example:
```python
from model import MSMedVision
from dataset import get_dataset

# Load dataset (automatically downloads MedMNIST data)
train_loader, val_loader = get_dataset('pathmnist', batch_size=32)

# Initialize model
model = MSMedVision(dataset='pathmnist')

# Train
python train_all.py --task pathmnist --batch-size 32 --epochs 100 --lr 0.001
```

## Model Architecture

The framework consists of three main components:

1. **Dual-Stream Feature Extraction**
   - 2D Stream: Processes planar medical images (e.g., X-rays, pathology slides)
   - 3D Stream: Handles volumetric data (e.g., CT, MRI volumes)
   
2. **Cross-Modal Feature Fusion**
   - Scale-aware attention mechanism for handling multi-resolution features
   - Adaptive feature calibration between 2D and 3D modalities
   
3. **Task-Specific Heads**
   - Configurable for different medical tasks
   - Built-in interpretability hooks for attention visualization

## Project Structure

```
SCALE/
├── src/                    # Source code
│   ├── models/            # Neural network models
│   │   └── model.py       # Core model implementation
│   ├── data/              # Data handling
│   │   └── dataset.py     # Dataset loading and preprocessing
│   ├── visualization/     # Visualization tools
│   │   └── interpretability.py  # Model interpretability
│   └── utils/             # Utility functions
│       └── colab_setup.py # Colab integration utilities
├── scripts/               # Training and analysis scripts
│   ├── train_all.py      # Main training pipeline
│   ├── analyze_chestmnist.py  # Analysis examples
│   └── colab_notebook.py  # Colab integration notebook
├── requirements.txt       # Project dependencies
├── LICENSE               # MIT License
└── README.md            # Project documentation
```

## Data

This project uses the [MedMNIST v2](https://medmnist.com/) dataset collection, which provides standardized biomedical images for benchmarking. The datasets are automatically downloaded when running the training scripts. All data usage complies with MedMNIST's terms and conditions.

## Contributing

This is a work in progress and contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you find this code useful in your research, please consider citing:

```bibtex
@misc{scale2024,
  author = {Patel, Salil},
  title = {SCALE: Scale-aware Cross-modal Attention Learning for Medical Vision},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/salilp42/SCALE}}
}
```

