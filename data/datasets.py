import torch
from torch.utils.data import Dataset, DataLoader
import medmnist
from medmnist import INFO
import numpy as np
from PIL import Image

class MedMNISTDataset(Dataset):
    def __init__(self, dataset_name, split='train', transform=None):
        """
        Args:
            dataset_name (str): Name of the MedMNIST dataset
            split (str): One of ['train', 'val', 'test']
            transform (callable, optional): Optional transform to be applied
        """
        # Convert dataset name to lowercase for INFO lookup
        dataset_name = dataset_name.lower()
        
        # Get dataset info
        self.info = INFO[dataset_name]
        
        # Get the correct class name (e.g., pathmnist -> PathMNIST)
        if '3d' in dataset_name:
            class_name = dataset_name.replace('3d', '3D').title().replace('3d', '3D')
        else:
            class_name = dataset_name[:-5].title() + 'MNIST'
        
        DataClass = getattr(medmnist, class_name)
        self.data = DataClass(split=split, download=True)
        self.transform = transform
        self.task = self.info['task']
        self.n_channels = self.info['n_channels']
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, label = self.data[idx]
        
        # Convert PIL Image to numpy array
        if isinstance(img, Image.Image):
            img = np.array(img)
        
        # Convert to float and normalize to [-1, 1]
        img = img.astype(np.float32) / 127.5 - 1.0
        
        # Convert RGB to grayscale if needed
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = np.mean(img, axis=2, keepdims=True)
        elif len(img.shape) == 2:
            img = img[..., np.newaxis]
        
        # Convert to tensor with shape (C,H,W)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        label = torch.from_numpy(label).float()
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

def get_dataloader(dataset_name, split='train', batch_size=32, num_workers=4):
    """
    Create a dataloader for a specific MedMNIST dataset
    
    Args:
        dataset_name (str): Name of the MedMNIST dataset
        split (str): One of ['train', 'val', 'test']
        batch_size (int): Batch size
        num_workers (int): Number of worker processes
        
    Returns:
        DataLoader: PyTorch dataloader
    """
    dataset = MedMNISTDataset(dataset_name, split)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )

# Dictionary mapping dataset names to their task types
TASK_TYPES = {
    'pathmnist': 'multiclass',
    'chestmnist': 'multilabel',
    'dermamnist': 'multiclass',
    'octmnist': 'multiclass',
    'pneumoniamnist': 'binary',
    'retinamnist': 'multiclass',
    'breastmnist': 'binary',
    'bloodmnist': 'multiclass',
    'tissuemnist': 'multiclass',
    'organamnist': 'multiclass',
    'organcmnist': 'multiclass',
    'organsmnist': 'multiclass',
    'organmnist3d': 'multiclass',
    'nodulemnist3d': 'binary',
    'adrenalmnist3d': 'binary',
    'fracturemnist3d': 'multiclass',
    'vesselmnist3d': 'binary',
    'synapsemnist3d': 'binary'
} 