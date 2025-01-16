import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import medmnist
from medmnist import INFO

class MedMNISTDataset(Dataset):
    def __init__(self, task, split='train'):
        super().__init__()
        
        # Get dataset info
        self.task = task.lower()
        self.info = INFO[task]
        self.task_type = self.info['task']
        self.n_channels = self.info['n_channels']
        self.is_3d = '3d' in task.lower()
        self.split = split
        
        # Set number of classes based on task type
        if self.task_type == 'multi-label, binary-class':
            self.n_classes = len(self.info['label'])
        else:
            self.n_classes = len(self.info['label'])
        
        # Load dataset
        DataClass = getattr(medmnist, INFO[task]['python_class'])
        self.dataset = DataClass(split=split, download=True)
        self.dataset.task_type = self.task_type
        self.dataset.n_classes = self.n_classes
        self.dataset.n_channels = self.n_channels
        
        # Enhanced transforms for 2D data
        if not self.is_3d:
            if split == 'train':
                if self.n_channels == 3:
                    self.transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(10),
                        transforms.ColorJitter(brightness=0.2, contrast=0.2),
                        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                else:
                    self.transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(10),
                        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5], std=[0.5])
                    ])
            else:  # val/test transforms
                if self.n_channels == 3:
                    self.transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                else:
                    self.transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5], std=[0.5])
                    ])
        
        # Calculate class weights for balanced sampling
        if split == 'train' and self.task_type != 'multi-label, binary-class':
            self.class_weights = self._compute_class_weights()
        
    def _compute_class_weights(self):
        """Compute class weights for balanced sampling."""
        labels = [self.dataset[i][1] for i in range(len(self.dataset))]
        if isinstance(labels[0], np.ndarray):
            labels = [label.item() for label in labels]
        class_counts = np.bincount(labels)
        total = len(labels)
        class_weights = total / (len(class_counts) * class_counts)
        return torch.FloatTensor(class_weights)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        img, label = self.dataset[index]
        
        # Handle 3D data
        if self.is_3d:
            if isinstance(img, np.ndarray):
                if img.dtype == np.float32:
                    img = (img * 255).astype(np.uint8)
                img = torch.from_numpy(img).float()
                
                # Ensure correct dimensions [C, D, H, W]
                if img.dim() == 3:  # [D, H, W]
                    img = img.unsqueeze(0)  # Add channel dimension
                elif img.dim() == 4:  # [C, D, H, W]
                    pass
                else:
                    raise ValueError(f"Unexpected 3D image dimensions: {img.shape}")
                
                # Apply 3D augmentations during training
                if self.split == 'train':
                    # Random flip
                    if torch.rand(1) > 0.5:
                        img = torch.flip(img, dims=[2])  # Flip height
                    if torch.rand(1) > 0.5:
                        img = torch.flip(img, dims=[3])  # Flip width
                    
                    # Add random noise
                    if torch.rand(1) > 0.5:
                        noise = torch.randn_like(img) * 0.02
                        img = img + noise
                
                # Normalize consistently
                img = img / 255.0  # Scale to [0, 1]
                img = (img - 0.5) / 0.5  # Normalize to [-1, 1]
        else:
            # Convert to PIL Image if necessary
            if not isinstance(img, Image.Image):
                if isinstance(img, np.ndarray):
                    if img.dtype == np.float32:
                        img = (img * 255).astype(np.uint8)
                    if len(img.shape) == 2:
                        img = Image.fromarray(img, mode='L')
                    else:
                        img = Image.fromarray(img)
            
            # Apply transforms
            img = self.transform(img)
        
        # Handle different target types
        if self.task_type == 'multi-label, binary-class':
            target = torch.FloatTensor(label)
        else:
            target = torch.tensor(label, dtype=torch.long).squeeze()
            
        return img, target 