import medmnist
from medmnist import INFO
import torch
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms

def analyze_chestmnist():
    # Load dataset
    data_flag = 'chestmnist'
    download = True
    
    print(f"\nAnalyzing {data_flag} dataset...")
    print("Dataset Info:", INFO[data_flag])
    
    # Get the dataset class
    DataClass = getattr(medmnist, INFO[data_flag]['python_class'])
    
    # Define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    
    # Load train and val sets with transform
    train_dataset = DataClass(split='train', transform=transform, download=download)
    val_dataset = DataClass(split='val', transform=transform, download=download)
    
    # Create data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)
    
    # Get a batch of data
    train_data, train_labels = next(iter(train_loader))
    
    print("\nData Analysis:")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"\nSample batch shapes:")
    print(f"Input shape: {train_data.shape}")
    print(f"Labels shape: {train_labels.shape}")
    print(f"\nLabel statistics:")
    print(f"Label dtype: {train_labels.dtype}")
    print(f"Sample labels (first 5 samples):")
    for i in range(min(5, len(train_labels))):
        print(f"Sample {i}:")
        print("  Raw label vector:", train_labels[i])
        print("  Active conditions:", [INFO[data_flag]['label'][str(j)] for j in range(len(train_labels[i])) if train_labels[i][j] == 1])
    
    # Analyze label distribution
    print("\nLabel distribution analysis:")
    label_sums = train_labels.sum(dim=0)
    total_samples = len(train_dataset)
    for i in range(len(label_sums)):
        condition = INFO[data_flag]['label'][str(i)]
        count = int(label_sums[i].item())
        percentage = (count / total_samples) * 100
        print(f"{condition}: {count} samples ({percentage:.1f}%)")
    
    # Calculate average conditions per image
    conditions_per_image = train_labels.sum(dim=1)
    avg_conditions = conditions_per_image.float().mean().item()
    max_conditions = conditions_per_image.max().item()
    print(f"\nAverage conditions per image: {avg_conditions:.2f}")
    print(f"Maximum conditions in a single image: {int(max_conditions)}")
    
    return train_dataset, val_dataset

if __name__ == "__main__":
    train_dataset, val_dataset = analyze_chestmnist() 