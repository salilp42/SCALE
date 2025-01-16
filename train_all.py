import os
import traceback
import json
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from interpretability import HierarchicalVizEngine, PublicationFigures, generate_publication_figures
from dataset import MedMNISTDataset
from model import MSMedVision
import medmnist
from medmnist import INFO
import gc
import multiprocessing as mp
from tqdm import tqdm
import argparse
import sys
import warnings
warnings.filterwarnings('ignore')

# Ensure using Agg backend for matplotlib in Colab
plt.switch_backend('Agg')

# Set plot style
plt.style.use('default')
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'figure.figsize': [10, 6],
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

class SubsetWithAttributes(Subset):
    """Subset class that preserves dataset attributes."""
    def __init__(self, dataset, indices, task_type, n_classes, class_weights=None):
        super().__init__(dataset, indices)
        self.task_type = task_type
        self.n_classes = n_classes
        self.class_weights = class_weights
        self.n_channels = dataset.n_channels if hasattr(dataset, 'n_channels') else 1
        self.labels = dataset.labels if hasattr(dataset, 'labels') else None  # Add labels attribute

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop

def calculate_confidence_interval(values, confidence=0.95):
    """Calculate confidence interval for a list of values."""
    try:
        values = np.array(values)
        mean = np.mean(values)
        std = np.std(values)
        z = stats.norm.ppf((1 + confidence) / 2)
        margin = z * std / np.sqrt(len(values))
        return {
            'mean': float(mean),
            'std': float(std),
            'ci_low': float(mean - margin),
            'ci_high': float(mean + margin)
        }
    except Exception as e:
        print(f"Warning: Error calculating confidence interval: {str(e)}")
        return {
            'mean': float(mean) if 'mean' in locals() else 0.0,
            'std': float(std) if 'std' in locals() else 0.0,
            'ci_low': 0.0,
            'ci_high': 0.0
        }

def plot_metrics_history(metrics_history, save_path):
    """Plot and save training metrics history."""
    try:
        plt.figure(figsize=(12, 8))
        
        # Plot loss
        plt.subplot(2, 1, 1)
        plt.plot(metrics_history['train_loss'], label='Train Loss')
        plt.plot(metrics_history['val_loss'], label='Val Loss')
        plt.title('Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot AUC and Accuracy
        plt.subplot(2, 1, 2)
        plt.plot(metrics_history['train_auc'], label='Train AUC')
        plt.plot(metrics_history['val_auc'], label='Val AUC')
        plt.plot(metrics_history['train_acc'], label='Train Acc')
        plt.plot(metrics_history['val_acc'], label='Val Acc')
        plt.title('AUC and Accuracy History')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        
        plt.tight_layout()
        
        # Create plots directory if it doesn't exist
        plots_dir = os.path.join(save_path, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Save in the plots directory
        plt.savefig(os.path.join(plots_dir, 'metrics_history.png'))
        plt.close()
    except Exception as e:
        print(f"Error plotting metrics history: {str(e)}")

def plot_confusion_matrix(y_true, y_pred, class_labels, save_dir, title=None):
    """Plot confusion matrix with proper class labels."""
    try:
        # Convert predictions to numpy if needed
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
            
        # Check if this is a multi-label case
        is_multilabel = (len(y_true.shape) > 1 and y_true.shape[1] > 1 and 
                        not np.all(np.sum(y_true, axis=1) == 1))
        
        if is_multilabel:
            # For multi-label, create a confusion matrix for each class
            n_classes = y_true.shape[1]
            fig, axes = plt.subplots(2, (n_classes + 1) // 2, 
                                   figsize=(15, 10))
            axes = axes.flatten()
            
            # Get label names from MedMNIST info dictionary
            if isinstance(class_labels, dict) and 'label' in class_labels:
                label_dict = class_labels['label']
                label_names = [str(label_dict.get(str(i), f'Class {i}')) for i in range(n_classes)]
            else:
                label_names = [f'Class {i}' for i in range(n_classes)]
            
            # Create a confusion matrix for each class
            for i in range(n_classes):
                cm = confusion_matrix(y_true[:, i], y_pred[:, i])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                          xticklabels=['Negative', 'Positive'],
                          yticklabels=['Negative', 'Positive'])
                axes[i].set_title(f'{label_names[i]}')
                
            # Remove extra subplots if odd number of classes
            if n_classes % 2:
                fig.delaxes(axes[-1])
                
            plt.tight_layout()
            
        else:
            # Handle prediction shape
            if len(y_pred.shape) > 1:
                if y_pred.shape[1] > 1:  # Multi-class case
                    y_pred = np.argmax(y_pred, axis=1)
                else:  # Binary case
                    y_pred = (y_pred > 0.5).astype(int).ravel()
                    
            # Handle true labels shape
            if len(y_true.shape) > 1:
                if y_true.shape[1] > 1:  # One-hot encoded
                    y_true = np.argmax(y_true, axis=1)
                else:  # Binary case
                    y_true = y_true.ravel()
            
            # Get unique classes from the data
            unique_classes = np.unique(np.concatenate([y_true, y_pred]))
            n_classes = len(unique_classes)
            
            # Get label names from MedMNIST info dictionary
            if isinstance(class_labels, dict) and 'label' in class_labels:
                label_dict = class_labels['label']
                label_names = [str(label_dict.get(str(i), f'Class {i}')) for i in range(n_classes)]
            else:
                label_names = [f'Class {i}' for i in range(n_classes)]
            
            # Compute confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Create figure with larger size for better label visibility
            plt.figure(figsize=(12, 10))
            
            # Create heatmap with proper labels and normalization
            sns.heatmap(
                cm, 
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=label_names,
                yticklabels=label_names,
                square=True,
                cbar_kws={'label': 'Count'}
            )
            
            # Customize labels and title
            plt.xlabel('Predicted Label', fontsize=12, labelpad=10)
            plt.ylabel('True Label', fontsize=12, labelpad=10)
            plt.title(title or 'Confusion Matrix', fontsize=14, pad=20)
            
            # Rotate labels for better readability
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            # Adjust layout
            plt.tight_layout()
        
        # Create plots directory if it doesn't exist
        plots_dir = os.path.join(save_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Save plot
        save_path = os.path.join(plots_dir, 'confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Generated confusion matrix")
        
    except Exception as e:
        print(f"Warning: Error in confusion matrix plotting: {str(e)}")
        traceback.print_exc()

def compute_metrics(scores, labels, task_type, dataset_name=''):
    """Compute accuracy and AUC metrics with special handling for ChestMNIST."""
    try:
        # Move tensors to CPU and convert to numpy
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        
        # Special handling for ChestMNIST (multi-label)
        if dataset_name.lower() == 'chestmnist':
            # Ensure scores and labels have the same shape
            if scores.shape != labels.shape:
                scores = scores.reshape(labels.shape)
            
            # Convert logits to probabilities using sigmoid
            probs = 1 / (1 + np.exp(-scores))
            preds = (probs > 0.5).astype(float)
            
            # Calculate accuracy (exact match)
            accuracy = np.mean(np.all(preds == labels, axis=1))
            
            # Calculate AUC for each class
            aucs = []
            n_classes = labels.shape[1]  # Number of classes (14 for ChestMNIST)
            
            for i in range(n_classes):
                # Only calculate AUC if both classes are present
                if len(np.unique(labels[:, i])) > 1:
                    try:
                        class_auc = roc_auc_score(labels[:, i], probs[:, i])
                        aucs.append(class_auc)
                    except ValueError as e:
                        print(f"Warning: Could not calculate AUC for class {i}: {str(e)}")
                        aucs.append(0.5)  # Default AUC for classes with insufficient data
            
            # Calculate mean AUC across all valid classes
            auc = np.mean(aucs) if aucs else 0.5
            
            # Calculate per-class accuracy
            per_class_acc = np.mean(preds == labels, axis=0)
            
            return {
                'accuracy': float(accuracy),
                'auc': float(auc),
                'per_class_auc': [float(x) for x in aucs],
                'per_class_acc': [float(x) for x in per_class_acc]
            }
        
        # Standard handling for other datasets
        else:
            if task_type == 'binary-class':
                probs = 1 / (1 + np.exp(-scores))  # sigmoid
                preds = (probs > 0.5).astype(float)
                
                # Ensure correct shape for binary classification
                if len(probs.shape) == 2 and probs.shape[1] == 1:
                    probs = probs.ravel()
                    preds = preds.ravel()
                    if len(labels.shape) == 2:
                        labels = labels.ravel()
            else:  # multi-class
                probs = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)  # softmax
                preds = np.argmax(probs, axis=1)
                if len(labels.shape) > 1:  # one-hot encoded
                    labels = np.argmax(labels, axis=1)
            
            # Calculate accuracy
            accuracy = np.mean((preds == labels).astype(float))
            
            # Calculate AUC
            try:
                if task_type == 'binary-class':
                    auc = roc_auc_score(labels, probs)
                else:  # multi-class
                    auc = roc_auc_score(labels, probs, multi_class='ovr')
            except ValueError as e:
                print(f"Warning: Could not calculate AUC for task type {task_type}: {str(e)}")
                auc = 0.5
            
            return {
                'accuracy': float(accuracy),
                'auc': float(auc)
            }
            
    except Exception as e:
        print(f"Error in {dataset_name} metric computation: {str(e)}")
        return {
            'accuracy': 0.0,
            'auc': 0.5
        }

def compute_auc(outputs, targets):
    """Compute AUC score."""
    metrics = compute_metrics(outputs, targets, 'multi-class')
    return metrics['auc']

def compute_acc(outputs, targets):
    """Compute accuracy score."""
    metrics = compute_metrics(outputs, targets, 'multi-class')
    return metrics['acc']

def generate_visualizations(model, dataset, dataset_name, results_dir):
    """Generate all visualizations in a consistent directory structure."""
    print(f"\nGenerating visualizations for {dataset_name}...")
    
    # All visualizations go in the plots subdirectory
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    try:
        # Get a batch of data
        device = next(model.parameters()).device
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        batch = next(iter(dataloader))
        inputs = batch[0].to(device)  # Move inputs to same device as model
        
        with torch.no_grad():
            # Run forward pass once to get all features
            _ = model(inputs)
            
            # 1. Generate hierarchical visualizations
            attention_maps, feature_correlations = model.get_attention_maps(inputs)
            viz_engine = HierarchicalVizEngine()
            viz_results = viz_engine.create_visualization(
                attention_maps,
                feature_correlations,
                inputs.cpu() if inputs.dim() == 4 else None,  # Move back to CPU for plotting
                inputs.cpu() if inputs.dim() == 5 else None
            )
            
            # Save hierarchical visualizations
            if viz_results:
                for name, fig in viz_results.items():
                    if fig is not None:
                        save_path = os.path.join(plots_dir, f'{name}.png')
                        fig.savefig(save_path, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                print("✓ Generated hierarchical visualizations")
            
            # 2. Generate feature response visualizations
            features = model.get_features(inputs)
            if '3d' in dataset_name.lower() and features.get('features_3d'):
                feature_tensor = features['features_3d'][-1][:1].detach().cpu()  # Move to CPU
                viz_engine.plot_3d_feature_responses(feature_tensor, plots_dir)
            elif features.get('features_2d'):
                feature_tensor = features['features_2d'][-1][:1].detach().cpu()  # Move to CPU
                viz_engine.plot_2d_feature_responses(feature_tensor, plots_dir)
            print("✓ Generated feature response visualizations")
            
            # 3. Generate cross-modal attention visualizations
            cross_modal_maps = model.get_cross_modal_maps(inputs)
            if cross_modal_maps:
                final_2d = cross_modal_maps['features'].get('final_2d')
                final_3d = cross_modal_maps['features'].get('final_3d')
                if final_2d is not None and final_3d is not None:
                    viz_engine.plot_cross_modal_attention(
                        final_2d.detach().cpu(),  # Move to CPU
                        final_3d.detach().cpu(),  # Move to CPU
                        fusion_module=model.fusion if hasattr(model, 'fusion') else None,
                        save_dir=plots_dir
                    )
                    print("✓ Generated cross-modal attention visualizations")
    
    except Exception as e:
        print(f"Error in visualization generation: {str(e)}")
        traceback.print_exc()
    finally:
        plt.close('all')
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"Visualizations saved to {plots_dir}")

def create_balanced_subset(dataset, labels, size):
    """Create a balanced subset ensuring minimum samples per class."""
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    # Ensure minimum 2 samples per class
    min_samples_per_class = 2
    required_size = n_classes * min_samples_per_class
    target_size = max(size, required_size)
    
    samples_per_class = target_size // n_classes
    subset_indices = []
    
    for label in unique_labels:
        label_indices = np.where(np.array(labels) == label)[0]
        if len(label_indices) < min_samples_per_class:
            print(f"Warning: Class {label} has fewer than {min_samples_per_class} samples")
            selected_indices = label_indices  # Take all available samples
        else:
            selected_indices = np.random.choice(
                label_indices,
                size=samples_per_class,
                replace=False
            )
        subset_indices.extend(selected_indices)
    
    return SubsetWithAttributes(
        dataset, subset_indices,
        dataset.task_type, dataset.n_classes,
        dataset.class_weights if hasattr(dataset, 'class_weights') else None
    )

def compute_confidence_intervals(metrics_history, confidence=0.95):
    """Compute confidence intervals for metrics."""
    ci_metrics = {}
    
    for metric in ['loss', 'auc', 'acc']:
        if f'val_{metric}' in metrics_history:
            values = metrics_history[f'val_{metric}']
            mean = np.mean(values)
            std = np.std(values)
            z = stats.norm.ppf((1 + confidence) / 2)
            margin = z * std / np.sqrt(len(values))
            
            ci_metrics[metric] = {
                'mean': mean,
                'std': std,
                'ci_low': mean - margin,
                'ci_high': mean + margin
            }
    
    return ci_metrics

class WarmupScheduler:
    """Warmup learning rate scheduler with minimal overhead."""
    def __init__(self, optimizer, warmup_epochs=3, warmup_factor=0.1, 
                 after_scheduler=None):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        self.after_scheduler = after_scheduler
        self.current_epoch = 0
        
        # Store initial learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self, epoch=None):
        if epoch is not None:
            self.current_epoch = epoch
            
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.current_epoch / self.warmup_epochs
            factor = self.warmup_factor * (1 - alpha) + alpha
            
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * factor
        elif self.after_scheduler is not None:
            self.after_scheduler.step()
            
        self.current_epoch += 1
        
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, 
                num_epochs, task_type, dataset_name='', early_stopping=True, use_warmup=True):
    """Train the model with optional warmup."""
    best_val_loss = float('inf')
    patience = 8 if '3d' in dataset_name.lower() else 5
    patience_counter = 0
    
    # Add warmup scheduler if requested
    if use_warmup and '3d' in dataset_name.lower():
        scheduler = WarmupScheduler(
            optimizer,
            warmup_epochs=3,
            warmup_factor=0.1,
            after_scheduler=scheduler
        )
    
    history = {
        'train_loss': [], 'train_acc': [], 'train_auc': [],
        'val_loss': [], 'val_acc': [], 'val_auc': [],
        'val_preds': None,
        'val_labels': None,
        'learning_rates': []  # Track learning rates
    }
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_scores = []
        train_labels = []
        
        train_pbar = tqdm(train_loader, desc='Training')
        for inputs, labels in train_pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Handle different task types
            if dataset_name.lower() == 'chestmnist':
                # ChestMNIST is multi-label, binary-class
                labels = labels.float()
                if len(outputs.shape) != len(labels.shape):
                    outputs = outputs.view(labels.shape)
            elif task_type == 'binary-class':
                labels = labels.float()
                if labels.ndim == 1:
                    labels = labels.unsqueeze(1)
                if outputs.shape[1] == 2:
                    outputs = outputs[:, 1].unsqueeze(1)
            else:
                if labels.dim() > 1 and labels.shape[1] > 1:
                    labels = torch.argmax(labels, dim=1)
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Apply gradient clipping for 3D datasets
            if '3d' in dataset_name.lower():
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Store metrics
            train_loss += loss.item() * inputs.size(0)
            train_scores.extend(outputs.detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_scores = []
        val_labels = []
        
        val_pbar = tqdm(val_loader, desc='Validation')
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                
                # Handle different task types
                if dataset_name.lower() == 'chestmnist':
                    # ChestMNIST is multi-label, binary-class
                    labels = labels.float()
                    if len(outputs.shape) != len(labels.shape):
                        outputs = outputs.view(labels.shape)
                elif task_type == 'binary-class':
                    labels = labels.float()
                    if labels.ndim == 1:
                        labels = labels.unsqueeze(1)
                    if outputs.shape[1] == 2:
                        outputs = outputs[:, 1].unsqueeze(1)
                else:
                    if labels.dim() > 1 and labels.shape[1] > 1:
                        labels = torch.argmax(labels, dim=1)
                
                # Compute loss
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                # Store predictions and labels
                val_scores.extend(outputs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_loader.dataset)
        
        # Store final epoch validation predictions and labels
        if epoch == num_epochs - 1 or (early_stopping and patience_counter >= patience):
            history['val_preds'] = np.array(val_scores)
            history['val_labels'] = np.array(val_labels)
        
        # Convert lists to numpy arrays for metric computation
        train_scores = np.array(train_scores)
        train_labels = np.array(train_labels)
        val_scores = np.array(val_scores)
        val_labels = np.array(val_labels)
        
        # Calculate epoch metrics
        train_metrics = compute_metrics(
            train_scores,
            train_labels,
            task_type,
            dataset_name
        )
        
        val_metrics = compute_metrics(
            val_scores,
            val_labels,
            task_type,
            dataset_name
        )
        
        # Update learning rate
        scheduler.step()
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_auc'].append(train_metrics['auc'])
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_auc'].append(val_metrics['auc'])
        
        # Print epoch results
        print(f'Train Loss: {train_loss:.4f} Acc: {train_metrics["accuracy"]:.4f} AUC: {train_metrics["auc"]:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_metrics["accuracy"]:.4f} AUC: {val_metrics["auc"]:.4f}')
        
        # Early stopping check
        if early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'\nEarly stopping triggered after {epoch + 1} epochs')
                    break
    
    return history, model

def plot_per_class_metrics(metrics, class_names, save_path):
    """Plot per-class metrics including AUC and accuracy."""
    try:
        n_classes = len(class_names)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot per-class AUC
        if 'per_class_auc' in metrics and metrics['per_class_auc']:
            aucs = metrics['per_class_auc']
            ax1.bar(range(len(aucs)), aucs)
            ax1.set_xticks(range(len(aucs)))
            ax1.set_xticklabels(class_names, rotation=45, ha='right')
            ax1.set_ylabel('AUC Score')
            ax1.set_title('Per-class AUC')
            ax1.grid(True, alpha=0.3)
        
        # Plot per-class accuracy
        if 'per_class_acc' in metrics and metrics['per_class_acc']:
            accs = metrics['per_class_acc']
            ax2.bar(range(len(accs)), accs)
            ax2.set_xticks(range(len(accs)))
            ax2.set_xticklabels(class_names, rotation=45, ha='right')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Per-class Accuracy')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'per_class_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Warning: Error plotting per-class metrics: {str(e)}")

def initialize_viz_results():
    """Initialize visualization results dictionary."""
    return {
        'scale_attention': False,
        'feature_responses': False,
        'cross_modal_attention': False,
        'hierarchical': False,
        'confusion_matrix': False,
        'roc_curves': False,
        'per_class_metrics': False
    }

def get_dataset_list(dataset_type=None):
    """Get list of datasets to process."""
    datasets_2d = [
        'chestmnist', 'pathmnist', 'dermamnist', 'octmnist', 
        'pneumoniamnist', 'retinamnist', 'breastmnist', 'bloodmnist',
        'tissuemnist', 'organamnist', 'organcmnist', 'organsmnist'
    ]
    
    datasets_3d = [
        'organmnist3d', 'nodulemnist3d', 'adrenalmnist3d',
        'fracturemnist3d', 'vesselmnist3d', 'synapsemnist3d'
    ]
    
    if dataset_type == '2d':
        return datasets_2d
    elif dataset_type == '3d':
        return datasets_3d
    else:
        return datasets_3d + datasets_2d

def get_dataset(dataset_name, split='train'):
    """Get dataset with proper initialization."""
    return MedMNISTDataset(dataset_name, split=split)

def create_model(in_channels, num_classes, is_3d=False):
    """Create model with proper initialization."""
    model = MSMedVision(
        in_channels=in_channels,
        num_classes=num_classes,
        is_3d=is_3d
    )
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    return model

def save_results(dataset_name, model, history):
    """Save model and training history."""
    results_dir = os.path.join('results', dataset_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(results_dir, 'model.pth'))
    
    # Save history
    with open(os.path.join(results_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=4)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--dataset-type', choices=['2d', '3d'], help='Type of datasets to process')
    return parser.parse_args()

def generate_publication_figures(results_dir):
    """Generate publication-quality figures from training results."""
    try:
        # Create paper_figures directory
        paper_figures_dir = os.path.join(results_dir, 'paper_figures')
        os.makedirs(paper_figures_dir, exist_ok=True)
        
        # Load results
        results_file = os.path.join(results_dir, 'all_results.json')
        if not os.path.exists(results_file):
            raise FileNotFoundError(f"Results file not found: {results_file}")
            
        with open(results_file, 'r') as f:
            all_results = json.load(f)
            
        # Separate 2D and 3D results
        results_2d = {k: v for k, v in all_results.items() if '3d' not in k.lower()}
        results_3d = {k: v for k, v in all_results.items() if '3d' in k.lower()}
        
        def create_performance_plot(results, title, is_3d=False):
            """Create performance plot with confidence intervals."""
            if not results:
                print(f"No {'3D' if is_3d else '2D'} results available for plotting")
                return None
                
            datasets = []
            metrics = {'acc': [], 'auc': []}
            ci_metrics = {'acc': {'lower': [], 'upper': []}, 
                         'auc': {'lower': [], 'upper': []}}
            
            for dataset, result in results.items():
                if result['status'] == 'success':
                    datasets.append(dataset.replace('mnist', '').replace('3d', ''))
                    
                    # Get metrics
                    metrics['acc'].append(result['final_val_acc'])
                    metrics['auc'].append(result['final_val_auc'])
                    
                    # Get confidence intervals
                    ci = result.get('confidence_intervals', {})
                    acc_ci = ci.get('acc', {})
                    auc_ci = ci.get('auc', {})
                    
                    # Handle accuracy CIs
                    if acc_ci and not np.isnan(acc_ci.get('ci_low', np.nan)):
                        ci_metrics['acc']['lower'].append(acc_ci['ci_low'])
                        ci_metrics['acc']['upper'].append(acc_ci['ci_high'])
                    else:
                        ci_metrics['acc']['lower'].append(metrics['acc'][-1])
                        ci_metrics['acc']['upper'].append(metrics['acc'][-1])
                    
                    # Handle AUC CIs
                    if auc_ci and not np.isnan(auc_ci.get('ci_low', np.nan)):
                        ci_metrics['auc']['lower'].append(auc_ci['ci_low'])
                        ci_metrics['auc']['upper'].append(auc_ci['ci_high'])
                    else:
                        ci_metrics['auc']['lower'].append(metrics['auc'][-1])
                        ci_metrics['auc']['upper'].append(metrics['auc'][-1])
            
            if not datasets:
                return None
                
            # Create figure
            plt.figure(figsize=(12, 6))
            x = np.arange(len(datasets))
            width = 0.35
            
            # Plot bars with error bars
            def plot_metric_bars(metric_name, offset, color):
                yerr = np.array([
                    np.array(metrics[metric_name]) - np.array(ci_metrics[metric_name]['lower']),
                    np.array(ci_metrics[metric_name]['upper']) - np.array(metrics[metric_name])
                ])
                plt.bar(x + offset, metrics[metric_name], width,
                       label=metric_name.upper(),
                       color=color, alpha=0.8,
                       yerr=yerr, capsize=5,
                       error_kw={'elinewidth': 1, 'capthick': 1})
            
            plot_metric_bars('acc', -width/2, '#3498db')
            plot_metric_bars('auc', width/2, '#2ecc71')
            
            # Customize plot
            plt.xlabel('Dataset')
            plt.ylabel('Score')
            plt.title(f'{title} ({"3D" if is_3d else "2D"} Datasets)')
            plt.xticks(x, datasets, rotation=45, ha='right')
            plt.legend(loc='lower right')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1.0)
            
            # Add value labels
            def add_value_labels(metric_name, offset):
                for i, v in enumerate(metrics[metric_name]):
                    plt.text(i + offset, v + 0.02,
                            f'{v:.2f}',
                            ha='center', va='bottom',
                            fontsize=8)
            
            add_value_labels('acc', -width/2)
            add_value_labels('auc', width/2)
            
            plt.tight_layout()
            return plt.gcf()
        
        # Generate and save 2D performance plot
        fig_2d = create_performance_plot(results_2d, 'Performance Metrics', is_3d=False)
        if fig_2d:
            fig_2d.savefig(os.path.join(paper_figures_dir, '2d_performance.png'),
                          dpi=300, bbox_inches='tight')
            plt.close(fig_2d)
        
        # Generate and save 3D performance plot
        fig_3d = create_performance_plot(results_3d, 'Performance Metrics', is_3d=True)
        if fig_3d:
            fig_3d.savefig(os.path.join(paper_figures_dir, '3d_performance.png'),
                          dpi=300, bbox_inches='tight')
            plt.close(fig_3d)
        
        # Generate combined performance comparison
        plt.figure(figsize=(15, 8))
        datasets = []
        val_accs = []
        val_aucs = []
        
        for dataset, result in all_results.items():
            if result['status'] == 'success':
                datasets.append(dataset.replace('mnist', '').replace('3d', ''))
                val_accs.append(result['final_val_acc'])
                val_aucs.append(result['final_val_auc'])
        
        x = np.arange(len(datasets))
        width = 0.35
        
        plt.bar(x - width/2, val_accs, width, label='Accuracy',
               color='#3498db', alpha=0.8)
        plt.bar(x + width/2, val_aucs, width, label='AUC',
               color='#2ecc71', alpha=0.8)
        
        plt.xlabel('Dataset')
        plt.ylabel('Score')
        plt.title('Performance Comparison Across All Datasets')
        plt.xticks(x, datasets, rotation=45, ha='right')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.0)
        
        # Add value labels
        for i, v in enumerate(val_accs):
            plt.text(i - width/2, v + 0.02, f'{v:.2f}',
                    ha='center', va='bottom', fontsize=8)
        for i, v in enumerate(val_aucs):
            plt.text(i + width/2, v + 0.02, f'{v:.2f}',
                    ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(paper_figures_dir, 'performance_comparison.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate per-dataset visualizations summary
        for dataset, result in all_results.items():
            if result['status'] == 'success':
                dataset_plots_dir = result.get('plots_dir')
                if dataset_plots_dir and os.path.exists(dataset_plots_dir):
                    # Copy key visualizations to paper_figures with dataset prefix
                    for plot_name in ['confusion_matrix.png', 'roc_curves.png', 'metrics_history.png']:
                        src_path = os.path.join(dataset_plots_dir, plot_name)
                        if os.path.exists(src_path):
                            dst_name = f"{dataset}_{plot_name}"
                            dst_path = os.path.join(paper_figures_dir, dst_name)
                            try:
                                import shutil
                                shutil.copy2(src_path, dst_path)
                            except Exception as e:
                                print(f"Warning: Could not copy {plot_name} for {dataset}: {str(e)}")
        
        print("Publication figures generated successfully in:", paper_figures_dir)
        print("\nGenerated figures:")
        print("1. 2D Performance Plot (2d_performance.png)")
        print("2. 3D Performance Plot (3d_performance.png)")
        print("3. Overall Performance Comparison (performance_comparison.png)")
        print("4. Per-dataset Confusion Matrices ({dataset}_confusion_matrix.png)")
        print("5. Per-dataset ROC Curves ({dataset}_roc_curves.png)")
        print("6. Per-dataset Training Metrics ({dataset}_metrics_history.png)")
        
    except Exception as e:
        print(f"Error generating publication figures: {str(e)}")
        traceback.print_exc()
    finally:
        plt.close('all')

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization.
    
    Parameters:
        obj: Object to convert, can be dict, list, numpy array, or numpy type
        
    Returns:
        Object with numpy types converted to Python native types
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in list(obj)]
    elif isinstance(obj, tuple):
        return [convert_numpy_types(item) for item in list(obj)]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    elif isinstance(obj, torch.Tensor):
        return convert_numpy_types(obj.cpu().detach().numpy())
    return obj

def main():
    args = parse_args()
    
    # Setup device and ensure CUDA is properly configured
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # Set memory growth
        for gpu in range(torch.cuda.device_count()):
            torch.cuda.set_per_process_memory_fraction(0.9, gpu)
    else:
        device = torch.device('cpu')
    print(f"\nUsing device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Create results directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_results_dir = os.path.join('results', f'all_datasets_{timestamp}')
    os.makedirs(base_results_dir, exist_ok=True)

    if args.debug:
        print("\nRunning in DEBUG mode with full datasets:")
        print("- Using full datasets (no subset)")
        print("- 3D datasets: 10 epochs")
        print("- 2D datasets: 5 epochs")
        print("- Batch size: 32")
        print("- Early stopping enabled")

    # Get list of datasets to process
    dataset_list = get_dataset_list(args.dataset_type)
    print("\nProcessing order:")
    print("3D datasets:", [d for d in dataset_list if '3d' in d])
    print("2D datasets:", [d for d in dataset_list if '3d' not in d])

    # Store all results
    all_results = {}

    # Process each dataset
    for dataset_name in dataset_list:
        print(f"\nProcessing {dataset_name}...")
        
        try:
            # Load dataset
            train_dataset = get_dataset(dataset_name, split='train')
            val_dataset = get_dataset(dataset_name, split='val')
            
            # Create data loaders
            batch_size = 16 if '3d' in dataset_name.lower() else args.batch_size
            if dataset_name.lower() in ['retinamnist', 'breastmnist']:
                batch_size = 24  # Adjusted batch size for better generalization
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True if torch.cuda.is_available() else False
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True if torch.cuda.is_available() else False
            )

            # Create model
            model = MSMedVision(train_dataset).to(device)

            # Set up training components based on task type
            criterion = nn.BCEWithLogitsLoss() if train_dataset.task_type == 'binary-class' else nn.CrossEntropyLoss()
            
            # Adjust learning rate based on dataset type
            lr = 0.0005 if '3d' in dataset_name.lower() else 0.001
            optimizer = optim.Adam(model.parameters(), lr=lr)
            
            # Adjust scheduler parameters for 3D datasets
            if '3d' in dataset_name.lower():
                scheduler = optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=7,
                    gamma=0.2
                )
            else:
                scheduler = optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=5,
                    gamma=0.1
                )

            print(f"\nTraining model on {dataset_name}...")
            num_epochs = 15 if '3d' in dataset_name.lower() else 5

            # Train model
            history, trained_model = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                num_epochs=num_epochs,
                task_type=train_dataset.task_type,
                dataset_name=dataset_name,
                early_stopping=True
            )

            # Save results and generate visualizations
            dataset_results_dir = os.path.join(base_results_dir, dataset_name)
            os.makedirs(dataset_results_dir, exist_ok=True)
            
            # Plot training metrics
            plot_metrics_history(history, dataset_results_dir)
            print("✓ Generated training metrics plot")
            
            # Plot confusion matrix using stored validation predictions
            if history['val_preds'] is not None and history['val_labels'] is not None:
                plot_confusion_matrix(
                    history['val_labels'],
                    history['val_preds'],
                    train_dataset.info,  # Pass the entire info dictionary instead of just the labels
                    dataset_results_dir,
                    title=f'{dataset_name} Confusion Matrix'
                )
            
            # Generate and save visualizations
            print(f"\nGenerating post-training visualizations for {dataset_name}...")
            generate_visualizations(trained_model, val_dataset, dataset_name, dataset_results_dir)
            
            # Store results
            all_results[dataset_name] = {
                'status': 'success',
                'metrics': history,
                'final_val_acc': history['val_acc'][-1],
                'final_val_auc': history['val_auc'][-1],
                'best_val_acc': max(history['val_acc']),
                'best_val_auc': max(history['val_auc']),
                'best_epoch': history['val_auc'].index(max(history['val_auc'])) + 1,
                'confidence_intervals': compute_confidence_intervals(history),
                'per_class_metrics': {
                    'auc': history.get('per_class_auc', []),
                    'acc': history.get('per_class_acc', [])
                },
                'plots_dir': os.path.join(dataset_results_dir, 'plots')
            }

            # Clear GPU memory after each dataset
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing {dataset_name}: {str(e)}")
            traceback.print_exc()
            all_results[dataset_name] = {
                'status': 'failed',
                'error': str(e)
            }
            continue
        finally:
            # Clean up
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save all results
    with open(os.path.join(base_results_dir, 'all_results.json'), 'w') as f:
        converted_results = convert_numpy_types(all_results)
        json.dump(converted_results, f, indent=4)

    # Generate publication figures using class-based implementation
    print("\nGenerating publication figures...")
    generate_publication_figures(base_results_dir)

    print("\nTraining complete! Check the results directory for outputs.")

if __name__ == '__main__':
    main() 