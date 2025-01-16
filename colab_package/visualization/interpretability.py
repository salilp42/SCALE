import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from typing import List, Optional, Tuple, Dict
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import pearsonr
from sklearn.metrics import mutual_info_score
import cv2
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from mpl_toolkits.axes_grid1 import ImageGrid
import traceback
import gc
from matplotlib.gridspec import GridSpec

class ScaleAwareAttention:
    def __init__(self):
        self.scales = ['cellular', 'tissue', 'organ']
        self.smoothing_factor = 0.5
        
    def compute_scale_attention_2d(self, features: torch.Tensor, scale: str) -> torch.Tensor:
        """Compute attention maps at different biological scales for 2D data."""
        B, C, H, W = features.shape
        
        # Scale-specific attention computation
        if scale == 'cellular':
            kernel_size = max(H // 16, 3)
        elif scale == 'tissue':
            kernel_size = max(H // 8, 5)
        else:  # organ
            kernel_size = max(H // 4, 7)
            
        # Compute attention weights
        attention = F.avg_pool2d(features, kernel_size, stride=1, padding=kernel_size//2)
        attention = F.softmax(attention.view(B, C, -1), dim=2).view(B, C, H, W)
        
        return attention

    def compute_scale_attention_3d(self, features: torch.Tensor, scale: str) -> torch.Tensor:
        """Compute attention maps at different biological scales for 3D data."""
        B, C, D, H, W = features.shape
        
        # Scale-specific attention computation
        if scale == 'cellular':
            kernel_size = max(H // 16, 3)
            depth_kernel = max(D // 16, 3)
        elif scale == 'tissue':
            kernel_size = max(H // 8, 5)
            depth_kernel = max(D // 8, 3)
        else:  # organ
            kernel_size = max(H // 4, 7)
            depth_kernel = max(D // 4, 3)
            
        # Compute 3D attention weights
        attention = F.avg_pool3d(features, 
                               kernel_size=(depth_kernel, kernel_size, kernel_size),
                               stride=1, 
                               padding=(depth_kernel//2, kernel_size//2, kernel_size//2))
        attention = F.softmax(attention.view(B, C, -1), dim=2).view(B, C, D, H, W)
        
        return attention
    
    def get_attention(self, features_2d: torch.Tensor, features_3d: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract multi-scale attention maps for both 2D and 3D features."""
        attention_maps = {}
        
        # Process 2D features
        for scale in self.scales:
            attention_maps[f'2d_{scale}'] = self.compute_scale_attention_2d(features_2d, scale)
        
        # Process 3D features
        if features_3d.dim() == 5:  # B, C, D, H, W
            for scale in self.scales:
                attention_3d = self.compute_scale_attention_3d(features_3d, scale)
                # Store both 3D attention and projected 2D attention
                attention_maps[f'3d_{scale}_full'] = attention_3d
                attention_maps[f'3d_{scale}'] = attention_3d.mean(dim=2)  # Project to 2D for visualization
        else:  # Already in 2D format
            for scale in self.scales:
                attention_maps[f'3d_{scale}'] = self.compute_scale_attention_2d(features_3d, scale)
        
        return attention_maps

    def compute_3d_attention(self, features, input_data):
        """Compute 3D attention maps."""
        # Handle input_data shape
        if input_data.dim() == 4:  # [B, C, H, W]
            B, C, H, W = input_data.shape
            D = H  # Assume cubic volume for 2D data
            input_data = input_data.unsqueeze(2).expand(-1, -1, D, -1, -1)
        else:  # [B, C, D, H, W]
            B, C, D, H, W = input_data.shape
        
        # Compute attention
        attention = torch.mean(features, dim=0)  # Average over channels
        attention = F.interpolate(attention.unsqueeze(0).unsqueeze(0), 
                                size=(D, H, W), 
                                mode='trilinear', 
                                align_corners=False)
        return attention.squeeze()
    
    def compute_2d_attention(self, features, input_data):
        """Compute 2D attention maps."""
        if input_data.dim() == 4:  # [B, C, H, W]
            B, C, H, W = input_data.shape
        else:  # [B, C, D, H, W]
            B, C, D, H, W = input_data.shape
            input_data = input_data[:, :, D//2]  # Take middle slice
        
        attention = torch.mean(features, dim=0)  # Average over channels
        attention = F.interpolate(attention.unsqueeze(0).unsqueeze(0),
                                size=(H, W),
                                mode='bilinear',
                                align_corners=False)
        return attention.squeeze()

class CrossModalFeatureMapper:
    def __init__(self):
        self.similarity_threshold = 0.5
        
    def compute_feature_similarity_2d(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """Compute similarity between 2D feature maps using cosine similarity."""
        feat1_flat = feat1.view(feat1.size(0), -1)
        feat2_flat = feat2.view(feat2.size(0), -1)
        
        # Normalize features
        feat1_norm = F.normalize(feat1_flat, p=2, dim=1)
        feat2_norm = F.normalize(feat2_flat, p=2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.mm(feat1_norm, feat2_norm.transpose(0, 1))
        
        return similarity

    def compute_feature_similarity_3d(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """Compute similarity between 3D feature maps using cosine similarity."""
        # Reshape 3D features to 2D for similarity computation
        B, C = feat1.shape[:2]
        feat1_flat = feat1.reshape(B, C, -1)  # B, C, D*H*W
        feat2_flat = feat2.reshape(B, C, -1)  # B, C, D*H*W
        
        # Normalize features
        feat1_norm = F.normalize(feat1_flat, p=2, dim=2)
        feat2_norm = F.normalize(feat2_flat, p=2, dim=2)
        
        # Compute similarity matrix for each batch
        similarity = torch.matmul(feat1_norm, feat2_norm.transpose(1, 2))  # B, C, C
        
        return similarity
    
    def map_relationships(self, attention_maps: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Map relationships between features across modalities and scales."""
        relationships = {}
        
        # Compute cross-modal relationships
        for scale in ['cellular', 'tissue', 'organ']:
            feat_2d = attention_maps[f'2d_{scale}']
            
            # Handle both 3D and projected 2D attention maps
            if f'3d_{scale}_full' in attention_maps:
                # 3D to 2D projection similarity
                feat_3d_proj = attention_maps[f'3d_{scale}']
                similarity_proj = self.compute_feature_similarity_2d(feat_2d, feat_3d_proj)
                relationships[f'cross_modal_{scale}_proj'] = similarity_proj
                
                # Full 3D similarity
                feat_3d_full = attention_maps[f'3d_{scale}_full']
                similarity_3d = self.compute_feature_similarity_3d(
                    feat_2d.unsqueeze(2).expand(-1, -1, feat_3d_full.size(2), -1, -1),
                    feat_3d_full
                )
                relationships[f'cross_modal_{scale}_3d'] = similarity_3d
            else:
                # Regular 2D similarity
                feat_3d = attention_maps[f'3d_{scale}']
                similarity = self.compute_feature_similarity_2d(feat_2d, feat_3d)
            relationships[f'cross_modal_{scale}'] = similarity
            
            # Compute scale transitions (e.g., cellular to tissue)
            if scale != 'organ':
                next_scale = 'tissue' if scale == 'cellular' else 'organ'
                # 2D scale transitions
                scale_transition_2d = self.compute_feature_similarity_2d(
                    attention_maps[f'2d_{scale}'],
                    attention_maps[f'2d_{next_scale}']
                )
                relationships[f'scale_transition_2d_{scale}_to_{next_scale}'] = scale_transition_2d
                
                # 3D scale transitions if available
                if f'3d_{scale}_full' in attention_maps:
                    scale_transition_3d = self.compute_feature_similarity_3d(
                        attention_maps[f'3d_{scale}_full'],
                        attention_maps[f'3d_{next_scale}_full']
                    )
                    relationships[f'scale_transition_3d_{scale}_to_{next_scale}'] = scale_transition_3d
        
        return relationships

class HierarchicalVizEngine:
    def __init__(self):
        self.cmap = plt.cm.viridis
        self.figsize = (15, 10)
        
    def create_attention_overlay_2d(self, image: torch.Tensor, attention: torch.Tensor) -> np.ndarray:
        """Create attention overlay on the original 2D image."""
        # Convert tensors to numpy arrays
        image_np = image.detach().cpu().numpy()
        attention_np = attention.detach().cpu().numpy()
        
        # Handle 4D tensors (B, C, H, W)
        if len(image_np.shape) == 4:
            image_np = image_np[0]  # Take first batch
        if len(attention_np.shape) == 4:
            attention_np = attention_np[0]  # Take first batch
            
        # Convert to grayscale if needed
        if image_np.shape[0] == 3:  # RGB
            image_np = np.transpose(image_np, (1, 2, 0))
            image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:  # Single channel
            image_gray = image_np[0]
            image_np = np.stack([image_gray] * 3, axis=-1)
            
        if attention_np.shape[0] > 1:  # Multiple channels
            attention_np = np.mean(attention_np, axis=0)
        else:
            attention_np = attention_np[0]
        
        # Ensure same dimensions
        if image_gray.shape != attention_np.shape:
            attention_np = cv2.resize(attention_np, (image_gray.shape[1], image_gray.shape[0]))
        
        # Normalize to [0, 1]
        image_gray = (image_gray - image_gray.min()) / (image_gray.max() - image_gray.min() + 1e-8)
        attention_np = (attention_np - attention_np.min()) / (attention_np.max() - attention_np.min() + 1e-8)
        
        # Convert to uint8
        image_gray = (image_gray * 255).astype(np.uint8)
        attention_np = (attention_np * 255).astype(np.uint8)
        
        # Create heatmap
        heatmap = cv2.applyColorMap(attention_np, cv2.COLORMAP_JET)
        
        # Overlay
        overlay = cv2.addWeighted(cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB), 0.7, heatmap, 0.3, 0)
        
        return overlay
    
    def create_attention_overlay_3d(self, volume: torch.Tensor, attention: torch.Tensor) -> Dict[str, np.ndarray]:
        """Create attention overlays for 3D volume data."""
        try:
            # Convert tensors to numpy arrays
            volume_np = volume.detach().cpu().numpy()
            attention_np = attention.detach().cpu().numpy()
            
            # Handle 5D tensors (B, C, D, H, W)
            if len(volume_np.shape) == 5:
                volume_np = volume_np[0]  # Take first batch
            if len(attention_np.shape) == 5:
                attention_np = attention_np[0]  # Take first batch
            
            # Get spatial dimensions
            d = min(volume_np.shape[-3], attention_np.shape[-3])  # Use minimum depth
            h = volume_np.shape[-2]
            w = volume_np.shape[-1]
            
            # Resize attention to match volume dimensions if needed
            if volume_np.shape[-3:] != attention_np.shape[-3:]:
                resized_attention = np.zeros((attention_np.shape[0], d, h, w))
                for c in range(attention_np.shape[0]):
                    for z in range(d):
                        resized_attention[c, z] = cv2.resize(attention_np[c, min(z, attention_np.shape[1]-1)], (w, h))
                attention_np = resized_attention
            
            # Get middle slices for each axis (safely)
            mid_d = d // 2
            mid_h = h // 2
            mid_w = w // 2
            
            slices = {
                'axial': {'vol': volume_np[:, mid_d, :, :], 'att': attention_np[:, mid_d, :, :]},
                'sagittal': {'vol': volume_np[:, :mid_d, :, mid_w], 'att': attention_np[:, :mid_d, :, mid_w]},
                'coronal': {'vol': volume_np[:, :mid_d, mid_h, :], 'att': attention_np[:, :mid_d, mid_h, :]}
            }
            
            overlays = {}
            for view, data in slices.items():
                # Create 2D overlay for each view
                overlays[view] = self.create_attention_overlay_2d(
                    torch.from_numpy(data['vol']),
                    torch.from_numpy(data['att'])
                )
            
            return overlays
        except Exception as e:
            print(f"Error in 3D attention overlay: {str(e)}")
            return {'axial': np.zeros((224, 224, 3), dtype=np.uint8)}  # Return black image on error
    
    def plot_feature_correlations(self, correlations: torch.Tensor, scale: str, is_3d: bool = False) -> plt.Figure:
        """Plot feature correlation matrix."""
        fig = plt.figure(figsize=(8, 8))
        correlations_np = correlations.detach().cpu().numpy()
        
        if is_3d:
            # Plot 3D correlations
            ax = fig.add_subplot(111, projection='3d')
            x, y, z = np.indices(correlations_np.shape)
            ax.scatter(x.flatten(), y.flatten(), z.flatten(), 
                      c=correlations_np.flatten(), cmap='viridis')
            ax.set_xlabel('Feature X')
            ax.set_ylabel('Feature Y')
            ax.set_zlabel('Feature Z')
        else:
            # Plot 2D correlations
            ax = fig.add_subplot(111)
            sns.heatmap(correlations_np, cmap='viridis', ax=ax)
            ax.set_xlabel('Feature Index')
            ax.set_ylabel('Feature Index')
        
        plt.title(f'Feature Correlations - {scale} scale')
        return fig
    
    def create_visualization(self, 
                           attention_maps: Dict[str, torch.Tensor],
                           feature_correlations: Dict[str, torch.Tensor],
                           original_image_2d: Optional[torch.Tensor] = None,
                           original_image_3d: Optional[torch.Tensor] = None) -> Dict[str, plt.Figure]:
        """Generate comprehensive visualization suite."""
        visualizations = {}
        
        # 1. Multi-scale Attention Visualization
        fig_attention = plt.figure(figsize=self.figsize)
        for i, scale in enumerate(['cellular', 'tissue', 'organ']):
            # 2D attention
            plt.subplot(2, 3, i + 1)
            if original_image_2d is not None:
                overlay = self.create_attention_overlay_2d(original_image_2d, attention_maps[f'2d_{scale}'])
                plt.imshow(overlay)
            else:
                plt.imshow(attention_maps[f'2d_{scale}'].mean(dim=1).cpu())
            plt.title(f'2D {scale} attention')
            
            # 3D attention
            plt.subplot(2, 3, i + 4)
            if original_image_3d is not None and f'3d_{scale}_full' in attention_maps:
                overlays = self.create_attention_overlay_3d(original_image_3d, attention_maps[f'3d_{scale}_full'])
                plt.imshow(overlays['axial'])  # Show axial view by default
                plt.title(f'3D {scale} attention (axial)')
            else:
                plt.imshow(attention_maps[f'3d_{scale}'].mean(dim=1).cpu())
                plt.title(f'3D {scale} attention (projected)')
        
        visualizations['attention_maps'] = fig_attention
        
        # 2. Feature Correlation Visualization
        for scale in ['cellular', 'tissue', 'organ']:
            # Handle both 2D and 3D correlations
            if f'cross_modal_{scale}_3d' in feature_correlations:
                fig_corr = self.plot_feature_correlations(
                    feature_correlations[f'cross_modal_{scale}_3d'],
                    scale,
                    is_3d=True
                )
                visualizations[f'correlation_{scale}_3d'] = fig_corr
                
                fig_corr_proj = self.plot_feature_correlations(
                    feature_correlations[f'cross_modal_{scale}_proj'],
                    f'{scale} (projected)'
                )
                visualizations[f'correlation_{scale}_proj'] = fig_corr_proj
            else:
                fig_corr = self.plot_feature_correlations(
                    feature_correlations[f'cross_modal_{scale}'],
                    scale
                )
                visualizations[f'correlation_{scale}'] = fig_corr
        
        # 3. Scale Transition Visualization
        fig_transition = plt.figure(figsize=(12, 4))
        for i, (start, end) in enumerate([('cellular', 'tissue'), ('tissue', 'organ')]):
            plt.subplot(1, 2, i + 1)
            if f'scale_transition_3d_{start}_to_{end}' in feature_correlations:
                # Show middle slice of 3D transition
                transition_3d = feature_correlations[f'scale_transition_3d_{start}_to_{end}']
                middle_slice = transition_3d.size(0) // 2
                sns.heatmap(transition_3d[middle_slice].cpu().numpy(), cmap='coolwarm')
                plt.title(f'3D Scale Transition: {start} → {end}')
            else:
                sns.heatmap(
                    feature_correlations[f'scale_transition_2d_{start}_to_{end}'].cpu().numpy(),
                    cmap='coolwarm'
                )
                plt.title(f'Scale Transition: {start} → {end}')
        
        visualizations['scale_transitions'] = fig_transition
        return visualizations

    def plot_3d_feature_responses(self, features: torch.Tensor, save_dir: str):
        """Plot 3D feature responses using volume rendering."""
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            # Handle input dimensions
            if features.dim() == 4:  # [B, C, H, W]
                features = features.unsqueeze(2)  # Add depth dimension
            elif features.dim() != 5:  # Not [B, C, D, H, W]
                raise ValueError(f"Expected 4D or 5D tensor, got {features.dim()}D")
            
            mean_features = features.mean(dim=1)  # Average across channels
            mean_features = mean_features.detach().cpu().numpy()
            
            for i in range(min(4, mean_features.shape[0])):  # Plot up to 4 samples
                volume = mean_features[i]
                
                # Create 3D figure with larger size for better visibility
                fig = plt.figure(figsize=(12, 12))
                ax = fig.add_subplot(111, projection='3d')
                
                # Create meshgrid for the volume
                x, y, z = np.meshgrid(
                    np.arange(volume.shape[0]),
                    np.arange(volume.shape[1]),
                    np.arange(volume.shape[2]),
                    indexing='ij'
                )
                
                # Normalize volume data
                volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
                
                # Use dynamic threshold based on data distribution
                threshold = np.percentile(volume, 70)  # Show top 30% activations
                mask = volume > threshold
                
                # Create scatter plot with enhanced visibility
                scatter = ax.scatter(
                    x[mask].flatten(),
                    y[mask].flatten(),
                    z[mask].flatten(),
                    c=volume[mask].flatten(),
                    cmap='hot',
                    alpha=0.6,
                    s=50  # Increased marker size
                )
                
                # Add colorbar and enhance its appearance
                cbar = plt.colorbar(scatter)
                cbar.set_label('Feature Response Intensity', fontsize=10)
                
                # Enhance 3D visualization
                ax.view_init(elev=20, azim=45)  # Set initial viewing angle
                ax.set_xlabel('Depth (D)', fontsize=10)
                ax.set_ylabel('Height (H)', fontsize=10)
                ax.set_zlabel('Width (W)', fontsize=10)
                ax.grid(True, alpha=0.3)
                
                ax.set_title(f'3D Feature Response - Sample {i+1}\nVolume Plot', fontsize=12, pad=20)
                
                # Save figure with high DPI
                plt.savefig(os.path.join(save_dir, f'3d_feature_response_{i+1}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            print(f"Error in plot_3d_feature_responses: {str(e)}")
            traceback.print_exc()
    
    def plot_scale_aware_attention(self, features: torch.Tensor, save_dir: str):
        """Plot scale-aware attention maps for 3D data."""
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            # Ensure input is 5D tensor [B, C, D, H, W]
            if features.dim() != 5:
                raise ValueError("Expected 5D tensor [B, C, D, H, W] for scale-aware attention")
            
            # Scale-specific parameters
            scale_params = {
                'cellular': {
                    'kernel': 3,
                    'cmap': 'magma',
                    'threshold': 75,  # Show top 25% attention for fine details
                    'alpha': 0.7
                },
                'tissue': {
                    'kernel': 5,
                    'cmap': 'plasma',
                    'threshold': 80,  # Show top 20% attention
                    'alpha': 0.6
                },
                'organ': {
                    'kernel': 7,
                    'cmap': 'inferno',
                    'threshold': 85,  # Show top 15% attention for broader patterns
                    'alpha': 0.5
                }
            }
            
            # Create figure with subplots for all scales
            fig = plt.figure(figsize=(20, 8))
            
            for idx, (scale, params) in enumerate(scale_params.items(), 1):
                # Compute attention for this scale
                attention = F.avg_pool3d(features, 
                                       kernel_size=params['kernel'],
                                       stride=1, 
                                       padding=params['kernel']//2)
                attention = F.softmax(attention.view(attention.size(0), -1), dim=1)
                attention = attention.view_as(features)
                
                # Take mean across channels and batch
                mean_attention = attention.mean(dim=(0,1)).detach().cpu().numpy()
                
                # Create subplot
                ax = fig.add_subplot(1, 3, idx, projection='3d')
                
                # Create volume plot
                x, y, z = np.meshgrid(np.arange(mean_attention.shape[0]),
                                    np.arange(mean_attention.shape[1]),
                                    np.arange(mean_attention.shape[2]))
                
                # Normalize attention values
                mean_attention = (mean_attention - mean_attention.min()) / (mean_attention.max() - mean_attention.min() + 1e-8)
                
                # Apply scale-specific threshold
                threshold = np.percentile(mean_attention, params['threshold'])
                mask = mean_attention > threshold
                
                # Create scatter plot with enhanced visibility
                scatter = ax.scatter(x[mask], y[mask], z[mask],
                                   c=mean_attention[mask],
                                   cmap=params['cmap'],
                                   alpha=params['alpha'],
                                   s=50)  # Increased marker size
                
                # Enhance visualization
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(f'{scale.capitalize()} Scale Attention', fontsize=10)
                
                ax.view_init(elev=20, azim=45)
                ax.set_xlabel('Depth (D)', fontsize=10)
                ax.set_ylabel('Height (H)', fontsize=10)
                ax.set_zlabel('Width (W)', fontsize=10)
                ax.grid(True, alpha=0.3)
                
                ax.set_title(f'{scale.capitalize()} Scale Attention\nKernel Size: {params["kernel"]}', 
                            fontsize=12, pad=20)
            
            plt.tight_layout(pad=3.0)
            plt.savefig(os.path.join(save_dir, 'scale_aware_attention_3d.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error in plot_scale_aware_attention: {str(e)}")
            traceback.print_exc()
    
    def plot_stream_attention(self, feat_2d, feat_3d, save_dir):
        """Plot enhanced stream attention with cross-modal interactions."""
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            # For 2D datasets
            if feat_3d.dim() == 4:  # [B, C, H, W]
                # Create figure with multiple visualizations
                fig = plt.figure(figsize=(20, 10))
                gs = plt.GridSpec(2, 2)
                
                try:
                    # 1. Channel Attention Map
                    ax1 = fig.add_subplot(gs[0, 0])
                    with torch.no_grad():
                        channel_attention = torch.matmul(
                            feat_2d.mean(dim=(2, 3)),  # [B, C]
                            feat_3d.mean(dim=(2, 3)).transpose(-1, -2)  # [B, C]
                        )
                        channel_attention = F.softmax(channel_attention, dim=-1)
                        attention_map = channel_attention[0].detach().cpu().numpy()
                        if attention_map.ndim == 1:
                            attention_map = attention_map.reshape(-1, 1)
                    
                    sns.heatmap(
                        attention_map,
                        cmap='viridis',
                        ax=ax1,
                        cbar_kws={'label': 'Channel Attention Strength'}
                    )
                    ax1.set_xlabel('Stream 2 Channels', fontsize=10)
                    ax1.set_ylabel('Stream 1 Channels', fontsize=10)
                    ax1.set_title('Channel-wise Cross-stream Attention', fontsize=12)
                except Exception as e:
                    print(f"Warning: Error in channel attention visualization: {str(e)}")
                    ax1.text(0.5, 0.5, 'Channel attention visualization failed', 
                            ha='center', va='center')
                
                try:
                    # 2. Spatial Attention Map
                    ax2 = fig.add_subplot(gs[0, 1])
                    with torch.no_grad():
                        spatial_attention = torch.einsum('bchw,bchw->bhw', feat_2d, feat_3d)
                        spatial_attention = F.softmax(spatial_attention.reshape(spatial_attention.size(0), -1), dim=-1)
                        spatial_attention = spatial_attention.reshape_as(spatial_attention)
                        spatial_map = spatial_attention[0].detach().cpu().numpy()
                    
                    im2 = ax2.imshow(spatial_map, cmap='hot')
                    plt.colorbar(im2, ax=ax2)
                    ax2.set_title('Spatial Cross-stream Attention', fontsize=12)
                except Exception as e:
                    print(f"Warning: Error in spatial attention visualization: {str(e)}")
                    ax2.text(0.5, 0.5, 'Spatial attention visualization failed', 
                            ha='center', va='center')
                
                try:
                    # 3. Feature Response Correlation
                    ax3 = fig.add_subplot(gs[1, 0])
                    
                    with torch.no_grad():
                        # Get features and project 3D to 2D if needed
                        feat_2d_flat = feat_2d[0].reshape(feat_2d.size(1), -1).detach()  # [C, H*W]
                        feat_3d_flat = feat_3d[0].reshape(feat_3d.size(1), -1).detach()  # [C, H*W]
                        
                        # Ensure same spatial dimensions by interpolation if needed
                        if feat_2d_flat.size(1) != feat_3d_flat.size(1):
                            target_size = min(feat_2d_flat.size(1), feat_3d_flat.size(1))
                            feat_2d_flat = F.interpolate(
                                feat_2d_flat.unsqueeze(0),
                                size=target_size,
                                mode='linear'
                            ).squeeze(0)
                            feat_3d_flat = F.interpolate(
                                feat_3d_flat.unsqueeze(0),
                                size=target_size,
                                mode='linear'
                            ).squeeze(0)
                        
                        # Normalize features
                        feat_2d_norm = F.normalize(feat_2d_flat, p=2, dim=1)
                        feat_3d_norm = F.normalize(feat_3d_flat, p=2, dim=1)
                        
                        # Compute correlation using cosine similarity
                        correlation = torch.mm(feat_2d_norm, feat_3d_norm.t()).cpu().numpy()
                    
                    sns.heatmap(correlation, cmap='RdBu_r', ax=ax3, center=0)
                    ax3.set_xlabel('Stream 2 Features', fontsize=10)
                    ax3.set_ylabel('Stream 1 Features', fontsize=10)
                    ax3.set_title('Feature Response Correlation', fontsize=12)
                except Exception as e:
                    print(f"Warning: Error in correlation visualization: {str(e)}")
                    ax3.text(0.5, 0.5, 'Correlation visualization failed', 
                            ha='center', va='center')
                
                try:
                    # 4. Combined Attention Visualization
                    ax4 = fig.add_subplot(gs[1, 1])
                    with torch.no_grad():
                        combined_attention = (
                            channel_attention[0].unsqueeze(-1).unsqueeze(-1) * 
                            spatial_attention[0].unsqueeze(0)
                        ).mean(0)
                        combined_map = combined_attention.detach().cpu().numpy()
                    
                    im4 = ax4.imshow(combined_map, cmap='viridis')
                    plt.colorbar(im4, ax=ax4)
                    ax4.set_title('Combined Channel-Spatial Attention', fontsize=12)
                except Exception as e:
                    print(f"Warning: Error in combined attention visualization: {str(e)}")
                    ax4.text(0.5, 0.5, 'Combined attention visualization failed', 
                            ha='center', va='center')
                
                plt.tight_layout(pad=3.0)
                plt.savefig(os.path.join(save_dir, 'stream_attention.png'), dpi=300, bbox_inches='tight')
                plt.close(fig)
            
            else:  # 3D dataset
                # Skip stream attention visualization for 3D datasets
                print("Skipping stream attention visualization for 3D dataset")
                return True
            
            # Simple cleanup without forced garbage collection
            plt.close('all')
            
            return True
            
        except Exception as e:
            print(f"Warning in stream attention visualization: {str(e)}")
            traceback.print_exc()
            
            # Simple cleanup on error
            plt.close('all')
            
            return False
    
    def plot_2d_feature_responses(self, features: torch.Tensor, save_dir: str):
        """Plot enhanced 2D feature responses with multi-scale visualization."""
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            # Ensure input is 4D [B, C, H, W]
            if features.dim() != 4:
                raise ValueError(f"Expected 4D tensor [B, C, H, W], got {features.dim()}D")
            
            # Create figure with multiple visualizations
            fig = plt.figure(figsize=(20, 10))
            gs = plt.GridSpec(2, 3)
            
            # 1. Channel-wise Feature Response
            ax1 = fig.add_subplot(gs[0, 0])
            mean_features = features.mean(dim=1)[0].detach().cpu().numpy()
            im1 = ax1.imshow(mean_features, cmap='viridis')
            plt.colorbar(im1, ax=ax1)
            ax1.set_title('Channel-wise Feature Response', fontsize=12)
            
            # 2. Top K Channel Responses
            ax2 = fig.add_subplot(gs[0, 1])
            channel_responses = features[0].detach().cpu()
            top_k = min(4, channel_responses.size(0))
            for i in range(top_k):
                response = channel_responses[i]
                response = (response - response.min()) / (response.max() - response.min() + 1e-8)
                ax2.imshow(response, cmap='viridis', alpha=0.5)
            ax2.set_title(f'Top {top_k} Channel Responses (Overlaid)', fontsize=12)
            
            # 3. Feature Activation Heatmap
            ax3 = fig.add_subplot(gs[0, 2])
            activation_map = torch.max(features[0], dim=0)[0].detach().cpu()
            activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min() + 1e-8)
            sns.heatmap(activation_map, cmap='hot', ax=ax3)
            ax3.set_title('Maximum Feature Activation', fontsize=12)
            
            # 4. Multi-scale Feature Response
            scales = [1, 2, 4]
            for i, scale in enumerate(scales):
                ax = fig.add_subplot(gs[1, i])
                pooled = F.avg_pool2d(features[0], kernel_size=scale, stride=1, padding=scale//2)
                pooled = pooled.mean(dim=0).detach().cpu()
                pooled = (pooled - pooled.min()) / (pooled.max() - pooled.min() + 1e-8)
                im = ax.imshow(pooled, cmap='viridis')
                plt.colorbar(im, ax=ax)
                ax.set_title(f'Scale {scale} Response', fontsize=12)
            
            plt.tight_layout(pad=3.0)
            plt.savefig(os.path.join(save_dir, '2d_feature_responses.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error in 2D feature response visualization: {str(e)}")
            traceback.print_exc()
    
    def plot_3d_feature_responses(self, features: torch.Tensor, save_dir: str):
        """Plot enhanced 3D feature responses with anatomical views."""
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            # Handle input dimensions
            if features.dim() == 4:  # [B, C, H, W]
                features = features.unsqueeze(2)  # Add depth dimension
            elif features.dim() != 5:  # Not [B, C, D, H, W]
                raise ValueError(f"Expected 4D or 5D tensor, got {features.dim()}D")
            
            # Average across channels
            mean_features = features.mean(dim=1)  # [B, D, H, W]
            mean_features = mean_features.detach().cpu().numpy()
            
            for i in range(min(4, mean_features.shape[0])):  # Plot up to 4 samples
                volume = mean_features[i]
                
                # Create figure with multiple views
                fig = plt.figure(figsize=(20, 15))
                gs = plt.GridSpec(2, 3)
                
                # 1. 3D Volume Rendering
                ax_vol = fig.add_subplot(gs[0, :], projection='3d')
                
                # Create meshgrid
                x, y, z = np.meshgrid(
                    np.arange(volume.shape[0]),
                    np.arange(volume.shape[1]),
                    np.arange(volume.shape[2]),
                    indexing='ij'
                )
                
                # Normalize volume data
                volume_norm = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
                
                # Use dynamic threshold based on data distribution
                threshold = np.percentile(volume_norm, 70)  # Show top 30% activations
                mask = volume_norm > threshold
                
                # Create scatter plot with enhanced visibility
                scatter = ax_vol.scatter(
                    x[mask].flatten(),
                    y[mask].flatten(),
                    z[mask].flatten(),
                    c=volume_norm[mask].flatten(),
                    cmap='hot',
                    alpha=0.6,
                    s=50  # Increased marker size
                )
                
                # Enhance 3D visualization
                cbar = plt.colorbar(scatter, ax=ax_vol)
                cbar.set_label('Feature Response Intensity', fontsize=10)
                ax_vol.view_init(elev=20, azim=45)
                ax_vol.set_xlabel('Depth (D)', fontsize=10)
                ax_vol.set_ylabel('Height (H)', fontsize=10)
                ax_vol.set_zlabel('Width (W)', fontsize=10)
                ax_vol.grid(True, alpha=0.3)
                ax_vol.set_title('3D Feature Response Volume', fontsize=12, pad=20)
                
                # 2. Anatomical Plane Views
                # Axial View (top-down)
                ax_axial = fig.add_subplot(gs[1, 0])
                mid_slice = volume.shape[0] // 2
                im_axial = ax_axial.imshow(volume[mid_slice], cmap='viridis')
                plt.colorbar(im_axial, ax=ax_axial)
                ax_axial.set_title(f'Axial View (z={mid_slice})', fontsize=12)
                
                # Sagittal View (side)
                ax_sagittal = fig.add_subplot(gs[1, 1])
                mid_slice = volume.shape[1] // 2
                im_sagittal = ax_sagittal.imshow(volume[:, mid_slice], cmap='viridis')
                plt.colorbar(im_sagittal, ax=ax_sagittal)
                ax_sagittal.set_title(f'Sagittal View (y={mid_slice})', fontsize=12)
                
                # Coronal View (front)
                ax_coronal = fig.add_subplot(gs[1, 2])
                mid_slice = volume.shape[2] // 2
                im_coronal = ax_coronal.imshow(volume[:, :, mid_slice], cmap='viridis')
                plt.colorbar(im_coronal, ax=ax_coronal)
                ax_coronal.set_title(f'Coronal View (x={mid_slice})', fontsize=12)
                
                plt.tight_layout(pad=3.0)
                plt.savefig(os.path.join(save_dir, f'3d_feature_response_{i+1}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            print(f"Error in 3D feature response visualization: {str(e)}")
            traceback.print_exc()

    def plot_cross_modal_attention(self, feat_2d, feat_3d, fusion_module, save_dir):
        """Plot cross-modal attention maps between 2D and 3D streams."""
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            with torch.no_grad():
                # Get attention weights
                b, c, h, w = feat_2d.shape
                feat_2d_flat = feat_2d.view(b, c, -1)
                
                if feat_3d.dim() == 5:  # [B, C, D, H, W]
                    d = feat_3d.size(2)
                    feat_3d_flat = feat_3d.view(b, c, -1)
                    
                    # Create 3D attention visualization
                    attn_3d = feat_3d_flat.view(b, c, d, h, w)[0].mean(0)
                    save_path = os.path.join(save_dir, 'cross_modal_attention_3d.png')
                    self.create_3d_visualization(
                        attn_3d.cpu().numpy(),
                        'Cross-Modal 3D Attention',
                        save_path
                    )
                
                # Create 2D attention visualization
                attn_2d = feat_2d_flat.view(b, c, h, w)[0].mean(0)
                
                # Ensure attention map has proper dimensions
                if attn_2d.shape != (h, w):
                    attn_2d = F.interpolate(
                        attn_2d.unsqueeze(0).unsqueeze(0),
                        size=(h, w),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()
                
                plt.figure(figsize=(8, 8))
                plt.imshow(attn_2d.cpu().numpy(), cmap='viridis')
                plt.title('Cross-Modal 2D Attention')
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, 'cross_modal_attention_2d.png'), dpi=300)
                plt.close()
                
                # Plot attention correlation if available
                if hasattr(fusion_module, 'attn_weights'):
                    attn = fusion_module.attn_weights[0].cpu().numpy()
                    
                    # Ensure attention weights have proper dimensions
                    if len(attn.shape) > 2:
                        attn = attn.mean(axis=0)  # Average over extra dimensions
                    
                    plt.figure(figsize=(10, 10))
                    plt.imshow(attn, cmap='viridis')
                    plt.title('Cross-Modal Attention Weights')
                    plt.colorbar()
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, 'cross_modal_attention_weights.png'), dpi=300)
                    plt.close()
                
            return True
        except Exception as e:
            print(f"Error in cross-modal attention visualization: {str(e)}")
            return False
    
    def create_hierarchical_visualization(self, data, features_2d, features_3d):
        """Create hierarchical visualization of features."""
        try:
            visualizations = {}
            max_subplots = 6  # Maximum number of subplots (2x3 grid)
            
            # 1. Original input visualization
            if data.dim() == 5:  # 3D input
                visualizations['input'] = self.create_attention_overlay_3d(
                    data, torch.ones_like(data))['axial']
            else:  # 2D input
                visualizations['input'] = self.create_attention_overlay_2d(
                    data, torch.ones_like(data))
            
            # 2. Feature hierarchy for 2D stream (limit to 2 levels)
            for i, feat in enumerate(features_2d[:2]):
                feat_viz = self.create_attention_overlay_2d(
                    data if data.dim() == 4 else data[:, :, data.size(2)//2],
                    feat.mean(dim=1, keepdim=True))
                visualizations[f'2d_level_{i+1}'] = feat_viz
            
            # 3. Feature hierarchy for 3D stream (limit to 2 levels)
            for i, feat in enumerate(features_3d[:2]):
                if feat.dim() == 5:
                    feat_viz = self.create_attention_overlay_3d(
                        data, feat.mean(dim=1, keepdim=True))['axial']
                else:
                    feat_viz = self.create_attention_overlay_2d(
                        data if data.dim() == 4 else data[:, :, data.size(2)//2],
                        feat.mean(dim=1, keepdim=True))
                visualizations[f'3d_level_{i+1}'] = feat_viz
            
            # Ensure we don't exceed max subplots
            if len(visualizations) > max_subplots:
                # Keep only the first max_subplots visualizations
                visualizations = dict(list(visualizations.items())[:max_subplots])
            
            return visualizations
        except Exception as e:
            print(f"Error in hierarchical visualization: {str(e)}")
            return {'error': np.zeros((224, 224, 3), dtype=np.uint8)}
            
    def visualize_feature_importance(self, feat_2d, feat_3d=None, attention_weights=None):
        """Visualize feature importance and relationships."""
        try:
            visualizations = {}
            
            # 1. Feature activation patterns
            feat_2d_act = feat_2d.mean(dim=0).abs()
            feat_3d_act = feat_3d.mean(dim=0).abs()
            
            visualizations['2d_importance'] = feat_2d_act.detach().cpu().numpy()
            visualizations['3d_importance'] = feat_3d_act.detach().cpu().numpy()
            
            # 2. Attention-weighted importance if available
            if attention_weights is not None:
                attn_2d = attention_weights[:feat_2d.size(-1)].mean(0)
                attn_3d = attention_weights[feat_2d.size(-1):].mean(0)
                
                visualizations['attention_2d'] = attn_2d.detach().cpu().numpy()
                visualizations['attention_3d'] = attn_3d.detach().cpu().numpy()
            
            return visualizations
        except Exception as e:
            print(f"Error in feature importance visualization: {str(e)}")
            return {'error': np.zeros((224, 224), dtype=np.float32)}
    
class PublicationFigures:
    def __init__(self):
        """Initialize publication figure generator with consistent styling."""
        plt.style.use('default')
        self.figsize = (15, 10)
        self.dpi = 300
        self.font = {'family': 'Arial', 'size': 10}
        plt.rc('font', **self.font)
        self.subplot_labels = list('abcdefghijklmnopqrstuvwxyz')
        self.colors = {'auc': '#2ecc71', 'acc': '#3498db'}
        
    def _add_subplot_label(self, ax, label_idx):
        """Add subplot label (a, b, c, etc.) to the axis."""
        if isinstance(ax, Axes3D):
            ax.text2D(-0.1, 1.1, f'({self.subplot_labels[label_idx]})', 
                    transform=ax.transAxes, fontsize=12, fontweight='bold')
        else:
            ax.text(-0.1, 1.1, f'({self.subplot_labels[label_idx]})',
                    transform=ax.transAxes, fontsize=12, fontweight='bold')
        
    def plot_3d_volume(self, ax, volume_data, title=None, threshold=None):
        """Plot 3D volume data with optional thresholding."""
        if torch.is_tensor(volume_data):
            volume_data = volume_data.detach().cpu().numpy()
        
        if threshold is None:
            threshold = np.percentile(volume_data, 75)
            
        x, y, z = np.where(volume_data > threshold)
        values = volume_data[x, y, z]
        
        # Create scatter plot with depth-dependent size and alpha
        sizes = np.interp(z, (z.min(), z.max()), (20, 100))
        alphas = np.interp(z, (z.min(), z.max()), (0.3, 0.8))
        
        scatter = ax.scatter(
            x[mask], y[mask], z[mask],
            c=values,
            cmap='viridis',
            s=sizes,
            alpha=alphas,
            edgecolors='white',
            linewidth=0.5
        )
        
        # Enhance depth perception
        ax.xaxis.set_pane_color((0.8, 0.8, 0.8, 0.2))
        ax.yaxis.set_pane_color((0.8, 0.8, 0.8, 0.2))
        ax.zaxis.set_pane_color((0.8, 0.8, 0.8, 0.2))
        
        # Add gridlines for better spatial understanding
        ax.grid(True, linestyle='--', alpha=0.3)
        
        if title:
            ax.set_title(title, pad=10)
            
        return scatter

    def plot_3d_attention(self, ax, attention_map, title=None):
        """Plot 3D attention map as a scatter plot."""
        return self.plot_3d_volume(ax, attention_map, title, threshold=attention_map.mean())
        
    def generate_performance_summary(self, results, is_3d=False):
        """Generate performance summary figure with AUC and accuracy in the same plot."""
        # Filter datasets based on dimensionality
        filtered_results = {k: v for k, v in results.items() 
                          if ('3d' in k.lower()) == is_3d 
                          if v['status'] == 'success'}  # Only include successful results
        
        if not filtered_results:
            return None
        
        # Create figure with better dimensions for publication
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)
        
        # Prepare data
        datasets = list(filtered_results.keys())
        metrics = [(d, 
                   filtered_results[d]['final_val_auc'],
                   filtered_results[d]['final_val_acc'],
                   filtered_results[d].get('ci_metrics', {}).get('auc', {}).get('ci_low', None),
                   filtered_results[d].get('ci_metrics', {}).get('auc', {}).get('ci_high', None),
                   filtered_results[d].get('ci_metrics', {}).get('acc', {}).get('ci_low', None),
                   filtered_results[d].get('ci_metrics', {}).get('acc', {}).get('ci_high', None)
                  ) for d in datasets]
        metrics.sort(key=lambda x: x[1], reverse=True)  # Sort by AUC
        
        datasets = [m[0] for m in metrics]
        aucs = [m[1] for m in metrics]
        accs = [m[2] / 100.0 for m in metrics]  # Convert accuracy to same scale as AUC
        
        # Calculate error bars (ensure non-negative values)
        def calculate_error_bars(values, ci_low, ci_high):
            """Calculate error bars ensuring non-negative values and proper scaling."""
            lower_errors = []
            upper_errors = []
            for val, low, high in zip(values, ci_low, ci_high):
                if np.isnan(val) or val is None:
                    lower_errors.append(0)
                    upper_errors.append(0)
                    continue
                    
                # Handle cases where confidence intervals are missing or invalid
                if low is None or np.isnan(low) or high is None or np.isnan(high):
                    # Use a small default error (1% of the value)
                    err = max(0.01 * abs(val), 0.01)
                    lower_errors.append(err)
                    upper_errors.append(err)
                else:
                    # Ensure errors are non-negative and properly bounded
                    lower_err = max(0, val - low)
                    upper_err = max(0, high - val)
                    
                    # Cap errors at the valid range (0-1 for both AUC and normalized accuracy)
                    lower_errors.append(min(lower_err, val))
                    upper_errors.append(min(upper_err, 1.0 - val))
            
            return np.array([lower_errors, upper_errors])
        
        # Calculate error bars with the improved function
        auc_yerr = calculate_error_bars(
            aucs,
            [m[3] if m[3] is not None else m[1] for m in metrics],
            [m[4] if m[4] is not None else m[1] for m in metrics]
        )
        
        acc_yerr = calculate_error_bars(
            accs,
            [m[5]/100.0 if m[5] is not None else m[2]/100.0 for m in metrics],
            [m[6]/100.0 if m[6] is not None else m[2]/100.0 for m in metrics]
        )
        
        # Create figure with improved error bar handling
        x = np.arange(len(datasets))
        width = 0.35
        
        # Create bars with refined styling and error bars
        rects1 = ax.bar(x - width/2, aucs, width, label='AUC',
                       color='#4C72B0', edgecolor='black',
                       linewidth=0.5, alpha=0.8,
                       yerr=auc_yerr, capsize=3,
                       error_kw={'elinewidth': 0.5, 'capthick': 0.5})
        rects2 = ax.bar(x + width/2, accs, width, label='Accuracy',
                       color='#55A868', edgecolor='black',
                       linewidth=0.5, alpha=0.8,
                       yerr=acc_yerr, capsize=3, error_kw={'elinewidth': 0.5, 'capthick': 0.5})
        
        # Customize plot for publication
        ax.set_ylabel('Score', fontsize=12, fontfamily='Arial')
        ax.set_title(f"Performance Metrics for {'3D' if is_3d else '2D'} Datasets",
                    fontsize=14, fontfamily='Arial', pad=20)
        
        # Customize x-axis
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=45, ha='right',
                          fontsize=10, fontfamily='Arial')
        
        # Remove gridlines for cleaner look
        ax.grid(False)
        
        # Customize spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        
        # Customize legend
        ax.legend(loc='upper right', fontsize=10, frameon=True,
                 fancybox=True, edgecolor='black',
                 facecolor='white', framealpha=1.0)
        
        # Set y-axis limits and ticks
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        return fig
        
    def generate_feature_response_figure(self, datasets_features):
        """Enhanced feature response visualization for publication"""
        n_datasets = len(datasets_features)
        n_cols = min(3, n_datasets)
        n_rows = (n_datasets + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(6*n_cols, 8*n_rows), dpi=self.dpi)
        
        for idx, (dataset_name, features) in enumerate(datasets_features.items()):
            is_3d = '3d' in dataset_name.lower()
            
            if is_3d:
                # Create subplot grid for 3D visualization
                gs = plt.GridSpec(2, 2, height_ratios=[1.5, 1])
                
                # 3D volume visualization
                ax_vol = fig.add_subplot(gs[0, :], projection='3d')
                feat_3d = features['3d'][-1].mean(1)[0].detach()  # Detach before visualization
                self.plot_3d_volume(ax_vol, feat_3d, dataset_name)
                
                # Anatomical plane views
                planes = ['axial', 'sagittal']
                for i, plane in enumerate(planes):
                    ax_plane = fig.add_subplot(gs[1, i])
                    if plane == 'axial':
                        slice_idx = feat_3d.shape[0] // 2
                        plt.imshow(feat_3d[slice_idx].cpu().numpy(), cmap='viridis')
                    else:
                        slice_idx = feat_3d.shape[1] // 2
                        plt.imshow(feat_3d[:, slice_idx].cpu().numpy(), cmap='viridis')
                    plt.title(f'{plane.capitalize()} View')
                    plt.colorbar()
            else:
                # Original 2D visualization
                ax = fig.add_subplot(n_rows, n_cols, idx + 1)
                feat_2d = features['2d'][-1].mean(1)[0].detach()  # Detach before visualization
                im = ax.imshow(feat_2d.cpu().numpy(), cmap='viridis')
                plt.colorbar(im, ax=ax)
                ax.set_title(dataset_name)
        
        plt.tight_layout()
        return fig
        
    def generate_attention_figure(self, datasets_attention):
        """Enhanced attention visualization for publication"""
        n_datasets = len(datasets_attention)
        n_cols = min(3, n_datasets)
        n_rows = (n_datasets + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(6*n_cols, 8*n_rows), dpi=self.dpi)
        
        for idx, (dataset_name, attention) in enumerate(datasets_attention.items()):
            is_3d = '3d' in dataset_name.lower()
            
            if is_3d:
                # Create subplot grid for 3D attention
                gs = plt.GridSpec(2, 3)
                
                # 3D attention volume
                ax_vol = fig.add_subplot(gs[0, :], projection='3d')
                att_3d = attention['3d'].mean(1)[0].detach()  # Detach before visualization
                self.plot_3d_volume(ax_vol, att_3d, f'{dataset_name} Attention')
                
                # Anatomical attention maps
                planes = ['axial', 'sagittal', 'coronal']
                for i, plane in enumerate(planes):
                    ax_plane = fig.add_subplot(gs[1, i])
                    if plane == 'axial':
                        slice_idx = att_3d.shape[0] // 2
                        plt.imshow(att_3d[slice_idx].cpu().numpy(), cmap='hot')
                    elif plane == 'sagittal':
                        slice_idx = att_3d.shape[1] // 2
                        plt.imshow(att_3d[:, slice_idx].cpu().numpy(), cmap='hot')
                    else:
                        slice_idx = att_3d.shape[2] // 2
                        plt.imshow(att_3d[:, :, slice_idx].cpu().numpy(), cmap='hot')
                    plt.title(f'{plane.capitalize()} Attention')
                    plt.colorbar()
            else:
                # Original 2D attention visualization
                ax = fig.add_subplot(n_rows, n_cols, idx + 1)
                att_2d = attention['2d'].mean(1)[0].detach()  # Detach before visualization
                im = ax.imshow(att_2d.cpu().numpy(), cmap='hot')
                plt.colorbar(im, ax=ax)
                ax.set_title(dataset_name)
        
        plt.tight_layout()
        return fig
        
    def generate_cross_modal_figure(self, datasets_cross_modal):
        """Generate cross-modal attention figure with subplots for each dataset."""
        n_datasets = len(datasets_cross_modal)
        n_cols = min(3, n_datasets)
        n_rows = (n_datasets + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(6*n_cols, 6*n_rows), dpi=self.dpi)
        
        for idx, (dataset_name, cross_modal) in enumerate(datasets_cross_modal.items()):
            is_3d = '3d' in dataset_name.lower()
            ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d' if is_3d else None)
            self._add_subplot_label(ax, idx)
            
            if is_3d:
                # Reshape cross-modal attention for 3D visualization
                attention_3d = cross_modal['combined']
                if torch.is_tensor(attention_3d):
                    attention_3d = attention_3d.detach()  # Detach before any processing
                    if attention_3d.dim() > 3:
                        attention_3d = attention_3d[0]  # Take first batch
                    
                    # Reshape to 3D volume if needed
                    if attention_3d.dim() == 2:
                        size = int(np.cbrt(attention_3d.shape[-1]))
                        attention_3d = attention_3d.reshape(size, size, size)
                    elif attention_3d.dim() > 3:
                        attention_3d = attention_3d.mean(dim=0)  # Average over heads
                
                # Use a higher threshold for cross-modal attention
                attention_3d_np = attention_3d.cpu().numpy() if torch.is_tensor(attention_3d) else attention_3d
                threshold = np.percentile(attention_3d_np, 90)
                self.plot_3d_volume(ax, attention_3d, dataset_name, threshold=threshold)
            else:
                # Handle 2D cross-modal attention
                attention_2d = cross_modal['combined']
                if torch.is_tensor(attention_2d):
                    attention_2d = attention_2d.detach()  # Detach before any processing
                    if attention_2d.dim() > 3:
                        attention_2d = attention_2d[0]  # Take first batch
                    if attention_2d.dim() > 2:
                        attention_2d = attention_2d.mean(dim=0)  # Average over heads
                    attention_2d = attention_2d.cpu().numpy()
                
                im = ax.imshow(attention_2d, cmap='viridis')
                plt.colorbar(im, ax=ax)
                ax.set_title(dataset_name)
        
        plt.tight_layout()
        return fig

    def generate_combined_attention_figure(self, dataset_name, attention_maps, save_dir):
        """Generate combined attention visualization figure."""
        is_3d = '3d' in dataset_name.lower()
        
        if is_3d:
            # Create figure with subplots for 3D data
            fig = plt.figure(figsize=(15, 10))
            gs = GridSpec(2, 3)
            
            # 3D attention volume
            ax_vol = fig.add_subplot(gs[0, :], projection='3d')
            att_3d = attention_maps['3d_attention'].mean(1)[0].detach()
            self.plot_3d_volume(ax_vol, att_3d, f'{dataset_name} 3D Attention')
            
            # Cross-modal attention
            ax_cross = fig.add_subplot(gs[1, 0])
            cross_att = attention_maps['cross_modal'].mean(0).detach()
            im = ax_cross.imshow(cross_att.cpu().numpy(), cmap='viridis')
            plt.colorbar(im, ax=ax_cross)
            ax_cross.set_title('Cross-Modal Attention')
            
            # Scale attention
            ax_scale = fig.add_subplot(gs[1, 1])
            scale_att = attention_maps['scale_attention'].mean(1)[0].detach()
            im = ax_scale.imshow(scale_att.cpu().numpy(), cmap='viridis')
            plt.colorbar(im, ax=ax_scale)
            ax_scale.set_title('Scale-Level Attention')
            
            # Feature response
            ax_feat = fig.add_subplot(gs[1, 2])
            feat_resp = attention_maps['feature_response'].mean(1)[0].detach()
            im = ax_feat.imshow(feat_resp.cpu().numpy(), cmap='viridis')
            plt.colorbar(im, ax=ax_feat)
            ax_feat.set_title('Feature Response')
            
        else:
            # Create figure with subplots for 2D data
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            axes = axes.ravel()
            
            # Original attention
            att_2d = attention_maps['2d_attention'].mean(1)[0].detach()
            im = axes[0].imshow(att_2d.cpu().numpy(), cmap='viridis')
            plt.colorbar(im, ax=axes[0])
            axes[0].set_title('2D Attention')
            
            # Cross-modal attention
            cross_att = attention_maps['cross_modal'].mean(0).detach()
            im = axes[1].imshow(cross_att.cpu().numpy(), cmap='viridis')
            plt.colorbar(im, ax=axes[1])
            axes[1].set_title('Cross-Modal Attention')
            
            # Feature response
            feat_resp = attention_maps['feature_response'].mean(1)[0].detach()
            im = axes[2].imshow(feat_resp.cpu().numpy(), cmap='viridis')
            plt.colorbar(im, ax=axes[2])
            axes[2].set_title('Feature Response')
            
            # Combined attention
            combined_att = (att_2d * cross_att.mean(0)).detach()
            im = axes[3].imshow(combined_att.cpu().numpy(), cmap='viridis')
            plt.colorbar(im, ax=axes[3])
            axes[3].set_title('Combined Attention')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{dataset_name}_combined_attention.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def generate_publication_figures(self, results_dir):
        """Generate all publication figures."""
        paper_figures_dir = os.path.join(results_dir, 'paper_figures')
        os.makedirs(paper_figures_dir, exist_ok=True)
        
        try:
            # Load results
            with open(os.path.join(results_dir, 'all_results.json'), 'r') as f:
                all_results = json.load(f)
            
            # Generate performance plots
            self._generate_performance_plots(all_results, paper_figures_dir)
            
            # Generate combined attention visualizations
            for dataset_name, result in all_results.items():
                if result['status'] == 'success' and 'attention_maps' in result:
                    self.generate_combined_attention_figure(
                        dataset_name,
                        result['attention_maps'],
                        paper_figures_dir
                    )
            
            print("Publication figures generated successfully.")
            
        except Exception as e:
            print(f"Error generating publication figures: {str(e)}")
            traceback.print_exc()
        finally:
            plt.close('all')

class AnatomicalVizEngine3D:
    """Enhanced visualization engine for 3D anatomical interpretability"""
    def __init__(self):
        self.cmap = plt.cm.viridis
        self.figsize = (15, 10)
        
    def plot_anatomical_attention(self, attention_maps, features, slice_idx=None):
        """Plot anatomical attention maps with plane-specific views"""
        fig = plt.figure(figsize=(15, 12))
        gs = plt.GridSpec(3, 4)
        
        # Get middle slice if not specified
        if slice_idx is None:
            slice_idx = {
                'axial': features.shape[2] // 2,
                'sagittal': features.shape[3] // 2,
                'coronal': features.shape[4] // 2
            }
        
        # Plot original feature volume
        ax_vol = fig.add_subplot(gs[0, 0], projection='3d')
        self.plot_3d_volume(ax_vol, features.mean(1)[0], 'Feature Volume')
        
        # Plot anatomical plane attention
        planes = ['axial', 'sagittal', 'coronal']
        slices = {
            'axial': lambda x: x[:, :, slice_idx['axial'], :, :],
            'sagittal': lambda x: x[:, :, :, slice_idx['sagittal'], :],
            'coronal': lambda x: x[:, :, :, :, slice_idx['coronal']]
        }
        
        for i, plane in enumerate(planes):
            # Plot attention map
            ax_att = fig.add_subplot(gs[0, i+1])
            att_map = attention_maps[plane][0, 0].detach().cpu()
            plt.imshow(att_map, cmap='hot')
            plt.title(f'{plane.capitalize()} Attention')
            plt.colorbar()
            
            # Plot feature slice with attention overlay
            ax_feat = fig.add_subplot(gs[1, i])
            feat_slice = slices[plane](features)[0].mean(0).detach().cpu()
            plt.imshow(feat_slice, cmap='gray')
            plt.imshow(att_map, cmap='hot', alpha=0.5)
            plt.title(f'{plane.capitalize()} Features + Attention')
            
            # Plot attended features
            ax_attended = fig.add_subplot(gs[2, i])
            attended_feat = feat_slice * att_map
            plt.imshow(attended_feat, cmap=self.cmap)
            plt.title(f'Attended {plane.capitalize()} Features')
            plt.colorbar()
        
        plt.tight_layout()
        return fig

    def plot_3d_hierarchy(self, features_dict, attention_dict):
        """Plot 3D hierarchical visualization with volume plots."""
        try:
            fig = plt.figure(figsize=(20, 15))
            n_levels = len(hierarchical_viz)
            gs = plt.GridSpec(2, n_levels)
            
            for i, (title, data) in enumerate(hierarchical_viz.items()):
                # Convert to numpy and ensure 3D
                if torch.is_tensor(data):
                    data = data.detach().cpu().numpy()
                if data.ndim == 2:
                    data = data.reshape(int(np.cbrt(data.size)), -1, -1)
                
                # 3D Volume Plot
                ax_3d = fig.add_subplot(gs[0, i], projection='3d')
                
                # Create volume plot
                x, y, z = np.meshgrid(
                    np.arange(data.shape[0]),
                    np.arange(data.shape[1]),
                    np.arange(data.shape[2])
                )
                
                # Normalize and threshold
                data_norm = (data - data.min()) / (data.max() - data.min() + 1e-8)
                threshold = np.percentile(data_norm, 75)
                mask = data_norm > threshold
                
                # Create scatter plot with enhanced visibility
                scatter = ax_3d.scatter(
                    x[mask], y[mask], z[mask],
                    c=data_norm[mask],
                    cmap='viridis',
                    alpha=0.6,
                    s=50
                )
                
                # Enhance 3D visualization
                cbar = plt.colorbar(scatter, ax=ax_3d)
                cbar.set_label('Feature Response Intensity', fontsize=10)
                ax_3d.view_init(elev=20, azim=45)
                ax_3d.set_xlabel('Depth (D)', fontsize=10)
                ax_3d.set_ylabel('Height (H)', fontsize=10)
                ax_3d.set_zlabel('Width (W)', fontsize=10)
                ax_3d.grid(True, alpha=0.3)
                ax_3d.set_title('3D Feature Response Volume', fontsize=12, pad=20)
                
                # 2. Anatomical Plane Views
                # Axial View (top-down)
                ax_axial = fig.add_subplot(gs[1, 0])
                mid_slice = data.shape[0] // 2
                im_axial = ax_axial.imshow(data[mid_slice], cmap='viridis')
                plt.colorbar(im_axial, ax=ax_axial)
                ax_axial.set_title(f'Axial View (z={mid_slice})', fontsize=12)
                
                # Sagittal View (side)
                ax_sagittal = fig.add_subplot(gs[1, 1])
                mid_slice = data.shape[1] // 2
                im_sagittal = ax_sagittal.imshow(data[:, mid_slice], cmap='viridis')
                plt.colorbar(im_sagittal, ax=ax_sagittal)
                ax_sagittal.set_title(f'Sagittal View (y={mid_slice})', fontsize=12)
                
                # Coronal View (front)
                ax_coronal = fig.add_subplot(gs[1, 2])
                mid_slice = data.shape[2] // 2
                im_coronal = ax_coronal.imshow(data[:, :, mid_slice], cmap='viridis')
                plt.colorbar(im_coronal, ax=ax_coronal)
                ax_coronal.set_title(f'Coronal View (x={mid_slice})', fontsize=12)
                
                plt.tight_layout(pad=3.0)
                plt.savefig(os.path.join(save_dir, f'3d_feature_response_{i+1}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            print(f"Error in 3D hierarchical visualization: {str(e)}")
            traceback.print_exc()

def enhance_and_save_plot(src_file, dst_file):
    """Enhance and save a plot with publication-quality settings."""
    try:
        # Read the source image
        img = plt.imread(src_file)
        
        # Create a new figure with high DPI
        plt.figure(figsize=(10, 8), dpi=300)
        
        # Display the image with enhanced settings
        plt.imshow(img)
        plt.axis('off')  # Remove axes for cleaner look
        
        # Save with high quality settings
        plt.savefig(dst_file, 
                   dpi=300, 
                   bbox_inches='tight',
                   pad_inches=0.1,
                   metadata={'Creator': 'MSMedVision'})
        
        plt.close()
        return True
    except Exception as e:
        print(f"Error enhancing plot {src_file}: {str(e)}")
        return False

def generate_publication_figures(results_dir):
    """Generate publication-ready figures from training results with robust error handling."""
    print("\nGenerating publication-ready figures...")
    
    paper_figures_dir = os.path.join(results_dir, 'paper_figures')
    os.makedirs(paper_figures_dir, exist_ok=True)
    
    # Track successful and failed figure generations
    generated_figures = {
        '2d_performance': False,
        '3d_performance': False,
        'combined_performance': False
    }
    
    try:
        # Load results with error handling
        try:
            with open(os.path.join(results_dir, 'all_results.json'), 'r') as f:
                all_results = json.load(f)
        except Exception as e:
            print(f"Error loading results file: {str(e)}")
            print("Attempting to proceed with empty results...")
            all_results = {}
        
        # Separate 2D and 3D datasets with validation
        results_2d = {}
        results_3d = {}
        
        for k, v in all_results.items():
            try:
                if v.get('status') != 'success':
                    continue
                    
                # Validate required metrics exist
                if not all(key in v for key in ['final_val_acc', 'final_val_auc']):
                    print(f"Warning: Missing metrics for {k}, skipping...")
                    continue
                
                if '3d' in k.lower():
                    results_3d[k] = v
                else:
                    results_2d[k] = v
            except Exception as e:
                print(f"Warning: Error processing dataset {k}: {str(e)}")
                continue
        
        def create_performance_plot(results, title, is_3d=False):
            """Create performance plot with enhanced error handling."""
            try:
                if not results:
                    print(f"Warning: No valid results for {'3D' if is_3d else '2D'} performance plot")
                    return None
                
                datasets = []
                metrics = {'acc': [], 'auc': []}
                ci_metrics = {'acc': {'lower': [], 'upper': []}, 
                             'auc': {'lower': [], 'upper': []}}
                
                # Process each dataset with individual error handling
                for dataset, result in results.items():
                    try:
                        # Basic metric validation
                        acc = result.get('final_val_acc', 0) / 100.0
                        auc = result.get('final_val_auc', 0.5)
                        
                        if not (0 <= acc <= 1) or not (0 <= auc <= 1):
                            print(f"Warning: Invalid metrics for {dataset}, using fallback values")
                            acc = max(0, min(1, acc))
                            auc = max(0, min(1, auc))
                        
                        datasets.append(dataset.replace('mnist', '').replace('3d', ''))
                        metrics['acc'].append(acc)
                        metrics['auc'].append(auc)
                        
                        # Handle confidence intervals with fallbacks
                        ci = result.get('ci_metrics', {})
                        for metric_name in ['acc', 'auc']:
                            metric_ci = ci.get(metric_name, {})
                            current_val = metrics[metric_name][-1]
                            
                            # Ensure CI values are valid
                            ci_low = metric_ci.get('ci_low', current_val)
                            ci_high = metric_ci.get('ci_high', current_val)
                            
                            if metric_name == 'acc':
                                ci_low = ci_low / 100.0 if ci_low is not None else current_val
                                ci_high = ci_high / 100.0 if ci_high is not None else current_val
                            
                            # Validate and bound CI values
                            ci_low = max(0, min(current_val, ci_low if not np.isnan(ci_low) else current_val))
                            ci_high = min(1, max(current_val, ci_high if not np.isnan(ci_high) else current_val))
                            
                            ci_metrics[metric_name]['lower'].append(ci_low)
                            ci_metrics[metric_name]['upper'].append(ci_high)
                            
                    except Exception as e:
                        print(f"Warning: Error processing {dataset}: {str(e)}")
                        continue
                
                if not datasets:
                    return None
                
                # Create figure with error handling
                fig = plt.figure(figsize=(12, 6))
                x = np.arange(len(datasets))
                width = 0.35
                
                def plot_metric_bars(metric_name, offset, color):
                    try:
                        yerr = np.array([
                            np.array(metrics[metric_name]) - np.array(ci_metrics[metric_name]['lower']),
                            np.array(ci_metrics[metric_name]['upper']) - np.array(metrics[metric_name])
                        ])
                        
                        # Ensure no negative error bars
                        yerr = np.maximum(0, yerr)
                        
                        plt.bar(x + offset, metrics[metric_name], width,
                               label=metric_name.upper(),
                               color=color, alpha=0.8,
                               yerr=yerr, capsize=5,
                               error_kw={'elinewidth': 1, 'capthick': 1})
                    except Exception as e:
                        print(f"Warning: Error plotting {metric_name} bars: {str(e)}")
                        # Fallback to simple bars without error bars
                        plt.bar(x + offset, metrics[metric_name], width,
                               label=metric_name.upper(),
                               color=color, alpha=0.8)
                
                # Plot with individual error handling
                try:
                    plot_metric_bars('acc', -width/2, '#3498db')
                except Exception as e:
                    print(f"Warning: Error plotting accuracy bars: {str(e)}")
                
                try:
                    plot_metric_bars('auc', width/2, '#2ecc71')
                except Exception as e:
                    print(f"Warning: Error plotting AUC bars: {str(e)}")
                
                # Customize plot with error handling
                plt.xlabel('Dataset')
                plt.ylabel('Score')
                plt.title(f'{title} ({"3D" if is_3d else "2D"} Datasets)')
                plt.xticks(x, datasets, rotation=45, ha='right')
                plt.legend(loc='lower right')
                plt.grid(True, alpha=0.3)
                plt.ylim(0, 1.0)
                
                # Add value labels with error handling
                def add_value_labels(metric_name, offset):
                    try:
                        for i, v in enumerate(metrics[metric_name]):
                            plt.text(i + offset, v + 0.02,
                                    f'{v:.2f}',
                                    ha='center', va='bottom',
                                    fontsize=8)
                    except Exception as e:
                        print(f"Warning: Error adding value labels for {metric_name}: {str(e)}")
                
                add_value_labels('acc', -width/2)
                add_value_labels('auc', width/2)
                
                plt.tight_layout()
                return fig
                
            except Exception as e:
                print(f"Error creating performance plot: {str(e)}")
                return None
        
        # Generate and save 2D performance plot
        try:
            fig_2d = create_performance_plot(results_2d, 'Performance Metrics', is_3d=False)
            if fig_2d:
                fig_2d.savefig(os.path.join(paper_figures_dir, '2d_performance.png'),
                              dpi=300, bbox_inches='tight')
                generated_figures['2d_performance'] = True
                plt.close(fig_2d)
        except Exception as e:
            print(f"Error saving 2D performance plot: {str(e)}")
        
        # Generate and save 3D performance plot
        try:
            fig_3d = create_performance_plot(results_3d, 'Performance Metrics', is_3d=True)
            if fig_3d:
                fig_3d.savefig(os.path.join(paper_figures_dir, '3d_performance.png'),
                              dpi=300, bbox_inches='tight')
                generated_figures['3d_performance'] = True
                plt.close(fig_3d)
        except Exception as e:
            print(f"Error saving 3D performance plot: {str(e)}")
        
        # Generate combined performance comparison
        try:
            plt.figure(figsize=(15, 8))
            datasets = []
            val_accs = []
            val_aucs = []
            
            for dataset, result in all_results.items():
                try:
                    if result.get('status') != 'success':
                        continue
                    
                    acc = result.get('final_val_acc', 0) / 100.0
                    auc = result.get('final_val_auc', 0.5)
                    
                    # Validate metrics
                    if not (0 <= acc <= 1) or not (0 <= auc <= 1):
                        acc = max(0, min(1, acc))
                        auc = max(0, min(1, auc))
                    
                    datasets.append(dataset.replace('mnist', '').replace('3d', ''))
                    val_accs.append(acc)
                    val_aucs.append(auc)
                except Exception as e:
                    print(f"Warning: Error processing dataset {dataset} for combined plot: {str(e)}")
                    continue
            
            if datasets:
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
                generated_figures['combined_performance'] = True
                plt.close()
        except Exception as e:
            print(f"Error generating combined performance plot: {str(e)}")
        
        # Report generation status
        print("\nFigure generation summary:")
        for fig_name, success in generated_figures.items():
            print(f"{fig_name}: {'✓ Success' if success else '✗ Failed'}")
        
        if any(generated_figures.values()):
            print("\nPublication figures generated successfully (some figures may have failed).")
            print(f"Figures saved in: {paper_figures_dir}")
        else:
            print("\nWarning: No figures were successfully generated.")
        
    except Exception as e:
        print(f"Error in publication figure generation: {str(e)}")
        traceback.print_exc()
    finally:
        # Ensure cleanup
        try:
            plt.close('all')
        except:
            pass