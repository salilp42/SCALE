import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from typing import List, Optional, Tuple, Dict, Union
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
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import matplotlib.patheffects as path_effects
import colorsys

# Enhanced color maps for medical visualization
TISSUE_CMAP = LinearSegmentedColormap.from_list('tissue', 
    ['#313695', '#4575b4', '#74add1', '#abd9e9', '#fee090', '#fdae61', '#f46d43', '#d73027'])
ATTENTION_CMAP = LinearSegmentedColormap.from_list('attention',
    [(0, '#FFFFFF00'), (0.2, '#FFE4B522'), (0.5, '#FF990044'), (0.8, '#FF440088'), (1, '#FF0000AA')])

def create_transparent_cmap(base_color: str, name: str = 'custom_transparent') -> LinearSegmentedColormap:
    """Create a transparent colormap from a base color."""
    rgb = plt.cm.colors.to_rgb(base_color)
    colors = [(0, (*rgb, 0)), (1, (*rgb, 1))]
    return LinearSegmentedColormap.from_list(name, colors)

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

    def plot_scale_aware_attention(self, attention_map, save_dir):
        """Plot scale-aware attention visualization."""
        try:
            # Ensure attention_map is on CPU and detached
            if torch.is_tensor(attention_map):
                attention_map = attention_map.cpu().detach()
            
            # Handle different input shapes
            if attention_map.ndim == 4:  # [B, C, H, W]
                attention_map = attention_map[0].mean(dim=0)  # Take first batch and average over channels
            elif attention_map.ndim == 5:  # [B, C, D, H, W]
                attention_map = attention_map[0].mean(dim=(0, 1))  # Take first batch and average over channels and depth
            elif attention_map.ndim == 3:  # [C, H, W]
                attention_map = attention_map.mean(dim=0)  # Average over channels
            elif attention_map.ndim == 2:  # [H, W]
                pass  # Already in correct format
            
            # Remove any remaining singleton dimensions
            attention_map = attention_map.squeeze()
            
            # Ensure we have a 2D array for plotting
            if attention_map.ndim != 2:
                raise ValueError(f"Unexpected attention map shape: {attention_map.shape}")
            
            # Create figure
            plt.figure(figsize=(8, 8))
            plt.imshow(attention_map.numpy(), cmap='viridis')
            plt.colorbar()
            plt.title('Scale-Aware Attention Map')
            plt.axis('off')
            
            # Save figure
            save_path = os.path.join(save_dir, 'scale_attention.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Warning: Error in plot_scale_aware_attention: {str(e)}")
            traceback.print_exc()

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
            feat_3d = attention_maps[f'3d_{scale}']
            
            # Regular 2D similarity
            similarity = self.compute_feature_similarity_2d(feat_2d, feat_3d)
            relationships[f'cross_modal_{scale}'] = similarity
            
            # Handle 3D attention maps if available
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
        super().__init__()
        self.figure_dpi = 300
        self.subplot_size = 3
        
    def create_multi_slice_grid(self, volume: torch.Tensor, attention: Optional[torch.Tensor] = None,
                              num_slices: int = 9, overlay_alpha: float = 0.4,
                              att_threshold: float = 0.1, orientation: str = 'axial') -> plt.Figure:
        """Create an enhanced grid of slices from a 3D volume with optional attention overlay.
        
        Args:
            volume (torch.Tensor): Input volume of shape [B, C, D, H, W]
            attention (Optional[torch.Tensor]): Optional attention map of same shape as volume
            num_slices (int): Number of slices to display in the grid (default: 9)
            overlay_alpha (float): Alpha value for attention overlay (default: 0.4)
            att_threshold (float): Threshold for masking low attention values (default: 0.1)
            orientation (str): Slice orientation ('axial', 'coronal', 'sagittal') (default: 'axial')
            
        Returns:
            plt.Figure: Figure containing the multi-slice grid with overlays
        """
        if volume.dim() != 5:  # [B, C, D, H, W]
            raise ValueError(f"Expected 5D volume, got {volume.dim()}D")
            
        # Take first batch and channel
        volume = volume[0, 0].cpu().numpy()
        if attention is not None:
            if attention.shape != volume.shape:
                raise ValueError(f"Attention shape {attention.shape} must match volume shape {volume.shape}")
            attention = attention[0, 0].cpu().numpy()
            
        D = volume.shape[0]
        
        # Calculate optimal slice spacing
        spacing = max(1, D // (num_slices + 1))
        start_idx = spacing
        end_idx = D - spacing
        if num_slices > 1:
            slice_indices = np.linspace(start_idx, end_idx, num_slices, dtype=int)
        else:
            slice_indices = [D // 2]  # Middle slice for single-slice case
        
        # Calculate grid layout
        n_rows = int(np.sqrt(num_slices))
        n_cols = int(np.ceil(num_slices / n_rows))
        
        # Create figure with space for colorbar
        fig = plt.figure(figsize=(n_cols * 4 + 1, n_rows * 4), dpi=self.figure_dpi)
        gs = GridSpec(n_rows, n_cols, figure=fig)
        
        # Normalize volume for display
        volume = (volume - volume.min()) / (volume.max() - volume.min())
        
        # Store axes for colorbar
        axes = []
        
        # Create subplots
        for idx, slice_idx in enumerate(slice_indices):
            row = idx // n_cols
            col = idx % n_cols
            ax = fig.add_subplot(gs[row, col])
            axes.append(ax)
            
            # Plot volume slice in grayscale
            ax.imshow(volume[slice_idx], cmap='gray', interpolation='nearest')
            
            # Overlay attention if provided
            if attention is not None:
                # Normalize attention values to [0, 1]
                att_slice = attention[slice_idx]
                att_slice = (att_slice - att_slice.min()) / (att_slice.max() - att_slice.min() + 1e-8)
                
                # Create masked array for low attention values
                att_masked = np.ma.masked_where(att_slice < att_threshold, att_slice)
                
                # Plot attention overlay with custom colormap
                ax.imshow(att_masked, cmap=ATTENTION_CMAP, alpha=overlay_alpha,
                         interpolation='nearest')
            
            # Add slice number and percentage
            slice_percent = (slice_idx / (D-1)) * 100
            ax.text(0.02, 0.98, f'Slice {slice_idx} ({slice_percent:.0f}%)', 
                   transform=ax.transAxes, color='white',
                   bbox=dict(facecolor='black', alpha=0.7),
                   verticalalignment='top', fontsize=8)
            
            # Add anatomical labels based on orientation
            if orientation.lower() == 'axial':
                labels = [('A', 0.02, 0.02), ('P', 0.98, 0.02),
                         ('L', 0.02, 0.5), ('R', 0.98, 0.5)]
            elif orientation.lower() == 'coronal':
                labels = [('S', 0.02, 0.02), ('I', 0.98, 0.02),
                         ('L', 0.02, 0.5), ('R', 0.98, 0.5)]
            elif orientation.lower() == 'sagittal':
                labels = [('S', 0.02, 0.02), ('I', 0.98, 0.02),
                         ('A', 0.02, 0.5), ('P', 0.98, 0.5)]
            
            for label, x, y in labels:
                ax.text(x, y, label, transform=ax.transAxes, color='white',
                       bbox=dict(facecolor='black', alpha=0.7),
                       ha='left' if x < 0.5 else 'right',
                       va='bottom' if y < 0.5 else 'center',
                       fontsize=8)
            
            ax.axis('off')
        
        # Add colorbar for attention if provided
        if attention is not None:
            # Create new axes for colorbar
            cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
            norm = plt.Normalize(vmin=0, vmax=1)
            sm = plt.cm.ScalarMappable(cmap=ATTENTION_CMAP, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, cax=cbar_ax)
            cbar.set_label('Attention', fontsize=10)
            cbar.ax.tick_params(labelsize=8)
        
        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0, 0, 0.9 if attention is not None else 1, 1])
        
        return fig
        
    def create_attention_overlay_2d(self, image: torch.Tensor, attention: torch.Tensor,
                                  alpha_range: Tuple[float, float] = (0.2, 0.8)) -> np.ndarray:
        """Create enhanced attention overlay for 2D images."""
        # Convert to numpy and normalize
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if isinstance(attention, torch.Tensor):
            attention = attention.cpu().numpy()
            
        # Ensure correct dimensions
        if image.ndim == 4:  # [B, C, H, W]
            image = image[0]
        if image.ndim == 3:  # [C, H, W]
            image = np.transpose(image, (1, 2, 0))
            
        if attention.ndim == 4:
            attention = attention[0, 0]
        elif attention.ndim == 3:
            attention = attention[0]
            
        # Normalize image and attention
        image = (image - image.min()) / (image.max() - image.min())
        attention = (attention - attention.min()) / (attention.max() - attention.min())
        
        # Create color overlay
        attention_colored = plt.cm.jet(attention)
        attention_colored[..., 3] = np.clip(attention, *alpha_range)
        
        # Blend
        result = image.copy()
        if image.shape[-1] == 1:  # Grayscale
            result = np.repeat(result, 3, axis=-1)
        
        mask = attention_colored[..., 3][..., None]
        result = result * (1 - mask) + attention_colored[..., :3] * mask
        
        return result
        
    def create_attention_overlay_3d(self, volume: torch.Tensor, attention: torch.Tensor,
                                  slices_per_axis: int = 3) -> Dict[str, np.ndarray]:
        """Create enhanced attention overlays for 3D volumes."""
        if isinstance(volume, torch.Tensor):
            volume = volume.cpu().numpy()
        if isinstance(attention, torch.Tensor):
            attention = attention.cpu().numpy()
            
        # Ensure 5D format [B, C, D, H, W]
        if volume.ndim == 4:
            volume = volume[None]
        if attention.ndim == 4:
            attention = attention[None]
            
        B, C, D, H, W = volume.shape
        
        # Get slice indices for each axis
        d_indices = np.linspace(0, D-1, slices_per_axis, dtype=int)
        h_indices = np.linspace(0, H-1, slices_per_axis, dtype=int)
        w_indices = np.linspace(0, W-1, slices_per_axis, dtype=int)
        
        results = {}
        
        # Create overlays for each axis
        for axis_name, indices, axis in [('sagittal', w_indices, 4),
                                       ('coronal', h_indices, 3),
                                       ('axial', d_indices, 2)]:
            # Create figure for this axis
            fig = plt.figure(figsize=(4*slices_per_axis, 4), dpi=self.figure_dpi)
            gs = GridSpec(1, slices_per_axis)
            
            slices = []
            for idx, slice_idx in enumerate(indices):
                # Extract slice
                if axis == 4:  # Sagittal
                    vol_slice = volume[0, 0, :, :, slice_idx]
                    att_slice = attention[0, 0, :, :, slice_idx]
                elif axis == 3:  # Coronal
                    vol_slice = volume[0, 0, :, slice_idx, :]
                    att_slice = attention[0, 0, :, slice_idx, :]
                else:  # Axial
                    vol_slice = volume[0, 0, slice_idx, :, :]
                    att_slice = attention[0, 0, slice_idx, :, :]
                
                # Create overlay
                ax = fig.add_subplot(gs[0, idx])
                im = ax.imshow(vol_slice, cmap='gray')
                
                # Add attention overlay with enhanced visibility
                att_masked = np.ma.masked_where(att_slice < 0.1, att_slice)
                ax.imshow(att_masked, cmap=ATTENTION_CMAP, alpha=0.6)
                
                # Add slice indicator
                ax.text(0.02, 0.98, f'Slice {slice_idx}',
                       transform=ax.transAxes, color='white',
                       bbox=dict(facecolor='black', alpha=0.7),
                       verticalalignment='top')
                
                ax.axis('off')
                
            plt.tight_layout()
            
            # Convert figure to image array using buffer_rgba
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            data = np.asarray(buf)
            results[axis_name] = data
            plt.close(fig)
            
        return results
        
    def plot_feature_correlations(self, correlations: torch.Tensor, scale: str,
                                is_3d: bool = False, enhanced: bool = True) -> plt.Figure:
        """Plot enhanced feature correlations with improved visibility."""
        if isinstance(correlations, torch.Tensor):
            correlations = correlations.cpu().numpy()
        
        # Ensure correlations is 2D
        if correlations.ndim == 1:
            correlations = correlations.reshape(-1, 1)
        elif correlations.ndim > 2:
            correlations = correlations.reshape(correlations.shape[0], -1)
        
        fig = plt.figure(figsize=(10, 8), dpi=self.figure_dpi)
        ax = fig.add_subplot(111)
        
        # Use enhanced colormap for better visualization
        im = ax.imshow(correlations, cmap=TISSUE_CMAP, aspect='auto')
        
        if enhanced:
            # Add correlation values as text
            for i in range(correlations.shape[0]):
                for j in range(correlations.shape[1]):
                    value = correlations[i, j]
                    text_color = 'white' if abs(value) > 0.5 else 'black'
                    text = ax.text(j, i, f'{value:.2f}',
                                 ha='center', va='center', color=text_color,
                                 fontsize=8, fontweight='bold')
                    # Add outline for better visibility
                    text.set_path_effects([
                        path_effects.Stroke(linewidth=2, foreground='black'),
                        path_effects.Normal()
                    ])
        
        plt.colorbar(im)
        
        # Enhanced title and labels
        dimension = '3D' if is_3d else '2D'
        plt.title(f'{dimension} Feature Correlations - {scale} Scale',
                 pad=20, fontsize=12, fontweight='bold')
        plt.xlabel('Feature Index', labelpad=10)
        plt.ylabel('Feature Index', labelpad=10)
        
        plt.tight_layout()
        return fig
    
    def create_visualization(self, attention_maps, feature_correlations, 
                           original_image_2d=None, original_image_3d=None):
        """Create comprehensive visualization with enhanced layouts and overlays."""
        results = {}
        
        try:
            # Create main figure with improved layout
            fig = plt.figure(figsize=(20, 15), dpi=self.figure_dpi)
            
            # Calculate needed columns for feature correlations
            n_correlations = len(feature_correlations) if feature_correlations else 0
            n_cols = max(4, n_correlations)  # At least 4 columns, or more if needed
            gs = GridSpec(3, n_cols, figure=fig)
            
            # 1. Original Data Display
            if original_image_2d is not None:
                ax_2d = fig.add_subplot(gs[0, 0])
                self._plot_2d_data(ax_2d, original_image_2d, "2D Input")
            
            if original_image_3d is not None:
                # Create multi-slice grid for 3D data
                multi_slice_fig = self.create_multi_slice_grid(
                    original_image_3d,
                    attention=attention_maps.get('3d_attention'),
                    num_slices=9
                )
                results['3d_slices'] = multi_slice_fig
            
            # 2. Attention Overlays
            if attention_maps:
                # 2D attention overlay
                if 'attention_2d' in attention_maps:
                    ax_att_2d = fig.add_subplot(gs[0, 1])
                    overlay_2d = self.create_attention_overlay_2d(
                        original_image_2d,
                        attention_maps['attention_2d']
                    )
                    ax_att_2d.imshow(overlay_2d)
                    ax_att_2d.set_title('2D Attention Map', pad=10)
                    ax_att_2d.axis('off')
                
                # 3D attention overlays
                if 'attention_3d' in attention_maps:
                    overlays_3d = self.create_attention_overlay_3d(
                        original_image_3d,
                        attention_maps['attention_3d'],
                        slices_per_axis=3
                    )
                    
                    # Plot each anatomical plane
                    for idx, (plane, overlay) in enumerate(overlays_3d.items()):
                        ax = fig.add_subplot(gs[1, idx])
                        ax.imshow(overlay)
                        ax.set_title(f'{plane.capitalize()} View', pad=10)
                        ax.axis('off')
            
            # 3. Feature Correlations
            if feature_correlations:
                for idx, (scale, corr) in enumerate(feature_correlations.items()):
                    if idx >= n_cols:  # Skip if we've run out of columns
                        print(f"Warning: Skipping correlation plot for {scale} due to space constraints")
                        continue
                        
                    ax_corr = fig.add_subplot(gs[2, idx])
                    corr_fig = self.plot_feature_correlations(
                        corr,
                        scale,
                        is_3d='3d' in scale.lower(),
                        enhanced=True
                    )
                    
                    # Copy the correlation plot to the main figure
                    ax_corr.clear()  # Clear existing content
                    for child in corr_fig.axes[0].get_children():
                        if isinstance(child, plt.matplotlib.image.AxesImage):
                            ax_corr.imshow(child.get_array(), cmap=child.get_cmap())
                        elif isinstance(child, plt.matplotlib.text.Text):
                            # Get text position and content
                            pos = child.get_position()
                            text = child.get_text()
                            # Copy text with basic properties
                            ax_corr.text(pos[0], pos[1], text,
                                       color=child.get_color(),
                                       fontsize=child.get_fontsize(),
                                       ha=child.get_horizontalalignment(),
                                       va=child.get_verticalalignment())
                    
                    # Copy title and labels
                    ax_corr.set_title(corr_fig.axes[0].get_title())
                    ax_corr.set_xlabel(corr_fig.axes[0].get_xlabel())
                    ax_corr.set_ylabel(corr_fig.axes[0].get_ylabel())
                    plt.close(corr_fig)
            
            # Add super title
            fig.suptitle('Hierarchical Feature Analysis', 
                        fontsize=16, fontweight='bold', y=0.95)
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            results['main_figure'] = fig
            
        except Exception as e:
            print(f"Error in visualization creation: {str(e)}")
            traceback.print_exc()
            
        return results
        
    def _plot_2d_data(self, ax, image, title):
        """Helper method to plot 2D data with enhanced appearance."""
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
            
        if image.ndim == 4:  # [B, C, H, W]
            image = image[0]
        if image.ndim == 3:  # [C, H, W]
            image = np.transpose(image, (1, 2, 0))
            
        # Normalize for display
        image = (image - image.min()) / (image.max() - image.min())
        
        if image.shape[-1] == 1:
            ax.imshow(image[..., 0], cmap='gray')
        else:
            ax.imshow(image)
            
        ax.set_title(title, pad=10)
        ax.axis('off')
    
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
            lower_errors = []
            upper_errors = []
            for val, low, high in zip(values, ci_low, ci_high):
                if low is not None and high is not None:
                    lower_errors.append(max(0, val - low))
                    upper_errors.append(max(0, high - val))
                else:
                    lower_errors.append(0)
                    upper_errors.append(0)
            return np.array([lower_errors, upper_errors])
        
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
        
        # Set up bar positions
        x = np.arange(len(datasets))
        width = 0.35
        
        # Create bars with refined styling and publication colors
        rects1 = ax.bar(x - width/2, aucs, width, label='AUC',
                       color='#4C72B0', edgecolor='black',
                       linewidth=0.5, alpha=0.8,
                       yerr=auc_yerr, capsize=3, error_kw={'elinewidth': 0.5, 'capthick': 0.5})
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
    """Generate publication-ready figures from training results."""
    print("\nGenerating publication-ready figures...")
    
    paper_figures_dir = os.path.join(results_dir, 'paper_figures')
    os.makedirs(paper_figures_dir, exist_ok=True)
    
    try:
        # Load all results
        with open(os.path.join(results_dir, 'all_results.json'), 'r') as f:
            all_results = json.load(f)
        
        # Separate 2D and 3D datasets
        results_2d = {k: v for k, v in all_results.items() if '3d' not in k.lower()}
        results_3d = {k: v for k, v in all_results.items() if '3d' in k.lower()}
        
        def create_performance_plot(results, title, is_3d=False):
            """Create performance plot with confidence intervals."""
            datasets = []
            metrics = {'acc': [], 'auc': []}
            ci_metrics = {'acc': {'lower': [], 'upper': []}, 
                         'auc': {'lower': [], 'upper': []}}
            
            for dataset, result in results.items():
                if result['status'] == 'success':
                    datasets.append(dataset.replace('mnist', '').replace('3d', ''))
                    
                    # Get metrics
                    metrics['acc'].append(result['final_val_acc'] / 100.0)  # Convert to [0,1]
                    metrics['auc'].append(result['final_val_auc'])
                    
                    # Get confidence intervals
                    ci = result.get('ci_metrics', {})
                    acc_ci = ci.get('acc', {})
                    auc_ci = ci.get('auc', {})
                    
                    # Handle accuracy CIs
                    if acc_ci and not np.isnan(acc_ci.get('ci_low', np.nan)):
                        ci_metrics['acc']['lower'].append(acc_ci['ci_low'] / 100.0)
                        ci_metrics['acc']['upper'].append(acc_ci['ci_high'] / 100.0)
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
        
        print("Publication figures generated successfully.")
        
    except Exception as e:
        print(f"Error generating publication figures: {str(e)}")
        traceback.print_exc()
    finally:
        plt.close('all')