import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import medmnist
from medmnist import INFO
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os
import scipy.ndimage

class PaperFigureGenerator:
    def __init__(self, save_dir='paper_figures'):
        """Initialize the figure generator with publication-ready settings."""
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set publication-ready style
        plt.style.use('default')
        self.font = {'family': 'Arial', 'size': 10}
        plt.rc('font', **self.font)
        
        # High-quality figure settings
        self.dpi = 600  # Publication quality
        plt.rc('figure', dpi=300)  # Base DPI for display
        plt.rc('savefig', dpi=600)  # Higher DPI for saving
        
        # Better image rendering
        plt.rc('image', interpolation='bicubic', resample=True)
        
        # Better vector graphics
        plt.rc('pdf', fonttype=42)
        plt.rc('ps', fonttype=42)
        
        # Better text rendering
        plt.rc('text', usetex=False)
        plt.rc('font', family='sans-serif')
        
        # Better line rendering
        plt.rc('lines', linewidth=1.5)
        plt.rc('axes', linewidth=1.5)
        
        self.subplot_labels = list('abcdefghijklmnopqrstuvwxyz')
    
    def _add_subplot_label(self, ax, label_idx):
        """Add subplot label (a, b, c, etc.) to the axis."""
        if isinstance(ax, Axes3D):
            ax.text2D(-0.1, 1.1, f'({self.subplot_labels[label_idx]})', 
                     transform=ax.transAxes, fontsize=12, fontweight='bold')
        else:
            ax.text(-0.1, 1.1, f'({self.subplot_labels[label_idx]})',
                   transform=ax.transAxes, fontsize=12, fontweight='bold')
    
    def plot_3d_volume(self, ax, volume_data, title=None):
        """Plot 3D volume data as a static, publication-ready visualization with multiple slices."""
        if torch.is_tensor(volume_data):
            volume_data = volume_data.numpy()
        
        # Normalize volume data
        volume_data = (volume_data - volume_data.min()) / (volume_data.max() - volume_data.min())
        
        # Calculate slice positions (9 slices with gaps for better depth perception)
        n_slices = 9
        slice_positions = np.linspace(3, volume_data.shape[2]-3, n_slices, dtype=int)
        
        # Create a custom colormap with transparency
        colors = plt.cm.viridis(np.linspace(0, 1, 256))
        colors[:, 3] = np.linspace(0.3, 1, 256)  # Alpha channel
        custom_cmap = plt.matplotlib.colors.ListedColormap(colors)
        
        # Plot slices with offset for 3D effect
        z_offset = np.linspace(0, 15, n_slices)  # Increased offset for better depth
        for i, (slice_pos, offset) in enumerate(zip(slice_positions, z_offset)):
            # Create meshgrid for this slice
            x, y = np.meshgrid(np.arange(volume_data.shape[0]),
                             np.arange(volume_data.shape[1]))
            
            # Plot the slice with enhanced quality
            slice_data = volume_data[:, :, slice_pos].T
            # Upsample the slice for better quality
            slice_data = scipy.ndimage.zoom(slice_data, 2, order=3)
            x = scipy.ndimage.zoom(x, 2, order=1)
            y = scipy.ndimage.zoom(y, 2, order=1)
            
            surf = ax.plot_surface(x, y, np.full_like(x, offset),
                                 facecolors=custom_cmap(slice_data),
                                 shade=False, alpha=0.95-i*0.08)
        
        # Add semi-transparent isosurface points for 3D context
        threshold = np.percentile(volume_data, 92)  # Slightly lower threshold
        x, y, z = np.where(volume_data > threshold)
        scatter = ax.scatter(x, y, z, 
                           c=volume_data[x, y, z],
                           cmap='viridis',
                           alpha=0.15, s=3)  # Smaller points, more transparent
        
        if title:
            ax.set_title(title, fontsize=12, pad=10)
        
        # Customize axis appearance
        ax.set_xlabel('X', fontsize=8, labelpad=5)
        ax.set_ylabel('Y', fontsize=8, labelpad=5)
        ax.set_zlabel('Z', fontsize=8, labelpad=5)
        
        # Remove ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        # Set optimal viewing angle
        ax.view_init(elev=25, azim=45)
        
        # Adjust axis limits with larger margin for better view
        margin = 8
        ax.set_xlim(-margin, volume_data.shape[0] + margin)
        ax.set_ylim(-margin, volume_data.shape[1] + margin)
        ax.set_zlim(-margin, volume_data.shape[2] + margin + 15)  # Extra space for offset
        
        # Add colorbar with better formatting
        cbar = plt.colorbar(scatter, ax=ax, label='Intensity', shrink=0.7,
                          ticks=[0, 0.5, 1])
        cbar.ax.set_yticklabels(['0', '0.5', '1.0'], fontsize=8)
        
        # Add grid for better depth perception
        ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.5)
        
        return ax
    
    def generate_dataset_examples(self):
        """Generate figure showing examples from each dataset with high resolution."""
        # List all datasets
        datasets_2d = ['pathmnist', 'octmnist', 'pneumoniamnist', 'retinamnist', 
                      'breastmnist', 'bloodmnist', 'tissuemnist', 'organamnist',
                      'organcmnist', 'organsmnist', 'dermamnist', 'chestmnist']
        
        datasets_3d = ['organmnist3d', 'nodulemnist3d', 'adrenalmnist3d', 
                      'fracturemnist3d', 'vesselmnist3d', 'synapsemnist3d']
        
        # Create figure for 2D datasets with higher resolution
        n_cols = 4
        n_rows = (len(datasets_2d) + n_cols - 1) // n_cols
        fig_2d = plt.figure(figsize=(16, 4*n_rows))
        
        for idx, dataset_name in enumerate(datasets_2d):
            # Load dataset info
            info = INFO[dataset_name]
            DataClass = getattr(medmnist, info['python_class'])
            
            # Create dataset instance without downloading
            dataset = DataClass(split='train', download=False)
            
            # Load only the first example using the npz file directly
            npz_file = os.path.join(os.path.expanduser('~'), '.medmnist', f"{dataset_name}.npz")
            if not os.path.exists(npz_file):
                # Download only if necessary
                dataset = DataClass(split='train', download=True)
            
            # Load single example from npz
            data = np.load(npz_file)
            image = data['train_images'][0]  # Get first image only
            
            # Create subplot
            ax = fig_2d.add_subplot(n_rows, n_cols, idx + 1)
            self._add_subplot_label(ax, idx)
            
            # Enhance image quality
            if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[0] == 1):
                # Grayscale image
                if len(image.shape) == 3:
                    image = image[0]
                # Upsample image for better quality
                image = self._enhance_image_quality(image)
                ax.imshow(image, cmap='gray', interpolation='bicubic')
            else:  # RGB
                if image.shape[0] == 3:  # If channels first, transpose
                    image = np.transpose(image, (1, 2, 0))
                # Upsample image for better quality
                image = self._enhance_image_quality(image)
                ax.imshow(image, interpolation='bicubic')
            
            # Get dataset info for title
            task = info.get('task', 'classification')
            n_classes = info.get('n_classes', '')
            title = f"{dataset_name}\n{task} ({n_classes} classes)"
            ax.set_title(title, fontsize=10, pad=5)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'dataset_examples_2d.png'), 
                   bbox_inches='tight', dpi=self.dpi)
        plt.close()
        
        # Create figure for 3D datasets with improved visualization
        n_cols = 2  # Reduced columns for larger visualizations
        n_rows = (len(datasets_3d) + n_cols - 1) // n_cols
        fig_3d = plt.figure(figsize=(12, 6*n_rows), dpi=self.dpi)
        
        for idx, dataset_name in enumerate(datasets_3d):
            # Load dataset info
            info = INFO[dataset_name]
            DataClass = getattr(medmnist, info['python_class'])
            
            # Create dataset instance without downloading
            dataset = DataClass(split='train', download=False)
            
            # Load only the first example using the npz file directly
            npz_file = os.path.join(os.path.expanduser('~'), '.medmnist', f"{dataset_name}.npz")
            if not os.path.exists(npz_file):
                # Download only if necessary
                dataset = DataClass(split='train', download=True)
            
            # Load single example from npz
            data = np.load(npz_file)
            volume = data['train_images'][0]  # Get first volume only
            
            # Create subplot
            ax = fig_3d.add_subplot(n_rows, n_cols, idx + 1, projection='3d')
            self._add_subplot_label(ax, idx)
            
            # Get dataset info for title
            task = info.get('task', 'classification')
            n_classes = info.get('n_classes', '')
            title = f"{dataset_name}\n{task} ({n_classes} classes)"
            
            # Plot volume with improved visualization
            self.plot_3d_volume(ax, volume[0] if len(volume.shape) == 4 else volume, title)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'dataset_examples_3d.png'),
                   bbox_inches='tight', dpi=600)  # Higher DPI for paper
        plt.close()
    
    def generate_model_architecture(self):
        """Generate publication-ready figure showing the detailed model architecture."""
        # Create figure with higher resolution and better proportions
        fig = plt.figure(figsize=(15, 12), dpi=600)
        gs = GridSpec(5, 3, figure=fig)
        
        # Main architecture overview
        ax_main = fig.add_subplot(gs[0:3, :])
        self._add_subplot_label(ax_main, 0)
        ax_main.axis('off')
        self._draw_architecture_components(ax_main)
        
        # Feature extraction detail
        ax_feature = fig.add_subplot(gs[3:, 0])
        self._add_subplot_label(ax_feature, 1)
        ax_feature.axis('off')
        self._draw_feature_extraction_detail(ax_feature)
        
        # Feature fusion detail
        ax_fusion = fig.add_subplot(gs[3:, 1])
        self._add_subplot_label(ax_fusion, 2)
        ax_fusion.axis('off')
        self._draw_fusion_detail(ax_fusion)
        
        # Classification head detail
        ax_class = fig.add_subplot(gs[3:, 2])
        self._add_subplot_label(ax_class, 3)
        ax_class.axis('off')
        self._draw_classification_detail(ax_class)
        
        # Add overall title
        fig.suptitle('Multi-Stream Medical Image Classification Architecture',
                    fontsize=14, fontweight='bold', y=0.98)
        
        # Adjust layout
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'model_architecture.png'),
                   bbox_inches='tight', dpi=600)
        plt.close()
    
    def _draw_stream(self, ax, x, y, width, height, title, layers):
        """Draw a single stream (2D or 3D) with detailed architecture."""
        # Draw stream container with rounded corners and gradient fill
        rect = plt.Rectangle((x, y-height/2), width, height, 
                           facecolor='#f0f0f0', edgecolor='#404040',
                           linewidth=2, alpha=0.3)
        ax.add_patch(rect)
        
        # Add title with better styling
        ax.text(x + width/2, y + height/2 + 0.02, title,
                ha='center', va='bottom', fontsize=14, fontweight='bold',
                color='#202020')
        
        # Add layers with detailed information
        n_layers = len(layers)
        layer_width = width * 0.8
        layer_spacing = layer_width / (n_layers - 1)
        layer_height = height * 0.15
        
        for i, layer_info in enumerate(layers):
            layer_x = x + width*0.1 + i * layer_spacing
            layer_y = y
            
            # Parse layer info
            if isinstance(layer_info, tuple):
                layer_name, details = layer_info
            else:
                layer_name, details = layer_info, None
            
            # Draw layer box with gradient fill
            layer_rect = plt.Rectangle((layer_x, layer_y-layer_height/2),
                                     layer_width/n_layers*0.8, layer_height,
                                     facecolor='#e6f3ff', edgecolor='#2060a0',
                                     linewidth=1.5, alpha=0.9)
            ax.add_patch(layer_rect)
            
            # Add layer text with better formatting
            ax.text(layer_x + layer_width/n_layers*0.4, layer_y + 0.01,
                   layer_name, ha='center', va='center',
                   fontsize=9, color='#202020', fontweight='bold')
            
            if details:
                ax.text(layer_x + layer_width/n_layers*0.4, layer_y - 0.02,
                       details, ha='center', va='center',
                       fontsize=8, color='#404040', style='italic')
            
            # Add arrow if not last layer
            if i < n_layers - 1:
                self._draw_arrow(ax, 
                               layer_x + layer_width/n_layers*0.8,
                               layer_y,
                               layer_spacing*0.15, 0)
    
    def _draw_arrow(self, ax, x, y, dx, dy):
        """Draw a styled arrow."""
        ax.arrow(x, y, dx, dy,
                head_width=0.015, head_length=0.015,
                fc='#2060a0', ec='#2060a0',
                linewidth=1.5, alpha=0.8)
    
    def _draw_architecture_components(self, ax):
        """Draw the main architecture components with detailed network structure."""
        # Define component positions
        input_x = 0.1
        stream_width = 0.3
        fusion_width = 0.25
        
        # Draw 2D Stream with detailed layers
        self._draw_stream(ax, input_x, 0.7, stream_width, 0.25, '2D Stream', [
            ('Input', '28×28×1'),
            ('Conv2D', '3×3, 64\nstride=1, pad=1'),
            ('BN + ReLU', 'batch norm\nReLU activation'),
            ('MaxPool2D', '2×2\nstride=2'),
            ('Conv2D', '3×3, 128\nstride=1, pad=1'),
            ('BN + ReLU', 'batch norm\nReLU activation'),
            ('MaxPool2D', '2×2\nstride=2'),
            ('Features', '128×7×7')
        ])
        
        # Draw 3D Stream with detailed layers
        self._draw_stream(ax, input_x, 0.3, stream_width, 0.25, '3D Stream', [
            ('Input', '28×28×28×1'),
            ('Conv3D', '3×3×3, 64\nstride=1, pad=1'),
            ('BN + ReLU', 'batch norm\nReLU activation'),
            ('MaxPool3D', '2×2×2\nstride=2'),
            ('Conv3D', '3×3×3, 128\nstride=1, pad=1'),
            ('BN + ReLU', 'batch norm\nReLU activation'),
            ('MaxPool3D', '2×2×2\nstride=2'),
            ('Features', '128×7×7×7')
        ])
        
        # Draw Fusion Module with more detail
        fusion_x = input_x + stream_width + 0.05
        self._draw_fusion_module(ax, fusion_x, 0.5, fusion_width, 0.5)
        
        # Draw Classification Head with more detail
        head_x = fusion_x + fusion_width + 0.05
        self._draw_classification_head(ax, head_x, 0.5, 0.2, 0.5)
        
        # Add arrows connecting components with better styling
        self._draw_connections(ax)
        
        # Add annotations for key components
        ax.text(input_x + stream_width/2, 0.95, 'Feature Extraction',
                ha='center', va='bottom', fontsize=12, fontweight='bold',
                color='#202020')
        ax.text(fusion_x + fusion_width/2, 0.95, 'Cross-Modal Fusion',
                ha='center', va='bottom', fontsize=12, fontweight='bold',
                color='#202020')
        ax.text(head_x + 0.1, 0.95, 'Classification',
                ha='center', va='bottom', fontsize=12, fontweight='bold',
                color='#202020')
    
    def _draw_fusion_module(self, ax, x, y, width, height):
        """Draw the feature fusion module with attention mechanism."""
        # Draw module container
        rect = plt.Rectangle((x, y-height/2), width, height,
                           facecolor='none', edgecolor='black')
        ax.add_patch(rect)
        
        # Add title
        ax.text(x + width/2, y + height/2 + 0.02,
                'Feature Fusion', ha='center', va='bottom',
                fontsize=12, fontweight='bold')
        
        # Draw attention components
        attention_height = height * 0.6
        self._draw_attention_component(ax, x + width*0.1, y, width*0.8, attention_height)
        
        # Add feature combination arrow
        ax.arrow(x + width*0.1, y - attention_height*0.4,
                width*0.8, 0, head_width=0.01, head_length=0.01,
                fc='black', ec='black')
    
    def _draw_attention_component(self, ax, x, y, width, height):
        """Draw the attention mechanism detail."""
        # Draw self-attention boxes
        box_width = width * 0.25
        box_height = height * 0.2
        
        # Query, Key, Value boxes
        components = ['Query', 'Key', 'Value']
        for i, comp in enumerate(components):
            box_x = x + i * (box_width + width*0.05)
            box_y = y + height*0.2
            
            rect = plt.Rectangle((box_x, box_y-box_height/2),
                               box_width, box_height,
                               facecolor='lightblue', edgecolor='black')
            ax.add_patch(rect)
            ax.text(box_x + box_width/2, box_y,
                   comp, ha='center', va='center', fontsize=8)
    
    def _draw_classification_head(self, ax, x, y, width, height):
        """Draw the classification head."""
        # Draw container
        rect = plt.Rectangle((x, y-height/2), width, height,
                           facecolor='none', edgecolor='black')
        ax.add_patch(rect)
        
        # Add title
        ax.text(x + width/2, y + height/2 + 0.02,
                'Classification', ha='center', va='bottom',
                fontsize=12, fontweight='bold')
        
        # Draw layers
        layers = ['Global\nPool', 'FC\n256', 'FC\nnum_classes']
        n_layers = len(layers)
        layer_height = height * 0.15
        layer_spacing = height * 0.2
        
        for i, layer_text in enumerate(layers):
            layer_y = y + (i - n_layers/2) * layer_spacing
            
            # Draw layer box
            layer_rect = plt.Rectangle((x + width*0.2, layer_y-layer_height/2),
                                     width*0.6, layer_height,
                                     facecolor='lightgray', edgecolor='black')
            ax.add_patch(layer_rect)
            
            # Add layer text
            ax.text(x + width*0.5, layer_y,
                   layer_text, ha='center', va='center',
                   fontsize=8, multialignment='center')
            
            # Add arrow if not last layer
            if i < n_layers - 1:
                ax.arrow(x + width*0.5, layer_y + layer_height/2,
                        0, layer_spacing*0.5,
                        head_width=0.01, head_length=0.01,
                        fc='black', ec='black')
    
    def _draw_connections(self, ax):
        """Draw arrows connecting the main components."""
        # Connect 2D Stream to Fusion
        ax.arrow(0.4, 0.7, 0.05, -0.1,
                head_width=0.01, head_length=0.01,
                fc='black', ec='black')
        
        # Connect 3D Stream to Fusion
        ax.arrow(0.4, 0.3, 0.05, 0.1,
                head_width=0.01, head_length=0.01,
                fc='black', ec='black')
        
        # Connect Fusion to Classification
        ax.arrow(0.65, 0.5, 0.05, 0,
                head_width=0.01, head_length=0.01,
                fc='black', ec='black')
    
    def _draw_fusion_detail(self, ax):
        """Draw detailed fusion mechanism."""
        # Draw title
        ax.text(0.5, 0.95, 'Feature Fusion Detail',
                ha='center', va='top', fontsize=12, fontweight='bold')
        
        # Draw attention mechanism components
        components = [
            ('2D Features\n128×H/4×W/4', 0.2, 0.8),
            ('3D Features\n128×H/4×W/4', 0.2, 0.2),
            ('Query Transform', 0.5, 0.8),
            ('Key Transform', 0.5, 0.5),
            ('Value Transform', 0.5, 0.2),
            ('Attention Scores', 0.8, 0.5),
            ('Output Features\n256×H/4×W/4', 0.8, 0.2)
        ]
        
        # Draw components
        for name, x, y in components:
            rect = plt.Rectangle((x-0.15, y-0.05), 0.3, 0.1,
                               facecolor='lightblue', edgecolor='black')
            ax.add_patch(rect)
            ax.text(x, y, name, ha='center', va='center',
                   fontsize=8, multialignment='center')
        
        # Draw arrows
        arrows = [
            (0.35, 0.8, 0.35, 0.0),  # 2D to Value
            (0.35, 0.2, 0.35, 0.0),  # 3D to Value
            (0.65, 0.8, 0.15, -0.3),  # Query to Attention
            (0.65, 0.5, 0.15, 0.0),   # Key to Attention
            (0.65, 0.2, 0.15, 0.3),   # Value to Output
        ]
        
        for x, y, dx, dy in arrows:
            ax.arrow(x, y, dx, dy, head_width=0.02, head_length=0.02,
                    fc='black', ec='black')
        
        # Add attention equation
        ax.text(0.5, 0.95, 'Attention(Q, K, V) = softmax(QK⊤/√d)V',
                ha='center', va='center', fontsize=10)
    
    def _draw_feature_extraction_detail(self, ax):
        """Draw detailed feature extraction process."""
        # Draw title
        ax.text(0.5, 0.95, 'Feature Extraction Detail',
                ha='center', va='top', fontsize=12, fontweight='bold')
        
        # Draw conv blocks with details
        blocks = [
            ('Input', 'N×1×H×W', 0.8),
            ('Conv2D\n3×3, 64', '64×H×W', 0.7),
            ('BatchNorm\nReLU', '64×H×W', 0.7),
            ('MaxPool\n2×2', '64×H/2×W/2', 0.6),
            ('Conv2D\n3×3, 128', '128×H/2×W/2', 0.5),
            ('BatchNorm\nReLU', '128×H/2×W/2', 0.5),
            ('MaxPool\n2×2', '128×H/4×W/4', 0.4),
            ('Features', '128×H/4×W/4', 0.3)
        ]
        
        for i, (name, shape, y_pos) in enumerate(blocks):
            # Draw block
            rect = plt.Rectangle((0.2, y_pos-0.05), 0.3, 0.1,
                               facecolor='lightgray', edgecolor='black')
            ax.add_patch(rect)
            
            # Add text
            ax.text(0.35, y_pos, name, ha='center', va='center',
                   fontsize=8, multialignment='center')
            ax.text(0.6, y_pos, shape, ha='left', va='center',
                   fontsize=8, family='monospace')
            
            # Add arrow if not last block
            if i < len(blocks) - 1:
                ax.arrow(0.35, y_pos-0.05, 0, -0.05,
                        head_width=0.02, head_length=0.02,
                        fc='black', ec='black')
    
    def _draw_classification_detail(self, ax):
        """Draw detailed classification process."""
        # Draw title
        ax.text(0.5, 0.95, 'Classification Detail',
                ha='center', va='top', fontsize=12, fontweight='bold')
        
        # Draw classification components
        components = [
            ('Fused Features', '256×H/4×W/4', 0.8),
            ('Global Avg Pool', '256', 0.65),
            ('Dropout(0.5)', '256', 0.5),
            ('Dense + ReLU', '128', 0.35),
            ('Dense + Softmax', 'num_classes', 0.2)
        ]
        
        for i, (name, shape, y_pos) in enumerate(components):
            # Draw component box
            rect = plt.Rectangle((0.2, y_pos-0.05), 0.3, 0.1,
                               facecolor='lightgray', edgecolor='black')
            ax.add_patch(rect)
            
            # Add text
            ax.text(0.35, y_pos, name, ha='center', va='center',
                   fontsize=8, multialignment='center')
            ax.text(0.6, y_pos, shape, ha='left', va='center',
                   fontsize=8, family='monospace')
            
            # Add arrow if not last component
            if i < len(components) - 1:
                ax.arrow(0.35, y_pos-0.05, 0, -0.1,
                        head_width=0.02, head_length=0.02,
                        fc='black', ec='black')
    
    def _draw_attention_detail(self, ax):
        """Draw the attention mechanism detail."""
        # Draw attention components
        components = [
            ('Query Transform', 0.2, 0.7),
            ('Key Transform', 0.2, 0.5),
            ('Value Transform', 0.2, 0.3),
            ('Attention Scores', 0.5, 0.5),
            ('Attention Output', 0.8, 0.5)
        ]
        
        # Draw boxes
        for name, x, y in components:
            self._draw_feature_box(ax, x, y, name)
        
        # Draw arrows
        # Query, Key to Attention Scores
        ax.arrow(0.3, 0.7, 0.1, -0.1,
                head_width=0.02, head_length=0.02,
                fc='black', ec='black')
        ax.arrow(0.3, 0.5, 0.1, 0,
                head_width=0.02, head_length=0.02,
                fc='black', ec='black')
        
        # Value to Output
        ax.arrow(0.3, 0.3, 0.1, 0.1,
                head_width=0.02, head_length=0.02,
                fc='black', ec='black')
        
        # Attention Scores to Output
        ax.arrow(0.6, 0.5, 0.1, 0,
                head_width=0.02, head_length=0.02,
                fc='black', ec='black')
        
        # Add equations
        ax.text(0.5, 0.8, 'Attention(Q, K, V) = softmax(QK⊤/√d)V',
                ha='center', va='center', fontsize=10)
        
        # Add title
        ax.text(0.5, 0.95, 'Attention Mechanism Detail',
                ha='center', va='top', fontsize=12, fontweight='bold')
    
    def _draw_feature_box(self, ax, x, y, text):
        """Helper method to draw a feature box with text."""
        width = 0.15
        height = 0.1
        rect = plt.Rectangle((x-width/2, y-height/2), width, height,
                           facecolor='lightgray', edgecolor='black')
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center',
                fontsize=8, multialignment='center')
    
    def _draw_attention_box(self, ax, x, y):
        """Helper method to draw attention mechanism box."""
        width = 0.3
        height = 0.2
        rect = plt.Rectangle((x-width/2, y-height/2), width, height,
                           facecolor='lightblue', edgecolor='black')
        ax.add_patch(rect)
        ax.text(x, y+height/4, 'Cross-Modal\nAttention',
                ha='center', va='center', fontsize=10,
                multialignment='center')
        ax.text(x, y-height/4, 'softmax(QK⊤/√d)V',
                ha='center', va='center', fontsize=8)
    
    def _enhance_image_quality(self, image):
        """Enhance image quality using bicubic interpolation."""
        # Convert to float for better interpolation
        image = image.astype(float)
        
        # Normalize to [0, 1]
        image = (image - image.min()) / (image.max() - image.min())
        
        # Upsample image by factor of 4 using bicubic interpolation
        if len(image.shape) == 2:
            return scipy.ndimage.zoom(image, 4, order=3)
        else:
            return scipy.ndimage.zoom(image, (4, 4, 1), order=3)

def main():
    """Generate all paper figures."""
    generator = PaperFigureGenerator()
    
    # Generate dataset example figures
    print("Generating dataset examples...")
    generator.generate_dataset_examples()
    
    # Generate model architecture figure
    print("Generating model architecture diagram...")
    generator.generate_model_architecture()
    
    print(f"All figures saved in: {generator.save_dir}")

if __name__ == "__main__":
    main() 