import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from visualization.interpretability import ScaleAwareAttention, CrossModalFeatureMapper, HierarchicalVizEngine

class EnhancedResBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # SE block
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 16, out_channels, 1),
            nn.Sigmoid()
        )
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Apply SE attention
        out = out * self.se(out)
        
        out += residual
        out = F.relu(out)
        return out

class EnhancedResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.1):
        super().__init__()
        
        # Main convolution branch with channel expansion
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        
        # Enhanced channel attention with better reduction ratio
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(out_channels, out_channels//4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels//4, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Anatomical plane attention with batch norm
        self.axial_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.sagittal_pool = nn.AdaptiveAvgPool3d((None, 1, None))
        self.coronal_pool = nn.AdaptiveAvgPool3d((None, None, 1))
        
        self.plane_attention = nn.Sequential(
            nn.Conv3d(out_channels*3, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout3d(p=dropout_rate)
        
    def forward(self, x):
        identity = self.shortcut(x)
        
        # Main branch with batch norm
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        # Multi-scale channel attention
        channel_att = self.se(out)
        out = out * channel_att
        
        # Anatomical plane attention
        axial = self.axial_pool(out)
        sagittal = self.sagittal_pool(out)
        coronal = self.coronal_pool(out)
        
        # Combine plane features
        plane_features = torch.cat([
            axial.expand_as(out),
            sagittal.expand_as(out),
            coronal.expand_as(out)
        ], dim=1)
        
        plane_att = self.plane_attention(plane_features)
        out = out * plane_att
        
        # Add identity connection and final activation
        out = out + identity
        out = F.relu(out)
        
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=28):
        super().__init__()
        pe = torch.zeros(max_len, max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(1), :x.size(2)].unsqueeze(0)

class Stream2D(nn.Module):
    def __init__(self, in_channels, base_channels=64, dropout_rate=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        
        # Initial convolution with input channel handling
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        # Main layers - progressive downsampling
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(base_channels * 2)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        
        # Final convolution to match 3D stream
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(base_channels * 4)
        self.dropout3 = nn.Dropout2d(dropout_rate)
        
    def forward(self, x):
        # Handle input dimensions
        if x.dim() == 5:  # [B, C, D, H, W]
            # Take middle slice for 3D data
            D = x.size(2)
            x = x[:, :, D//2, :, :]  # Take middle slice along depth dimension
        elif x.dim() != 4:  # Not [B, C, H, W]
            raise ValueError(f"Expected 4D or 5D input, got {x.dim()}D")
            
        # Handle single channel input
        if x.size(1) == 1 and self.in_channels == 3:
            x = x.repeat(1, 3, 1, 1)
        elif x.size(1) != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {x.size(1)}")
            
        # Store intermediate features for interpretability
        features = []
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)  # [B, 64, 28, 28]
        features.append(x)  # cellular level
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)  # [B, 128, 14, 14]
        features.append(x)  # tissue level
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)  # [B, 256, 7, 7]
        features.append(x)  # organ level
        
        return x, features

class Stream3D(nn.Module):
    def __init__(self, in_channels, base_channels=64):
        super().__init__()
        self.in_channels = in_channels
        
        # Initial convolution with increased channels and batch norm
        self.conv1 = nn.Conv3d(in_channels, base_channels*2, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(base_channels*2)
        
        # First stage - high resolution features
        self.stage1 = nn.Sequential(
            EnhancedResBlock3D(base_channels*2, base_channels*2, dropout_rate=0.1),
            EnhancedResBlock3D(base_channels*2, base_channels*2, dropout_rate=0.1),
            EnhancedResBlock3D(base_channels*2, base_channels*2, dropout_rate=0.1)
        )
        
        # First transition with increased channels
        self.transition1 = nn.Sequential(
            EnhancedResBlock3D(base_channels*2, base_channels*4, stride=2, dropout_rate=0.1),
            nn.BatchNorm3d(base_channels*4),
            nn.ReLU(inplace=True)
        )
        
        # Second stage - medium resolution
        self.stage2 = nn.Sequential(
            EnhancedResBlock3D(base_channels*4, base_channels*4, dropout_rate=0.1),
            EnhancedResBlock3D(base_channels*4, base_channels*4, dropout_rate=0.1),
            EnhancedResBlock3D(base_channels*4, base_channels*4, dropout_rate=0.1)
        )
        
        # Second transition
        self.transition2 = nn.Sequential(
            EnhancedResBlock3D(base_channels*4, base_channels*8, stride=2, dropout_rate=0.1),
            nn.BatchNorm3d(base_channels*8),
            nn.ReLU(inplace=True)
        )
        
        # Final stage - low resolution, high channels
        self.stage3 = nn.Sequential(
            EnhancedResBlock3D(base_channels*8, base_channels*8, dropout_rate=0.1),
            EnhancedResBlock3D(base_channels*8, base_channels*8, dropout_rate=0.1)
        )
        
        # Cross-scale attention with higher capacity
        self.cross_scale_attention = nn.Sequential(
            nn.Conv3d(base_channels*8, base_channels*2, kernel_size=1),
            nn.BatchNorm3d(base_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels*2, base_channels*8, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Final 1x1 conv to match 2D stream dimensions
        self.final_conv = nn.Conv3d(base_channels*8, base_channels*4, kernel_size=1)
        self.final_bn = nn.BatchNorm3d(base_channels*4)
        
    def forward(self, x):
        # Handle input dimensions
        if x.dim() == 4:  # [B, C, H, W]
            x = x.unsqueeze(2)  # Add depth dimension [B, C, 1, H, W]
            x = x.repeat(1, 1, 28, 1, 1)  # Repeat along depth dimension
        elif x.dim() == 5:  # [B, C, D, H, W]
            if x.size(2) != 28:  # Ensure depth dimension is 28
                x = F.interpolate(x, size=(28, x.size(3), x.size(4)), mode='trilinear', align_corners=False)
        else:
            raise ValueError(f"Expected 4D or 5D input, got {x.dim()}D")
            
        # Handle single channel input
        if x.size(1) == 1 and self.in_channels == 3:
            x = x.repeat(1, 3, 1, 1, 1)
        elif x.size(1) != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {x.size(1)}")
            
        # Store intermediate features for interpretability
        features = []
        
        # Initial convolution with batch norm
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 128, 28, 28, 28]
        features.append(x)  # cellular level
        
        # First stage - high resolution features
        x1 = self.stage1(x)  # [B, 128, 28, 28, 28]
        features.append(x1)  # tissue level
        
        # First transition and second stage
        x = self.transition1(x1)  # [B, 256, 14, 14, 14]
        x2 = self.stage2(x)  # [B, 256, 14, 14, 14]
        features.append(x2)  # organ level
        
        # Second transition and final stage
        x = self.transition2(x2)  # [B, 512, 7, 7, 7]
        x3 = self.stage3(x)  # [B, 512, 7, 7, 7]
        
        # Apply cross-scale attention
        att = self.cross_scale_attention(x3)
        x = x3 * att
        
        # Final 1x1 conv to match dimensions
        x = F.relu(self.final_bn(self.final_conv(x)))  # [B, 256, 7, 7, 7]
        
        features.append(x)  # final attended features
        
        # Global average pooling over depth dimension to match 2D stream
        x = x.mean(dim=2)  # [B, 256, 7, 7]
        
        return x, features

class CrossModalAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x, attn

class FeatureCalibration(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        y = torch.mean(x, dim=[2, 3])
        y = self.fc(y)
        return x * y.view(B, C, 1, 1)

class CrossModalFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # Feature refinement for each modality
        self.refine_2d = FeatureCalibration(dim)
        self.refine_3d = FeatureCalibration(dim)
        
        # Cross-modal attention
        self.attention = CrossModalAttention(dim)
        
        # Final fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x_2d, x_3d):
        # Ensure inputs have the same shape and expected dimensions
        if x_2d.shape != x_3d.shape:
            raise ValueError(f"Shape mismatch: x_2d {x_2d.shape} vs x_3d {x_3d.shape}")
        
        B, C, H, W = x_2d.shape
        if C != self.dim:
            raise ValueError(f"Expected {self.dim} channels, got {C}")
        
        # Refine features
        x_2d = self.refine_2d(x_2d)  # [B, 256, 7, 7]
        x_3d = self.refine_3d(x_3d)  # [B, 256, 7, 7]
        
        # Reshape for attention
        x_2d_flat = x_2d.flatten(2).transpose(1, 2)  # [B, 49, 256]
        x_3d_flat = x_3d.flatten(2).transpose(1, 2)  # [B, 49, 256]
        
        # Compute cross-modal attention
        x_combined = torch.cat([x_2d_flat, x_3d_flat], dim=1)  # [B, 98, 256]
        x_fused, self.attn_weights = self.attention(x_combined)  # [B, 98, 256]
        
        # Extract relevant part of fused features
        x_fused = x_fused[:, :H*W]  # [B, 49, 256]
        
        # Reshape back to spatial dimensions
        x_fused = x_fused.transpose(1, 2).reshape(B, C, H, W)  # [B, 256, 7, 7]
        
        # Concatenate and fuse
        x_cat = torch.cat([x_2d, x_fused], dim=1)  # [B, 512, 7, 7]
        out = self.fusion_conv(x_cat)  # [B, 256, 7, 7]
        
        return out

class MSMedVision(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.n_classes = dataset.n_classes
        self.task_type = dataset.task_type
        
        # Get dataset-specific configuration
        config = get_dataset_config(dataset.task)
        dropout_rate = config['dropout_rate']
        
        # Input channels
        in_channels = dataset.n_channels
        
        # Streams with dropout
        self.stream_2d = Stream2D(in_channels, dropout_rate=dropout_rate)
        self.stream_3d = Stream3D(in_channels)  # 3D stream already has dropout
        
        # Cross-modal fusion
        self.fusion = CrossModalFusion(256)  # Using base_channels * 4
        
        # Classification head with dropout
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),  # Add dropout before final classification
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),  # Add another dropout layer
            nn.Linear(512, self.n_classes)
        )
        
        # Initialize interpretability engines
        self.scale_attention = ScaleAwareAttention()
        self.cross_modal_mapper = CrossModalFeatureMapper()
        self.viz_engine = HierarchicalVizEngine()
        
    def forward(self, x):
        # Get features from both streams
        x_2d, features_2d = self.stream_2d(x)
        x_3d, features_3d = self.stream_3d(x)
        
        # Store features for visualization
        self.last_features = {
            '2d': features_2d,
            '3d': features_3d,
            'final_2d': x_2d,
            'final_3d': x_3d
        }
        
        # Fuse features
        fused = self.fusion(x_2d, x_3d)
        self.last_features['fused'] = fused
        
        # Store attention weights
        self.last_attention = self.fusion.attn_weights if hasattr(self.fusion, 'attn_weights') else None
        
        # Compute and store scale-aware attention maps
        self.last_scale_attention = self.scale_attention.get_attention(x_2d, x_3d)
        
        # Classification
        out = self.classifier(fused)
        
        return out
    
    def get_intermediate_features(self, x):
        """Get intermediate features from both streams without running full forward pass."""
        x_2d, features_2d = self.stream_2d(x)
        x_3d, features_3d = self.stream_3d(x)
        return {
            '2d': features_2d,
            '3d': features_3d,
            'final_2d': x_2d,
            'final_3d': x_3d
        }
    
    def get_features(self, x):
        """Get intermediate features from both 2D and 3D streams."""
        # Process through 2D stream
        x_2d, features_2d = self.stream_2d(x)
        
        # Process through 3D stream
        x_3d, features_3d = self.stream_3d(x)
        
        return {
            '2d': x_2d,
            '3d': x_3d,
            'features_2d': features_2d,
            'features_3d': features_3d
        }
    
    def get_attention_maps(self):
        """Return attention maps from the last forward pass."""
        if not hasattr(self, 'last_features') or not hasattr(self, 'last_scale_attention'):
            raise RuntimeError("No features or attention maps available. Run forward pass first.")
        
        return {
            'scale_attention': self.last_scale_attention,
            'cross_modal_attention': self.last_attention if hasattr(self, 'last_attention') else None
        }
    
    def get_cross_modal_maps(self):
        """Return cross-modal attention maps and feature relationships."""
        if not hasattr(self, 'last_features') or not hasattr(self, 'last_scale_attention'):
            raise RuntimeError("No features or attention maps available. Run forward pass first.")
        
        # Get cross-modal relationships
        relationships = self.cross_modal_mapper.map_relationships(self.last_scale_attention)
        
        return {
            'attention_weights': self.last_attention if hasattr(self, 'last_attention') else None,
            'features': self.last_features,
            'relationships': relationships
        }
    
    def get_hierarchical_visualization(self, x):
        """Generate hierarchical visualization for input."""
        features = self.get_intermediate_features(x)
        attention_maps = self.scale_attention.get_attention(features['final_2d'], features['final_3d'])
        
        return self.viz_engine.create_visualization(
            attention_maps,
            self.cross_modal_mapper.map_relationships(attention_maps),
            x if x.dim() == 4 else x.squeeze(2),  # Handle 2D input
            x if x.dim() == 5 else x.unsqueeze(2)  # Handle 3D input
        ) 
    