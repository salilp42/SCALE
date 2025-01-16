import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from interpretability import ScaleAwareAttention, CrossModalFeatureMapper, HierarchicalVizEngine

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
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.1, norm_type='batch', num_groups=8):
        super().__init__()
        
        # Main convolution branch with channel expansion
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        
        # Flexible normalization
        if norm_type == 'group':
            self.norm1 = nn.GroupNorm(num_groups, out_channels)
            self.norm2 = nn.GroupNorm(num_groups, out_channels)
        elif norm_type == 'instance':
            self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
            self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        else:  # default to batch norm
            self.norm1 = nn.BatchNorm3d(out_channels)
            self.norm2 = nn.BatchNorm3d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            shortcut_norm = (nn.GroupNorm(num_groups, out_channels) if norm_type == 'group'
                           else nn.InstanceNorm3d(out_channels, affine=True) if norm_type == 'instance'
                           else nn.BatchNorm3d(out_channels))
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                shortcut_norm
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
        
        # Add augmentation flags
        self.training_augmentation = True  # Can be disabled if needed
        
    def forward(self, x):
        identity = self.shortcut(x)
        
        # Main branch with normalization
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.norm2(self.conv2(out))
        
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
    def __init__(self, in_channels, base_channels=64):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        
        # Initial convolution with input channel handling
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(base_channels)
        
        # Main layers - progressive downsampling
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(base_channels * 2)
        
        # Final convolution to match 3D stream
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(base_channels * 4)
        
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
        
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 64, 28, 28]
        features.append(x)  # cellular level
        
        x = F.relu(self.bn2(self.conv2(x)))  # [B, 128, 14, 14]
        features.append(x)  # tissue level
        
        x = F.relu(self.bn3(self.conv3(x)))  # [B, 256, 7, 7]
        features.append(x)  # organ level
        
        return x, features

class Stream3D(nn.Module):
    def __init__(self, in_channels, base_channels=64, norm_type='batch', use_checkpointing=True):
        super().__init__()
        self.in_channels = in_channels
        self.norm_type = norm_type
        self.use_checkpointing = use_checkpointing
        self.training_augmentation = True
        
        # Initial convolution with increased channels and normalization
        self.conv1 = nn.Conv3d(in_channels, base_channels*2, kernel_size=7, stride=1, padding=3, bias=False)
        if norm_type == 'group':
            self.norm1 = nn.GroupNorm(8, base_channels*2)
        elif norm_type == 'instance':
            self.norm1 = nn.InstanceNorm3d(base_channels*2, affine=True)
        else:
            self.norm1 = nn.BatchNorm3d(base_channels*2)
        
        # Progressive dropout rates (lower in early layers, higher in later layers)
        dropout_rates = [0.05, 0.1, 0.15]
        
        # First stage - high resolution features
        self.stage1 = nn.ModuleList([
            EnhancedResBlock3D(base_channels*2, base_channels*2, dropout_rate=dropout_rates[0], norm_type=norm_type),
            EnhancedResBlock3D(base_channels*2, base_channels*2, dropout_rate=dropout_rates[0], norm_type=norm_type),
            EnhancedResBlock3D(base_channels*2, base_channels*2, dropout_rate=dropout_rates[0], norm_type=norm_type)
        ])
        
        # First transition with increased channels
        self.transition1 = nn.Sequential(
            EnhancedResBlock3D(base_channels*2, base_channels*4, stride=2, dropout_rate=dropout_rates[1], norm_type=norm_type),
            nn.BatchNorm3d(base_channels*4) if norm_type == 'batch' else 
            nn.GroupNorm(8, base_channels*4) if norm_type == 'group' else
            nn.InstanceNorm3d(base_channels*4, affine=True),
            nn.ReLU(inplace=True)
        )
        
        # Second stage - medium resolution
        self.stage2 = nn.ModuleList([
            EnhancedResBlock3D(base_channels*4, base_channels*4, dropout_rate=dropout_rates[1], norm_type=norm_type),
            EnhancedResBlock3D(base_channels*4, base_channels*4, dropout_rate=dropout_rates[1], norm_type=norm_type),
            EnhancedResBlock3D(base_channels*4, base_channels*4, dropout_rate=dropout_rates[1], norm_type=norm_type)
        ])
        
        # Second transition
        self.transition2 = nn.Sequential(
            EnhancedResBlock3D(base_channels*4, base_channels*8, stride=2, dropout_rate=dropout_rates[2], norm_type=norm_type),
            nn.BatchNorm3d(base_channels*8) if norm_type == 'batch' else 
            nn.GroupNorm(8, base_channels*8) if norm_type == 'group' else
            nn.InstanceNorm3d(base_channels*8, affine=True),
            nn.ReLU(inplace=True)
        )
        
        # Final stage - low resolution, high channels
        self.stage3 = nn.ModuleList([
            EnhancedResBlock3D(base_channels*8, base_channels*8, dropout_rate=dropout_rates[2], norm_type=norm_type),
            EnhancedResBlock3D(base_channels*8, base_channels*8, dropout_rate=dropout_rates[2], norm_type=norm_type)
        ])
        
        # Cross-scale attention
        self.cross_scale_attention = nn.Sequential(
            nn.Conv3d(base_channels*8, base_channels*2, kernel_size=1),
            nn.BatchNorm3d(base_channels*2) if norm_type == 'batch' else 
            nn.GroupNorm(8, base_channels*2) if norm_type == 'group' else
            nn.InstanceNorm3d(base_channels*2, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels*2, base_channels*8, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Final 1x1 conv
        self.final_conv = nn.Conv3d(base_channels*8, base_channels*4, kernel_size=1)
        if norm_type == 'group':
            self.final_norm = nn.GroupNorm(8, base_channels*4)
        elif norm_type == 'instance':
            self.final_norm = nn.InstanceNorm3d(base_channels*4, affine=True)
        else:
            self.final_norm = nn.BatchNorm3d(base_channels*4)
    
    def _smart_depth_handling(self, x, target_depth=28):
        """Smart depth handling based on input volume characteristics"""
        current_depth = x.size(2)
        
        # If depth is close to target (±25%), use direct interpolation
        if 0.75 * target_depth <= current_depth <= 1.25 * target_depth:
            return F.interpolate(x, size=(target_depth, x.size(3), x.size(4)), 
                               mode='trilinear', align_corners=False)
        
        # For very thin volumes (depth < target/2), replicate slices
        elif current_depth < target_depth / 2:
            # First interpolate to half target depth
            x = F.interpolate(x, size=(target_depth//2, x.size(3), x.size(4)),
                            mode='trilinear', align_corners=False)
            # Then repeat to reach target depth
            x = x.repeat_interleave(2, dim=2)
            # Adjust if odd target depth
            if x.size(2) != target_depth:
                x = F.interpolate(x, size=(target_depth, x.size(3), x.size(4)),
                                mode='trilinear', align_corners=False)
        
        # For very thick volumes (depth > target*2), use max pooling then interpolation
        elif current_depth > target_depth * 2:
            # First reduce depth by max pooling
            x = F.max_pool3d(x, kernel_size=(2,1,1), stride=(2,1,1))
            # Then interpolate to target depth
            x = F.interpolate(x, size=(target_depth, x.size(3), x.size(4)),
                            mode='trilinear', align_corners=False)
        
        # For other cases, use direct interpolation
        else:
            x = F.interpolate(x, size=(target_depth, x.size(3), x.size(4)),
                            mode='trilinear', align_corners=False)
            
        return x
        
    def _run_stage(self, x, stage_blocks):
        """Helper function to run stage blocks with optional checkpointing"""
        for block in stage_blocks:
            if self.use_checkpointing and self.training:
                x = checkpoint(block, x)
            else:
                x = block(x)
        return x
        
    def _apply_efficient_augmentation(self, x):
        """Apply efficient 3D augmentations during training"""
        if not self.training or not self.training_augmentation:
            return x
            
        B, C, D, H, W = x.shape
        device = x.device
        
        # Random flips (very efficient, no compute overhead)
        if torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[2])  # Depth flip
        if torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[3])  # Height flip
        if torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[4])  # Width flip
            
        # Efficient intensity augmentation (vectorized)
        if self.training and torch.rand(1).item() > 0.5:
            gamma = torch.FloatTensor(1).uniform_(0.8, 1.2).to(device)
            min_val = x.min()
            range_val = x.max() - min_val
            if range_val > 0:
                x_normalized = (x - min_val) / range_val
                x = x_normalized.pow(gamma) * range_val + min_val
        
        return x
        
    def forward(self, x):
        # Handle input dimensions
        if x.dim() == 4:  # [B, C, H, W]
            x = x.unsqueeze(2)  # Add depth dimension [B, C, 1, H, W]
            x = x.repeat(1, 1, 28, 1, 1)  # Repeat along depth dimension
        elif x.dim() == 5:  # [B, C, D, H, W]
            if x.size(2) != 28:  # Smart depth handling
                x = self._smart_depth_handling(x, target_depth=28)
        else:
            raise ValueError(f"Expected 4D or 5D input, got {x.dim()}D")
            
        # Apply efficient augmentations
        x = self._apply_efficient_augmentation(x)
            
        # Handle single channel input
        if x.size(1) == 1 and self.in_channels == 3:
            x = x.repeat(1, 3, 1, 1, 1)
        elif x.size(1) != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {x.size(1)}")
            
        # Store intermediate features for interpretability
        features = []
        
        # Initial convolution with normalization
        x = F.relu(self.norm1(self.conv1(x)))  # [B, 128, 28, 28, 28]
        features.append(x)  # cellular level
        
        # First stage with checkpointing
        x1 = self._run_stage(x, self.stage1)
        features.append(x1)  # tissue level
        
        # First transition and second stage
        x = self.transition1(x1)
        x2 = self._run_stage(x, self.stage2)
        features.append(x2)  # organ level
        
        # Second transition and final stage
        x = self.transition2(x2)
        x3 = self._run_stage(x, self.stage3)
        
        # Apply cross-scale attention
        att = self.cross_scale_attention(x3)
        x = x3 * att
        
        # Final 1x1 conv to match dimensions
        x = F.relu(self.final_norm(self.final_conv(x)))  # [B, 256, 7, 7, 7]
        
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
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        
        # Feature refinement for each modality with gating
        self.refine_2d = FeatureCalibration(dim)
        self.refine_3d = FeatureCalibration(dim)
        
        # Learnable modality weights
        self.modality_gate = nn.Parameter(torch.ones(2))
        
        # Enhanced cross-modal attention
        self.attention = CrossModalAttention(dim, num_heads=num_heads)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Final fusion with residual connection
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        
        # Optional residual connection
        self.residual_alpha = nn.Parameter(torch.zeros(1))
        
    def forward(self, x_2d, x_3d):
        # Ensure inputs have the same shape and expected dimensions
        if x_2d.shape != x_3d.shape:
            raise ValueError(f"Shape mismatch: x_2d {x_2d.shape} vs x_3d {x_3d.shape}")
        
        B, C, H, W = x_2d.shape
        if C != self.dim:
            raise ValueError(f"Expected {self.dim} channels, got {C}")
        
        # Normalize modality weights
        modality_weights = F.softmax(self.modality_gate, dim=0)
        
        # Refine features with gating
        x_2d = self.refine_2d(x_2d) * modality_weights[0]
        x_3d = self.refine_3d(x_3d) * modality_weights[1]
        
        # Store original features for residual
        x_2d_orig = x_2d
        x_3d_orig = x_3d
        
        # Reshape for attention
        x_2d_flat = x_2d.flatten(2).transpose(1, 2)  # [B, 49, 256]
        x_3d_flat = x_3d.flatten(2).transpose(1, 2)  # [B, 49, 256]
        
        # Compute cross-modal attention with dropout
        x_combined = torch.cat([x_2d_flat, x_3d_flat], dim=1)  # [B, 98, 256]
        x_combined = self.dropout(x_combined)
        x_fused, self.attn_weights = self.attention(x_combined)  # [B, 98, 256]
        
        # Extract relevant part of fused features
        x_fused = x_fused[:, :H*W]  # [B, 49, 256]
        
        # Reshape back to spatial dimensions
        x_fused = x_fused.transpose(1, 2).reshape(B, C, H, W)  # [B, 256, 7, 7]
        
        # Concatenate and fuse with residual connection
        x_cat = torch.cat([x_2d, x_fused], dim=1)  # [B, 512, 7, 7]
        out = self.fusion_conv(x_cat)  # [B, 256, 7, 7]
        
        # Add gated residual connection
        residual = x_2d_orig + x_3d_orig
        out = out + self.residual_alpha * residual
        
        return out

class MSMedVision(nn.Module):
    def __init__(self, dataset, norm_type='batch'):
        super().__init__()
        self.n_classes = dataset.n_classes
        self.task_type = dataset.task_type
        
        # Input channels
        in_channels = dataset.n_channels
        
        # Streams
        self.stream_2d = Stream2D(in_channels)  # 2D stream remains unchanged
        self.stream_3d = Stream3D(in_channels, norm_type=norm_type)  # Enhanced 3D stream
        
        # Cross-modal fusion
        self.fusion = CrossModalFusion(256)  # Using base_channels * 4
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, self.n_classes)
        )
        
        # Initialize interpretability engines
        self.scale_attention = ScaleAwareAttention()
        self.cross_modal_mapper = CrossModalFeatureMapper()
        self.viz_engine = HierarchicalVizEngine()
        
        # Store configuration
        self.norm_type = norm_type
        
        # Add inference optimization flags
        self.efficient_inference = False
        self.store_visualization_data = True
        
        # Add TTA configuration
        self.use_tta = False
        self.tta_transforms = ['original', 'flip_d', 'flip_h', 'flip_w']
        
    def set_inference_mode(self, efficient=True, store_viz=True):
        """Configure model for inference.
        
        Args:
            efficient: If True, reduces memory usage by not storing intermediate features
            store_viz: If True, still stores visualization data even in efficient mode
        """
        self.efficient_inference = efficient
        self.store_visualization_data = store_viz
        
        # Propagate to sub-modules
        if hasattr(self.stream_3d, 'training_augmentation'):
            self.stream_3d.training_augmentation = False
            
    def set_tta_mode(self, enabled=True, transforms=None):
        """Configure test-time augmentation.
        
        Args:
            enabled: Whether to use TTA during inference
            transforms: List of transforms to use. Default: ['original', 'flip_d', 'flip_h', 'flip_w']
        """
        self.use_tta = enabled
        if transforms is not None:
            self.tta_transforms = transforms
            
    def _apply_tta_transform(self, x, transform_name):
        """Apply a single TTA transform."""
        if transform_name == 'original':
            return x
        elif transform_name == 'flip_d' and x.dim() == 5:
            return torch.flip(x, dims=[2])
        elif transform_name == 'flip_h':
            return torch.flip(x, dims=[-2])
        elif transform_name == 'flip_w':
            return torch.flip(x, dims=[-1])
        return x
        
    def _reverse_tta_transform(self, output, transform_name):
        """Reverse a TTA transform for the output if needed."""
        # For classification, no reversal needed as we're averaging probabilities
        return output
        
    def forward_tta(self, x):
        """Forward pass with test-time augmentation."""
        if not self.use_tta or self.training:
            return self.forward(x)
            
        outputs = []
        with torch.no_grad():
            for transform in self.tta_transforms:
                # Apply transform
                x_transformed = self._apply_tta_transform(x, transform)
                
                # Get prediction
                output = self.forward(x_transformed)
                
                # Reverse transform if needed
                output = self._reverse_tta_transform(output, transform)
                
                outputs.append(output)
        
        # Average predictions
        if self.task_type == 'multi-label, binary-class':
            return torch.stack(outputs).mean(0)
        else:
            # For regular classification, average logits
            return torch.stack(outputs).mean(0)
            
    def forward(self, x):
        # Use TTA during inference if enabled
        if self.use_tta and not self.training:
            return self.forward_tta(x)
            
        # Get features from both streams
        if self.efficient_inference and not self.training:
            with torch.no_grad():
                x_2d, features_2d = self.stream_2d(x)
                x_3d, features_3d = self.stream_3d(x)
                
                # Only store features if needed for visualization
                if self.store_visualization_data:
                    self.last_features = {
                        '2d': features_2d,
                        '3d': features_3d,
                        'final_2d': x_2d,
                        'final_3d': x_3d
                    }
                
                # Fuse features
                fused = self.fusion(x_2d, x_3d)
                
                if self.store_visualization_data:
                    self.last_features['fused'] = fused
                    self.last_attention = self.fusion.attn_weights if hasattr(self.fusion, 'attn_weights') else None
                    self.last_scale_attention = self.scale_attention.get_attention(x_2d, x_3d)
                
                # Classification
                out = self.classifier(fused)
                
                # Clear unnecessary stored features
                if not self.store_visualization_data:
                    del features_2d, features_3d, x_2d, x_3d, fused
                    
                return out
        else:
            # Regular forward pass for training
            x_2d, features_2d = self.stream_2d(x)
            x_3d, features_3d = self.stream_3d(x)
            
            # Store features for visualization if needed
            if self.store_visualization_data:
                self.last_features = {
                    '2d': features_2d,
                    '3d': features_3d,
                    'final_2d': x_2d,
                    'final_3d': x_3d
                }
            
            # Fuse features
            fused = self.fusion(x_2d, x_3d)
            
            if self.store_visualization_data:
                self.last_features['fused'] = fused
                self.last_attention = self.fusion.attn_weights if hasattr(self.fusion, 'attn_weights') else None
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
        
        # Ensure proper dimensions for 3D features
        if '3d' in str(self.__class__).lower():
            # Add batch dimension if missing
            if len(features_3d) > 0 and features_3d[-1].dim() == 4:
                features_3d[-1] = features_3d[-1].unsqueeze(0)
            # Ensure 5D format [B, C, D, H, W]
            if x_3d.dim() == 4:
                x_3d = x_3d.unsqueeze(2)  # Add depth dimension
        
        return {
            '2d': x_2d,
            '3d': x_3d,
            'features_2d': features_2d,
            'features_3d': features_3d
        }
    
    def get_attention_maps(self, x):
        """Get attention maps and feature correlations for visualization."""
        attention_maps = {}
        feature_correlations = {}
        
        # Get features from both streams
        features = self.get_features(x)
        
        # Get 2D features
        feat_2d = features.get('features_2d', [])
        if feat_2d:
            feat_2d = feat_2d[-1]  # Use last layer features
        
        # Get 3D features
        feat_3d = features.get('features_3d', [])
        if feat_3d:
            feat_3d = feat_3d[-1]  # Use last layer features
        
        if feat_2d is not None and feat_3d is not None:
            # Initialize attention engines
            scale_attention = ScaleAwareAttention()
            cross_modal = CrossModalFeatureMapper()
            
            # Get attention maps
            attention_maps = scale_attention.get_attention(feat_2d, feat_3d)
            
            # Get feature correlations
            feature_correlations = cross_modal.map_relationships(attention_maps)
        
        return attention_maps, feature_correlations
    
    def get_cross_modal_maps(self, x):
        """Return cross-modal attention maps and feature relationships."""
        # Run forward pass if not already done
        if not hasattr(self, 'last_features'):
            _ = self(x)
        
        # Get features from last forward pass
        final_2d = self.last_features.get('final_2d')
        final_3d = self.last_features.get('final_3d')
        
        if final_2d is None or final_3d is None:
            return None
        
        return {
            'attention_weights': self.last_attention if hasattr(self, 'last_attention') else None,
            'features': self.last_features,
            'relationships': self.cross_modal_mapper.map_relationships(self.last_scale_attention) if hasattr(self, 'last_scale_attention') else None
        }
    
    def get_hierarchical_visualization(self, x):
        """Get hierarchical visualization of features and attention."""
        attention_maps, feature_correlations = self.get_attention_maps(x)
        return self.viz_engine.create_visualization(
            attention_maps,
            feature_correlations,
            x if x.dim() == 4 else None,  # 2D image
            x if x.dim() == 5 else None   # 3D volume
        ) 
    