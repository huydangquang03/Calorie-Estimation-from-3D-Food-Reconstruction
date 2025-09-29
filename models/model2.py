import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel
import torchvision.models as models
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import warnings
warnings.filterwarnings("ignore")

# ViT Encoder (2D Feature Extraction)
class ViTEncoder(nn.Module):
    """Vision Transformer Encoder for 2D feature extraction"""
    
    def __init__(self, pretrained: bool = True, freeze: bool = False):
        super().__init__()
        
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        
        for param in self.vit.parameters():
            param.requires_grad = not freeze
            
        self.feature_dim = self.vit.config.hidden_size  # 768
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images [B, 3, 224, 224]
        Returns:
            features: [B, 196, 768] (patch tokens only)
        """
        outputs = self.vit(pixel_values=x)
        features = outputs.last_hidden_state[:, 1:, :]  # [B, 196, 768]
        return features

# Contextual Feature Integrator (CFI)
class ContextualFeatureIntegrator(nn.Module):
    """Enhanced CFI Module with 1D Convolution, Feed-Forward, and Multi-Head Attention"""
    
    def __init__(self, embed_dim: int = 768, ff_dim: int = 1024, num_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        
        # 1D Convolutional layer to capture spatial relationships
        self.conv1d = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        
        # Feed-forward network(FFN)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Multi-head attention
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, N, embed_dim]
        Returns:
            enhanced_features: [B, N, embed_dim]
        """
        # Apply 1D convolution to capture spatial relationships
        x = x.permute(0, 2, 1)  # [B, embed_dim, N]
        x = self.conv1d(x)  # [B, embed_dim, N]
        x = x.permute(0, 2, 1)  # [B, N, embed_dim]
        
        # Apply feed-forward network
        x_ff = self.ffn(x)
        
        # Apply multi-head attention (self-attention)
        attn_out, _ = self.mha(x_ff, x_ff, x_ff)
        
        # Residual connection and layer norm
        output = self.layer_norm(attn_out + x_ff)
        
        return output

# Geometric Projection Module (GPM)
class GeometricProjectionModule(nn.Module):
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 2048, num_points: int = 1024):
        super().__init__()
        self.num_points = num_points
        
        # Attention layer for pooling
        self.attention = nn.Linear(input_dim, 1)
        
        # Progressive projection layers
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 4, num_points * 3)  # Output 3D coordinates
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, N, input_dim]
        Returns:
            point_cloud: [B, num_points, 3]
        """
        # Attention-based pooling
        attn_scores = self.attention(x)  # [B, N, 1]
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, N, 1]
        x_pooled = torch.sum(x * attn_weights, dim=1)  # [B, input_dim]
        # Project to 3D coordinates
        points = self.projection(x_pooled)  # [B, num_points * 3]
        # Reshape to point cloud format
        point_cloud = points.view(-1, self.num_points, 3)  # [B, num_points, 3]
        
        return point_cloud

# RGB to Point Cloud Model
class RGB2PointModel(nn.Module):
    """Complete RGB2Point model"""
    
    def __init__(self, 
                 num_points: int = 1024,
                 freeze_vit: bool = True,
                 ff_dim: int = 1024,
                 num_heads: int = 4,
                 hidden_dim: int = 2048):
        super().__init__()
        self.vit_encoder = ViTEncoder(pretrained=True, freeze=freeze_vit)
        self.cfi = ContextualFeatureIntegrator(
            embed_dim=self.vit_encoder.feature_dim,
            ff_dim=ff_dim,
            num_heads=num_heads
        )
        self.gpm = GeometricProjectionModule(
            input_dim=self.vit_encoder.feature_dim,
            hidden_dim=hidden_dim,
            num_points=num_points
        )
        self.num_points = num_points
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        'images: Input RGB images [B, 3, 224, 224] => [B, num_points, 3]'
        #  Extract 2D features using ViT
        features_2d = self.vit_encoder(images)  # [B, 196, 768]
        
        #  Enhance features using CFI
        enhanced_features = self.cfi(features_2d)  # [B, 196, 768]
        
        #  Project to 3D point cloud using GPM
        point_clouds = self.gpm(enhanced_features)  # [B, num_points, 3]
        
        return point_clouds

# 2D Feature Head
class ViT2DFeatureHead(nn.Module):
    def __init__(self, vit_feature_dim: int = 768, out_dim: int = 256):
        super().__init__()
        self.pooler = nn.AdaptiveAvgPool1d(1)
        self.projector = nn.Sequential(
            nn.Linear(vit_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, out_dim)
        )

    def forward(self, vit_features: torch.Tensor) -> torch.Tensor:
        # (B, N, D) -> (B, D, N)
        x = vit_features.transpose(2, 1)
        # Pooling across N (196 patches)
        x_pooled = self.pooler(x).squeeze(-1) # [B, D]
        # Project to final feature space
        features_2d = self.projector(x_pooled) # [B, out_dim]
        return features_2d

# 3D Feature Extractor
class PointNet2Encoder(nn.Module):
    """ PointNet++ for 3D feature extraction"""
    
    def __init__(self, out_dim: int = 256):
        super().__init__()
        # Point-wise MLPs
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        # Global feature extraction
        self.global_conv = nn.Conv1d(256, 512, 1)
        self.global_bn = nn.BatchNorm1d(512)
        # Final projection
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, out_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            x: Input point cloud [B, N, 3]
           output: features: [B, out_dim]
        """
        # Transpose for conv1d: [B, 3, N]
        x = x.transpose(2, 1)
        # Point-wise convolutions
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # Global feature
        x = F.relu(self.global_bn(self.global_conv(x)))
        # Global max pooling
        x = torch.max(x, 2)[0]  # [B, 512]
        # Final projection
        x = self.fc(x)  # [B, out_dim]
        
        return x

class FeatureExtractor3D(nn.Module):
    """3D Feature Extractor"""
    
    def __init__(self, out_dim: int = 256, backbone: str = 'pointnet2'):
        super().__init__()
        
        if backbone == 'pointnet2':
            self.encoder = PointNet2Encoder(out_dim)
        else:
            raise ValueError(f"Unsupported 3D backbone: {backbone}")
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
            Input point cloud [B, N, 3]
            features: [B, out_dim]
        """
        return self.encoder(points)

# Enhanced Fusion Module with Classification
class FusionRegressor(nn.Module):
    """Enhanced Fusion Module with Classification and Regression"""
    
    def __init__(self, 
                 feat_2d_dim: int = 256,
                 feat_3d_dim: int = 256,
                 num_classes: int = 108,  # Number of food categories
                 hidden_dims: list = [512, 256, 128],
                 dropout: float = 0.3):
        super().__init__()
        
        # Input dimension after concatenation
        input_dim = feat_2d_dim + feat_3d_dim
        
        # Shared feature layers
        shared_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.shared_mlp = nn.Sequential(*shared_layers)
        
        # Task-specific heads
        self.classification_head = nn.Linear(prev_dim, num_classes)
        self.calorie_head = nn.Linear(prev_dim, 1)
        self.weight_head = nn.Linear(prev_dim, 1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, f2d: torch.Tensor, f3d: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            f2d: 2D features [B, feat_2d_dim]
            f3d: 3D features [B, feat_3d_dim]
        Returns:
            Dict containing predictions for classification, calories, and weight
        """
        # Concatenate features
        f_combined = torch.cat([f2d, f3d], dim=1)
        
        # Shared feature extraction
        shared_features = self.shared_mlp(f_combined)
        
        # Task-specific predictions
        class_logits = self.classification_head(shared_features)  # [B, num_classes]
        calories = self.calorie_head(shared_features).squeeze(1)  # [B]
        weight = self.weight_head(shared_features).squeeze(1)      # [B]
        
        return {
            'class_logits': class_logits,
            'calories': calories,
            'weight': weight
        }

# Main Multi-task Model
class FoodCalorieEstimationModelWithClassification(nn.Module):
    """Enhanced model with food classification capability"""
    
    def __init__(self,
                 num_points: int = 1024,
                 num_classes: int = 108,  # Number of food categories
                 freeze_vit: bool = False,
                 ff_dim: int = 1024,
                 num_heads: int = 4,
                 hidden_dim: int = 2048,
                 feat_2d_dim: int = 256,
                 feat_3d_dim: int = 256,
                 regression_hidden_dims: list = [512, 256, 128],
                 dropout: float = 0.3):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_points = num_points
        
        self.vit_encoder = ViTEncoder(pretrained=True, freeze=freeze_vit)
        self.cfi = ContextualFeatureIntegrator(
            embed_dim=self.vit_encoder.feature_dim,
            ff_dim=ff_dim,
            num_heads=num_heads
        )
        
        self.gpm = GeometricProjectionModule(
            input_dim=self.vit_encoder.feature_dim,
            hidden_dim=hidden_dim,
            num_points=num_points
        )
        
        self.feature_2d_head = ViT2DFeatureHead(
            vit_feature_dim=self.vit_encoder.feature_dim,
            out_dim=feat_2d_dim
        )
        
        self.feature_3d = FeatureExtractor3D(out_dim=feat_3d_dim)
        
        self.fusion_regressor = FusionRegressor(
            feat_2d_dim=feat_2d_dim,
            feat_3d_dim=feat_3d_dim,
            num_classes=num_classes,
            hidden_dims=regression_hidden_dims,
            dropout=dropout
        )
    
    def forward(self,
                images: torch.Tensor,
                return_point_cloud: bool = False) -> Dict[str, torch.Tensor]:
        
        # Extract ViT features
        features_vit = self.vit_encoder(images)  # [B, 196, 768]
        
        # Enhance features using CFI
        enhanced_features = self.cfi(features_vit)  # [B, 196, 768]
        
        # Generate point cloud
        point_cloud = self.gpm(enhanced_features)  # [B, num_points, 3]
        
        # Extract 2D and 3D features
        f2d = self.feature_2d_head(enhanced_features)  # [B, feat_2d_dim]
        f3d = self.feature_3d(point_cloud)  # [B, feat_3d_dim]
        
        # Fusion and multi-task prediction
        predictions = self.fusion_regressor(f2d, f3d)
        
        output = {
            'class_logits': predictions['class_logits'],
            'calories': predictions['calories'],
            'weight': predictions['weight']
        }
        
        if return_point_cloud:
            output['point_cloud'] = point_cloud
        
        return output

    def get_model_info(self) -> str:
        """Return model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = f"""
Model: FoodCalorieEstimationModelWithClassification
- Number of classes: {self.num_classes}
- Point cloud size: {self.num_points}
- Total parameters: {total_params:,}
- Trainable parameters: {trainable_params:,}
- ViT feature dim: {self.vit_encoder.feature_dim}
"""
        return info

# Loss Functions
def multi_task_loss(pred_class_logits: torch.Tensor, target_class: torch.Tensor,
                   pred_calories: torch.Tensor, target_calories: torch.Tensor,
                   pred_weight: torch.Tensor, target_weight: torch.Tensor,
                   class_weight: float = 1.0, calorie_weight: float = 1.0, 
                   weight_weight: float = 1.0) -> Dict[str, torch.Tensor]:
    """
    Multi-task loss combining classification and regression
    """
    # Classification loss (Cross-entropy)
    class_loss = F.cross_entropy(pred_class_logits, target_class)
    
    # Regression losses (L1 loss)
    calorie_loss = F.l1_loss(pred_calories, target_calories)
    weight_loss = F.l1_loss(pred_weight, target_weight)
    
    # Combined loss
    total_loss = (class_weight * class_loss + 
                  calorie_weight * calorie_loss + 
                  weight_weight * weight_loss)
    
    return {
        'total_loss': total_loss,
        'class_loss': class_loss,
        'calorie_loss': calorie_loss,
        'weight_loss': weight_loss
    }

def compute_classification_metrics(pred_logits: torch.Tensor, 
                                 target_class: torch.Tensor) -> Dict[str, float]:
    """
    Compute classification metrics
    """
    pred_class = torch.argmax(pred_logits, dim=1)
    accuracy = (pred_class == target_class).float().mean().item()
    
    # Convert to numpy for detailed metrics
    pred_np = pred_class.detach().cpu().numpy()
    target_np = target_class.detach().cpu().numpy()
    
    try:
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(target_np, pred_np, average='weighted', zero_division=0)
        recall = recall_score(target_np, pred_np, average='weighted', zero_division=0)
        f1 = f1_score(target_np, pred_np, average='weighted', zero_division=0)
    except ImportError:
        precision = recall = f1 = accuracy  # Fallback if sklearn not available
    except:
        precision = recall = f1 = 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def compute_regression_metrics(pred_values: torch.Tensor, 
                              target_values: torch.Tensor,
                              eps: float = 1e-8) -> Dict[str, float]:
    """
    Compute regression metrics (MAE, MAPE, RMSE)
    """
    mae = F.l1_loss(pred_values, target_values).item()
    mse = F.mse_loss(pred_values, target_values).item()
    rmse = np.sqrt(mse)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = torch.mean(torch.abs((pred_values - target_values) / (torch.abs(target_values) + eps))).item() * 100
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape
    }

# Point Cloud Metrics
def chamfer_distance(pred_pc: torch.Tensor, gt_pc: torch.Tensor) -> torch.Tensor:
    """Compute Chamfer Distance between two point clouds"""
    pred_expanded = pred_pc.unsqueeze(2)  # [B, N, 1, 3]
    gt_expanded = gt_pc.unsqueeze(1)  # [B, 1, M, 3]
    
    distances = torch.sum((pred_expanded - gt_expanded) ** 2, dim=-1)  # [B, N, M]
    
    forward_dist = torch.min(distances, dim=2)[0]  # [B, N]
    forward_loss = torch.mean(forward_dist)
    
    backward_dist = torch.min(distances, dim=1)[0]  # [B, M]
    backward_loss = torch.mean(backward_dist)
    
    chamfer_loss = forward_loss + backward_loss
    return chamfer_loss

def earth_movers_distance(pred_pc: torch.Tensor, gt_pc: torch.Tensor) -> torch.Tensor:
    """Compute Earth Mover's Distance between two point clouds"""
    batch_size = pred_pc.shape[0]
    total_emd = 0.0
    
    for b in range(batch_size):
        pred_points = pred_pc[b].detach().cpu().numpy()
        gt_points = gt_pc[b].detach().cpu().numpy()
        
        distance_matrix = cdist(pred_points, gt_points, metric='euclidean')
        row_indices, col_indices = linear_sum_assignment(distance_matrix)
        emd = distance_matrix[row_indices, col_indices].mean()
        total_emd += emd
    
    return torch.tensor(total_emd / batch_size, dtype=torch.float32)

def compute_all_point_cloud_metrics(pred_pc: torch.Tensor, gt_pc: torch.Tensor) -> Dict[str, float]:
    """Compute all point cloud metrics"""
    cd = chamfer_distance(pred_pc, gt_pc)
    emd = earth_movers_distance(pred_pc, gt_pc)
    return {
        'chamfer_distance': cd.item(),
        'earth_movers_distance': emd.item(),
    }

# Model Factory
def create_model(num_classes: int, **kwargs) -> FoodCalorieEstimationModelWithClassification:
    """Factory function to create model with default parameters"""
    default_params = {
        'num_points': 1024,
        'freeze_vit': False,
        'ff_dim': 1024,
        'num_heads': 4,
        'hidden_dim': 2048,
        'feat_2d_dim': 256,
        'feat_3d_dim': 256,
        'regression_hidden_dims': [512, 256, 128],
        'dropout': 0.3
    }
    
    # Update with any provided kwargs
    default_params.update(kwargs)
    
    return FoodCalorieEstimationModelWithClassification(
        num_classes=num_classes,
        **default_params
    )

if __name__ == "__main__":
    print("Testing Food Calorie Estimation Model with Classification...")
    
    # Test model creation
    num_classes = 108
    model = create_model(num_classes=num_classes)
    
    print(model.get_model_info())
    
    # Test forward pass
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    
    print(f"Input image shape: {images.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(images, return_point_cloud=True)
    
    print(f"\nOutput shapes:")
    print(f"  Classification logits: {output['class_logits'].shape}")
    print(f"  Predicted calories: {output['calories'].shape}")
    print(f"  Predicted weight: {output['weight'].shape}")
    print(f"  Generated point cloud: {output['point_cloud'].shape}")
    
    # Test predictions
    pred_classes = torch.argmax(output['class_logits'], dim=1)
    print(f"\nSample predictions:")
    print(f"  Predicted classes: {pred_classes.tolist()}")
    print(f"  Predicted calories: {output['calories'].tolist()}")
    print(f"  Predicted weights: {output['weight'].tolist()}")
    
    # Test loss computation
    target_class = torch.randint(0, num_classes, (batch_size,))
    target_calories = torch.randn(batch_size) * 100 + 200  # Random calories around 200
    target_weight = torch.randn(batch_size) * 50 + 100     # Random weight around 100g
    
    losses = multi_task_loss(
        output['class_logits'], target_class,
        output['calories'], target_calories,
        output['weight'], target_weight
    )
    
    print(f"\nSample losses:")
    for loss_name, loss_value in losses.items():
        print(f"  {loss_name}: {loss_value.item():.4f}")
    
    # Test metrics
    class_metrics = compute_classification_metrics(output['class_logits'], target_class)
    calorie_metrics = compute_regression_metrics(output['calories'], target_calories)
    weight_metrics = compute_regression_metrics(output['weight'], target_weight)
    
    print(f"\nSample metrics:")
    print(f"  Classification accuracy: {class_metrics['accuracy']:.4f}")
    print(f"  Calorie MAE: {calorie_metrics['mae']:.4f}")
    print(f"  Weight MAE: {weight_metrics['mae']:.4f}")
    
    print("\nModel testing completed successfully!")