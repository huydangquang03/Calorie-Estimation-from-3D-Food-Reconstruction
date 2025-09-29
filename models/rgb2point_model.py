import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel
import torchvision.models as models
import numpy as np
from typing import Dict
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

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
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
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
        # self.simple_projection = nn.Sequential(
        #     nn.Linear(768, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, num_points * 3),
        # self.num_points = num_points
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        'images: Input RGB images [B, 3, 224, 224] => [B, num_points, 3]'
        #  Extract 2D features using ViT
        features_2d = self.vit_encoder(images)  # [B, 196, 768]
        
        #  Enhance features using CFI
        enhanced_features = self.cfi(features_2d)  # [B, 196, 768]
        
        #  Project to 3D point cloud using GPM
        point_clouds = self.gpm(enhanced_features)  # [B, num_points, 3]
        
        return point_clouds

#  Feature Extraction (2D & 3D )

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
        # Pooling trên chiều N (196 patches)
        x_pooled = self.pooler(x).squeeze(-1) # [B, D]
        # Chiếu xuống không gian đặc trưng cuối cùng
        features_2d = self.projector(x_pooled) # [B, out_dim]
        return features_2d

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

# Fusion & Regression

class FusionRegressor(nn.Module):
    """Fusion and Regression Module for Calorie Estimation"""
    
    def __init__(self, 
                 feat_2d_dim: int = 256,
                 feat_3d_dim: int = 256,
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
        
        # Separate heads for calories and weight
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
            Dict containing predicted calories and weight [B]
        """
        # Concatenate features
        f_combined = torch.cat([f2d, f3d], dim=1)
        
        # Shared feature extraction
        shared_features = self.shared_mlp(f_combined)
        
        # Separate predictions
        calories = self.calorie_head(shared_features).squeeze(1)  # [B]
        weight = self.weight_head(shared_features).squeeze(1)      # [B]
        
        return {
            'calories': calories,
            'weight': weight
        }

#  END-TO-END MODEL

class FoodCalorieEstimationModel(nn.Module):
    def __init__(self,
                 num_points: int = 1024,
                 freeze_vit: bool = False,
                 ff_dim: int = 1024,
                 num_heads: int = 4,
                 hidden_dim: int = 2048,
                 feat_2d_dim: int = 256,
                 feat_3d_dim: int = 256,
                 regression_hidden_dims: list = [512, 256, 128],
                 dropout: float = 0.3):
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
        
        self.feature_2d_head = ViT2DFeatureHead(
            vit_feature_dim=self.vit_encoder.feature_dim,
            out_dim=feat_2d_dim
        )
        
        self.feature_3d = FeatureExtractor3D(out_dim=feat_3d_dim)
        
        self.fusion_regressor = FusionRegressor(
            feat_2d_dim=feat_2d_dim,
            feat_3d_dim=feat_3d_dim,
            hidden_dims=regression_hidden_dims,
            dropout=dropout
        )
        
        self.num_points = num_points
    
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
        
        # Fusion and regression
        predictions = self.fusion_regressor(f2d, f3d)
        
        output = {
            'calories': predictions['calories'],
            'weight': predictions['weight']
        }
        
        if return_point_cloud:
            output['point_cloud'] = point_cloud
        
        return output
# LOSS FUNCTIONS
def combined_loss(pred_calories: torch.Tensor, target_calories: torch.Tensor,
                  pred_weight: torch.Tensor, target_weight: torch.Tensor,
                  calorie_weight: float = 1.0, weight_weight: float = 1.0) -> torch.Tensor:
    calorie_loss = F.l1_loss(pred_calories, target_calories)
    weight_loss = F.l1_loss(pred_weight, target_weight)
    
    total_loss = calorie_weight * calorie_loss + weight_weight * weight_loss
    return total_loss, calorie_loss, weight_loss

def combined_mape_loss(pred_calories: torch.Tensor, target_calories: torch.Tensor,
                       pred_weight: torch.Tensor, target_weight: torch.Tensor,
                       eps: float = 1e-8) -> Dict[str, torch.Tensor]:
    calorie_mape = torch.mean(torch.abs((pred_calories - target_calories) / (torch.abs(target_calories) + eps))) * 100
    weight_mape = torch.mean(torch.abs((pred_weight - target_weight) / (torch.abs(target_weight) + eps))) * 100
    
    return {
        'calorie_mape': calorie_mape,
        'weight_mape': weight_mape
    }

# POINT CLOUD METRICS 
def chamfer_distance(pred_pc: torch.Tensor, gt_pc: torch.Tensor) -> torch.Tensor:
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
    batch_size = pred_pc.shape[0]
    total_emd = 0.0
    
    for b in range(batch_size):
        pred_points = pred_pc[b].detach().cpu().numpy()#nograd
        gt_points = gt_pc[b].detach().cpu().numpy()
        
        distance_matrix = cdist(pred_points, gt_points, metric='euclidean')#[N,M]
        row_indices, col_indices = linear_sum_assignment(distance_matrix) #Hungarian algorithm
        emd = distance_matrix[row_indices, col_indices].mean()
        total_emd += emd
    
    return torch.tensor(total_emd / batch_size, dtype=torch.float32)

def compute_all_metrics(pred_pc: torch.Tensor, gt_pc: torch.Tensor) -> dict:
    cd = chamfer_distance(pred_pc, gt_pc)
    emd = earth_movers_distance(pred_pc, gt_pc)
    return {
        'chamfer_distance': cd.item(),
        'earth_movers_distance': emd.item(),
       
    }
if __name__ == "__main__":
    model = FoodCalorieEstimationModel(
        num_points=1024,
        feat_2d_dim=256,
        feat_3d_dim=256,
        regression_hidden_dims=[512, 256, 128]
    )
    batch_size = 8
    images = torch.randn(batch_size, 3, 224, 224)
    output = model(images, return_point_cloud=True)
    print(f"Predicted calories shape: {output['calories'].shape}")
    print(f"Generated point cloud shape: {output['point_cloud'].shape}")
    print(f"Sample calorie predictions: {output['calories'][:3]}")
    print(f"Sample weight predictions: {output['weight'][:3]}")
    
