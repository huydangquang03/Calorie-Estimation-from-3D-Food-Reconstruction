import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
import numpy as np
from typing import Dict

class RGBOnlyCalorieModel(nn.Module):
    
    def __init__(self, 
                 feat_dim: int = 512,
                 hidden_dims: list = [512, 256, 128],
                 dropout: float = 0.3):
        super().__init__()
        self.backbone = resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        layers = []
        prev_dim = 2048  
        
        layers.extend([
            nn.Linear(prev_dim, feat_dim),
            nn.ReLU(),
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(dropout)
        ])
        prev_dim = feat_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.regression_head = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:

        features = self.backbone(images)  # [B, 2048, 1, 1]
        features = features.view(features.size(0), -1)  # [B, 2048]
        calories = self.regression_head(features)  # [B, 1]
        
        return calories.squeeze(1)  # [B]

def mae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)

def mape_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    target_abs = torch.abs(target)
    return torch.mean(torch.abs((pred - target) / (target_abs + eps))) * 100
