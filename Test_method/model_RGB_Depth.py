import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights
import numpy as np
from typing import Dict

class DepthAs4thChannelCalorieModel(nn.Module):

    def __init__(
        self,
        dropout: float = 0.3,
        pretrained: bool = True,
        input_size: int = 256
    ):
        super(DepthAs4thChannelCalorieModel, self).__init__()
        
        self.input_size = input_size
        weights = Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
        
        self.backbone = inception_v3(weights=None, init_weights=False) 
        
        self.backbone.aux_logits = False
        self.backbone.AuxLogits = None 
        original_conv = self.backbone.Conv2d_1a_3x3.conv
        self.backbone.Conv2d_1a_3x3.conv = nn.Conv2d(
            in_channels=4,  
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        self._initialize_weights(pretrained)
        feature_dim = self.backbone.fc.in_features  
        self.backbone.fc = nn.Identity()
        
        self.calorie_head = nn.Sequential(
            nn.Linear(feature_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(4096, 1) 
        )
    
    def _initialize_weights(self, pretrained: bool):
        with torch.no_grad():
            if pretrained:
                pretrained_model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
                pretrained_model.aux_logits = False
                pretrained_model.AuxLogits = None
                pretrained_dict = pretrained_model.state_dict()
                model_dict = self.backbone.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                                 if k in model_dict and k != 'Conv2d_1a_3x3.conv.weight'}
                model_dict.update(pretrained_dict)
                self.backbone.load_state_dict(model_dict, strict=False)
                pretrained_conv_weight = pretrained_model.Conv2d_1a_3x3.conv.weight
                self.backbone.Conv2d_1a_3x3.conv.weight[:, :3] = pretrained_conv_weight
                self.backbone.Conv2d_1a_3x3.conv.weight[:, 3:] = 0.01 * torch.randn_like(
                    self.backbone.Conv2d_1a_3x3.conv.weight[:, 3:]
                )
            else:
                nn.init.kaiming_normal_(self.backbone.Conv2d_1a_3x3.conv.weight, mode='fan_out')
    
    def forward(self, rgbd_input: torch.Tensor) -> torch.Tensor:
        features = self.backbone(rgbd_input) 
        calories = self.calorie_head(features).squeeze(-1) 
        
        return calories


class CalorieLoss(nn.Module):
    
    def __init__(self, loss_type: str = 'mae'):
        super(CalorieLoss, self).__init__()
        
        self.criterion = nn.L1Loss()  
        
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(predictions, targets)


def calculate_calorie_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        pred = predictions.cpu().numpy()
        target = targets.cpu().numpy()
        mae = np.mean(np.abs(pred - target))
        mask = target != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((pred[mask] - target[mask]) / target[mask]) * 100)
        else:
            mape = 0.0
        rmse = np.sqrt(np.mean((pred - target) ** 2))
        correlation = np.corrcoef(pred, target)[0, 1] if len(pred) > 1 else 0.0
        
        return {
            'mae': mae,
            'mape': mape,
            'rmse': rmse,
            'correlation': correlation
        }


def create_rgbd_input(rgb_tensor: torch.Tensor, depth_tensor: torch.Tensor) -> torch.Tensor:

    return torch.cat([rgb_tensor, depth_tensor], dim=1)


def create_model(pretrained: bool = True, dropout: float = 0.3) -> DepthAs4thChannelCalorieModel:

    return DepthAs4thChannelCalorieModel(
        dropout=dropout,
        pretrained=pretrained,
        input_size=256 
    )


if __name__ == "__main__":
    """Test the Depth as 4th Channel model"""
    
    print(" Testing Depth as 4th Channel Calorie Model")
    print(" Based on Nutrition5k paper")
    
    batch_size = 4
    height, width = 256, 256 
    
    rgb_input = torch.randn(batch_size, 3, height, width)    
    depth_input = torch.randn(batch_size, 1, height, width) 
    calorie_targets = torch.randn(batch_size) * 100 + 200   
    rgbd_input = create_rgbd_input(rgb_input, depth_input)
    
    print(f"\n Model Input/Output:")
    print(f"  RGB: {rgb_input.shape}")
    print(f"  Depth: {depth_input.shape}")
    print(f"  RGBD: {rgbd_input.shape}")
    print(f"  Target calories: {calorie_targets.shape}")
    
    print(f"\n Creating Model...")
    model = create_model(pretrained=True, dropout=0.3)
    print(f" Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    print(f"\n Forward Pass...")
    model.eval() 
    with torch.no_grad():
        calorie_predictions = model(rgbd_input)
    
    print(f" SUCCESS! Forward pass completed")
    print(f" Predictions shape: {calorie_predictions.shape}")
    print(f" Sample predictions: {[f'{x:.2f}' for x in calorie_predictions.tolist()]}")
    print(f"\n Testing Loss & Metrics...")
    loss_fn = CalorieLoss(loss_type='mae')
    loss = loss_fn(calorie_predictions, calorie_targets)
    print(f" Loss (MAE): {loss.item():.4f} kcal")
    metrics = calculate_calorie_metrics(calorie_predictions, calorie_targets)
    print(f"\n Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
 
    print(f"\n Testing with different input sizes...")
    test_sizes = [(224, 224), (299, 299)] 
    
    for h, w in test_sizes:
        try:
            test_rgb = torch.randn(2, 3, h, w)
            test_depth = torch.randn(2, 1, h, w)
            test_rgbd = create_rgbd_input(test_rgb, test_depth)
            
            with torch.no_grad():
                test_pred = model(test_rgbd)
            
            print(f"✅ Size {h}x{w}: {test_pred.shape}")
        except Exception as e:
            print(f" Size {h}x{w}: {str(e)}")
