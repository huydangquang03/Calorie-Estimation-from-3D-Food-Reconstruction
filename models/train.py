import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import time
from pathlib import Path
import json
from tqdm import tqdm
import argparse
import warnings
import sys
warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import custom modules
from Data.dataset2 import MetaFood3DDataModule
from models.model2 import (
    FoodCalorieEstimationModelWithClassification, 
    create_model,
    multi_task_loss,
    compute_classification_metrics,
    compute_regression_metrics,
    compute_all_point_cloud_metrics
)

class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

class MetricTracker:
    """Track and compute average of metrics"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics = {}
        self.counts = {}
    
    def update(self, metric_dict):
        for key, value in metric_dict.items():
            if key not in self.metrics:
                self.metrics[key] = 0
                self.counts[key] = 0
            self.metrics[key] += value
            self.counts[key] += 1
    
    def get_average(self):
        return {key: self.metrics[key] / self.counts[key] for key in self.metrics}

class FoodCalorieTrainer:
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: torch.device,
                 save_dir: str = "checkpoints_class",
                 log_dir: str = "logs"):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Create directories
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize tensorboard
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=1e-4, 
            weight_decay=1e-4
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=100, 
            eta_min=1e-6
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=10)
        
        # Move model to device
        self.model.to(device)
        
    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch"""
        self.model.train()
        metric_tracker = MetricTracker()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['image'].to(self.device)
            target_class = batch['class_id'].to(self.device)
            target_calories = batch['calories'].to(self.device)
            target_weight = batch['weight'].to(self.device)
            pointclouds = batch['pointcloud'].to(self.device)
            
            # Forward pass
            outputs = self.model(images, return_point_cloud=True)
            
            # Compute multi-task loss
            losses = multi_task_loss(
                outputs['class_logits'], target_class,
                outputs['calories'], target_calories,
                outputs['weight'], target_weight,
                class_weight=1.0,
                calorie_weight=1.0,
                weight_weight=1.0
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Compute metrics
            class_metrics = compute_classification_metrics(
                outputs['class_logits'], target_class
            )
            calorie_metrics = compute_regression_metrics(
                outputs['calories'], target_calories
            )
            weight_metrics = compute_regression_metrics(
                outputs['weight'], target_weight
            )
            
            # Compute point cloud metrics if available
            try:
                pc_metrics = compute_all_point_cloud_metrics(
                    outputs['point_cloud'], pointclouds
                )
            except:
                pc_metrics = {'chamfer_distance': 0.0, 'earth_movers_distance': 0.0}
            
            # Update metrics
            batch_metrics = {
                'total_loss': losses['total_loss'].item(),
                'class_loss': losses['class_loss'].item(),
                'calorie_loss': losses['calorie_loss'].item(),
                'weight_loss': losses['weight_loss'].item(),
                'accuracy': class_metrics['accuracy'],
                'calorie_mae': calorie_metrics['mae'],
                'weight_mae': weight_metrics['mae'],
                'chamfer_distance': pc_metrics['chamfer_distance']
            }
            
            metric_tracker.update(batch_metrics)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{losses['total_loss'].item():.4f}",
                'Acc': f"{class_metrics['accuracy']:.3f}",
                'Cal_MAE': f"{calorie_metrics['mae']:.2f}",
                'Wt_MAE': f"{weight_metrics['mae']:.2f}"
            })
        
        return metric_tracker.get_average()
    
    def validate_epoch(self, epoch: int) -> dict:
        """Validate for one epoch"""
        self.model.eval()
        metric_tracker = MetricTracker()
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")
        
        with torch.no_grad():
            for batch in pbar:
                # Move data to device
                images = batch['image'].to(self.device)
                target_class = batch['class_id'].to(self.device)
                target_calories = batch['calories'].to(self.device)
                target_weight = batch['weight'].to(self.device)
                pointclouds = batch['pointcloud'].to(self.device)
                
                # Forward pass
                outputs = self.model(images, return_point_cloud=True)
                
                # Compute multi-task loss
                losses = multi_task_loss(
                    outputs['class_logits'], target_class,
                    outputs['calories'], target_calories,
                    outputs['weight'], target_weight,
                    class_weight=1.0,
                    calorie_weight=1.0,
                    weight_weight=1.0
                )
                
                # Compute metrics
                class_metrics = compute_classification_metrics(
                    outputs['class_logits'], target_class
                )
                calorie_metrics = compute_regression_metrics(
                    outputs['calories'], target_calories
                )
                weight_metrics = compute_regression_metrics(
                    outputs['weight'], target_weight
                )
                
                # Compute point cloud metrics if available
                try:
                    pc_metrics = compute_all_point_cloud_metrics(
                        outputs['point_cloud'], pointclouds
                    )
                except:
                    pc_metrics = {'chamfer_distance': 0.0, 'earth_movers_distance': 0.0}
                
                # Update metrics
                batch_metrics = {
                    'total_loss': losses['total_loss'].item(),
                    'class_loss': losses['class_loss'].item(),
                    'calorie_loss': losses['calorie_loss'].item(),
                    'weight_loss': losses['weight_loss'].item(),
                    'accuracy': class_metrics['accuracy'],
                    'calorie_mae': calorie_metrics['mae'],
                    'weight_mae': weight_metrics['mae'],
                    'chamfer_distance': pc_metrics['chamfer_distance']
                }
                
                metric_tracker.update(batch_metrics)
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f"{losses['total_loss'].item():.4f}",
                    'Acc': f"{class_metrics['accuracy']:.3f}",
                    'Cal_MAE': f"{calorie_metrics['mae']:.2f}",
                    'Wt_MAE': f"{weight_metrics['mae']:.2f}"
                })
        
        return metric_tracker.get_average()
    
    def train(self, num_epochs: int, save_every: int = 10):
        """Main training loop"""
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validate epoch
            val_metrics = self.validate_epoch(epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics
            self.log_metrics(train_metrics, val_metrics, epoch)
            
            # Print epoch results
            epoch_time = time.time() - start_time
            self.print_epoch_results(epoch, train_metrics, val_metrics, epoch_time)
            
            # Save best model
            val_loss = val_metrics['total_loss']
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, train_metrics, val_metrics, is_best=True)
            
            # Save periodic checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(epoch, train_metrics, val_metrics, is_best=False)
            
            # Early stopping
            if self.early_stopping(val_loss, self.model):
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        self.writer.close()
        print("Training completed!")
    
    def log_metrics(self, train_metrics: dict, val_metrics: dict, epoch: int):
        """Log metrics to tensorboard"""
        for key, value in train_metrics.items():
            self.writer.add_scalar(f'Train/{key}', value, epoch)
        
        for key, value in val_metrics.items():
            self.writer.add_scalar(f'Val/{key}', value, epoch)
        
        # Log learning rate
        self.writer.add_scalar('Learning_Rate', self.scheduler.get_last_lr()[0], epoch)
    
    def print_epoch_results(self, epoch: int, train_metrics: dict, val_metrics: dict, epoch_time: float):
        """Print epoch results"""
        print(f"\nEpoch {epoch+1} Results ({epoch_time:.1f}s):")
        print(f"Train - Loss: {train_metrics['total_loss']:.4f}, Acc: {train_metrics['accuracy']:.3f}, "
              f"Cal_MAE: {train_metrics['calorie_mae']:.2f}, Wt_MAE: {train_metrics['weight_mae']:.2f}")
        print(f"Val   - Loss: {val_metrics['total_loss']:.4f}, Acc: {val_metrics['accuracy']:.3f}, "
              f"Cal_MAE: {val_metrics['calorie_mae']:.2f}, Wt_MAE: {val_metrics['weight_mae']:.2f}")
    
    def save_checkpoint(self, epoch: int, train_metrics: dict, val_metrics: dict, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'model_config': {
                'num_classes': self.model.num_classes,
                'num_points': self.model.num_points
            }
        }
        
        filename = f"checkpoint_epoch_{epoch+1}.pth"
        if is_best:
            filename = "best_model.pth"
        
        filepath = self.save_dir / filename
        torch.save(checkpoint, filepath)
        
        if is_best:
            print(f"Best model saved at epoch {epoch+1}")

def test_model(model, test_loader, device, class_names):
    """Test the model on test set"""
    model.eval()
    metric_tracker = MetricTracker()
    
    predictions = []
    ground_truths = []
    
    print("Testing model...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            # Move data to device
            images = batch['image'].to(device)
            target_class = batch['class_id'].to(device)
            target_calories = batch['calories'].to(device)
            target_weight = batch['weight'].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Compute losses
            losses = multi_task_loss(
                outputs['class_logits'], target_class,
                outputs['calories'], target_calories,
                outputs['weight'], target_weight
            )
            
            # Compute metrics
            class_metrics = compute_classification_metrics(
                outputs['class_logits'], target_class
            )
            calorie_metrics = compute_regression_metrics(
                outputs['calories'], target_calories
            )
            weight_metrics = compute_regression_metrics(
                outputs['weight'], target_weight
            )
            
            # Store predictions for analysis
            pred_classes = torch.argmax(outputs['class_logits'], dim=1)
            predictions.extend(pred_classes.cpu().numpy())
            ground_truths.extend(target_class.cpu().numpy())
            
            # Update metrics
            batch_metrics = {
                'total_loss': losses['total_loss'].item(),
                'class_loss': losses['class_loss'].item(),
                'calorie_loss': losses['calorie_loss'].item(),
                'weight_loss': losses['weight_loss'].item(),
                'accuracy': class_metrics['accuracy'],
                'calorie_mae': calorie_metrics['mae'],
                'weight_mae': weight_metrics['mae'],
                'calorie_mape': calorie_metrics['mape'],
                'weight_mape': weight_metrics['mape']
            }
            
            metric_tracker.update(batch_metrics)
    
    final_metrics = metric_tracker.get_average()
    
    # Print test results
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    print(f"Overall Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Total Loss: {final_metrics['total_loss']:.4f}")
    print(f"Classification Loss: {final_metrics['class_loss']:.4f}")
    print(f"Calorie MAE: {final_metrics['calorie_mae']:.2f} kcal")
