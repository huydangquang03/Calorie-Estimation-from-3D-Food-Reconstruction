import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from tqdm import tqdm
import sys
import signal
import json
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch.nn.functional as F

sys.path.append('E:/MetaFood3D_Calo')
from Data.Dataset import MetaFood3DDataModule
from rgb_only_model import RGBOnlyCalorieModel, mae_loss, mape_loss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RGBOnlyTrainer:    
    def __init__(self,
                 model: RGBOnlyCalorieModel,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 nutrition_stats: dict,
                 device: str = 'cuda',
                 lr: float = 1e-4,
                 save_dir: str = './rgb_only_checkpoints'):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.save_dir = save_dir
        self.nutrition_stats = nutrition_stats
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=7, verbose=True
        )
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Training 
        self.train_losses = []
        self.val_losses = []
        self.train_maes = []
        self.val_maes = []
        self.best_val_loss = float('inf')
        self.best_mae = float('inf')
        self.start_epoch = 0
        self.should_stop = False
        
        self.setup_signal_handlers()
    
    def setup_signal_handlers(self):
        def signal_handler(signum, frame):
            logger.info("\n Saving checkpoint...")
            self.should_stop = True
        
        signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, signal_handler)
    
    def denormalize_calories(self, normalized_calories: torch.Tensor) -> torch.Tensor:
        """Denormalize calories"""
        mean_cal = self.nutrition_stats['calories']['mean']
        std_cal = self.nutrition_stats['calories']['std']
        return normalized_calories * std_cal + mean_cal
    
    def train_epoch(self) -> dict:
        self.model.train()
        
        total_mae = 0.0
        total_mape = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            if self.should_stop:
                break
            images = batch['image'].to(self.device)
            target_calories = batch['calories'].to(self.device)

            self.optimizer.zero_grad()
            pred_calories_norm = self.model(images)

            pred_calories_denorm = self.denormalize_calories(pred_calories_norm)

            loss = mae_loss(pred_calories_denorm, target_calories)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            with torch.no_grad():
                mae_val = loss.item()
                mape_val = mape_loss(pred_calories_denorm, target_calories).item()
                total_mae += mae_val
                total_mape += mape_val
                num_batches += 1
                pbar.set_postfix({
                    'Loss(MAE)': f'{mae_val:.2f} kcal',
                    'MAPE': f'{mape_val:.2f}%'
                })
        
        avg_mae = total_mae / num_batches if num_batches > 0 else 0
        avg_mape = total_mape / num_batches if num_batches > 0 else 0
        
        return {
            'loss': avg_mae,
            'mae': avg_mae,
            'mape': avg_mape
        }
    
    def validate(self) -> dict:
        self.model.eval()
        
        total_loss = 0.0
        total_mae = 0.0
        total_mape = 0.0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Validation"):
                if self.should_stop:
                    break
                
                images = batch['image'].to(self.device)
                target_calories = batch['calories'].to(self.device)
                
                # Forward pass
                pred_calories_norm = self.model(images)
                pred_calories_denorm = self.denormalize_calories(pred_calories_norm)
                
                # Calculate metrics
                loss_val = F.l1_loss(pred_calories_denorm, target_calories)
                mae_val = loss_val.item()
                
                epsilon = 1e-8
                mape_val = torch.mean(torch.abs((pred_calories_denorm - target_calories) /
                                               (torch.abs(target_calories) + epsilon))) * 100
                
                total_loss += loss_val.item()
                total_mae += mae_val
                total_mape += mape_val.item()
                num_batches += 1
                all_predictions.extend(pred_calories_denorm.cpu().numpy())
                all_targets.extend(target_calories.cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        epsilon = 1e-8
        abs_error = np.abs(all_predictions - all_targets)
        within_10_percent = np.mean(abs_error / (np.abs(all_targets) + epsilon) < 0.1) * 100
        within_20_percent = np.mean(abs_error / (np.abs(all_targets) + epsilon) < 0.2) * 100
        within_50_kcal = np.mean(abs_error < 50) * 100
        rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_mae = total_mae / num_batches if num_batches > 0 else 0
        avg_mape = total_mape / num_batches if num_batches > 0 else 0
        
        return {
            'loss': avg_loss,
            'mae': avg_mae,
            'mape': avg_mape,
            'within_10_percent': within_10_percent,
            'within_20_percent': within_20_percent,
            'within_50_kcal': within_50_kcal,
            'rmse': rmse
        }
    
    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_mae': self.best_mae,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'nutrition_stats': self.nutrition_stats,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(self.save_dir, 'latest_rgb_only.pth')
        torch.save(checkpoint, latest_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_rgb_only.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"New best RGB Val MAE: {metrics['mae']:.2f} kcal")
    
    def train(self, num_epochs: int):
        """Main training loop"""
        logger.info(f" Starting  training  {num_epochs} epochs")
        logger.info(f" Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        try:
            for epoch in range(num_epochs):
                if self.should_stop:
                    break
                logger.info(f" Epoch {epoch + 1}/{num_epochs}")
                train_metrics = self.train_epoch()
                if self.should_stop: 
                    break

                val_metrics = self.validate()
                if self.should_stop: 
                    break

                self.scheduler.step(val_metrics['mae'])

                self.train_losses.append(train_metrics['loss'])
                self.val_losses.append(val_metrics['loss'])
                self.train_maes.append(train_metrics['mae'])
                self.val_maes.append(val_metrics['mae'])
                
                is_best = val_metrics['mae'] < self.best_mae
                if is_best:
                    self.best_mae = val_metrics['mae']
                
                # Log results
                logger.info(f" Train - Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae']:.2f} kcal, MAPE: {train_metrics['mape']:.2f}%")
                logger.info(f" Val - Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.2f} kcal, MAPE: {val_metrics['mape']:.2f}%")
                logger.info(f" Accuracy - Within 10%: {val_metrics['within_10_percent']:.1f}%, Within 20%: {val_metrics['within_20_percent']:.1f}%, Within 50 kcal: {val_metrics['within_50_kcal']:.1f}%")
                
                # Save checkpoint
                self.save_checkpoint(epoch, val_metrics, is_best)
        
        except KeyboardInterrupt:
            logger.info("Training interrupted.")
        finally:
            logger.info("training completed!")

def main():
    # Config
    config = {
        'data_root': "E:/MetaFood3D_Calo",
        'batch_size': 8,
        'num_workers': 4,
        'num_epochs': 40, 
        'learning_rate': 1e-4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'feat_dim': 512,
        'hidden_dims': [512, 256, 128],
        'dropout': 0.3
    }
    
    print("Estimation Training")
    print("=" * 50)
    print(f" Device: {config['device']}")
    print(f" Batch size: {config['batch_size']}")
    print(f" Epochs: {config['num_epochs']}")
    print("=" * 50)
    
    logger.info(" Loading MetaFood3D dataset...")
    data_module = MetaFood3DDataModule(
        root_dir=config['data_root'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    train_loader = data_module.get_train_loader()
    test_loader = data_module.get_test_loader()
    nutrition_stats = data_module.get_nutrition_stats()
    
    logger.info(" Creating RGB Only Model...")
    model = RGBOnlyCalorieModel(
        feat_dim=config['feat_dim'],
        hidden_dims=config['hidden_dims'],
        dropout=config['dropout']
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f" Model parameters: {trainable_params:,} / {total_params:,} trainable")
    
    trainer = RGBOnlyTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        nutrition_stats=nutrition_stats,
        device=config['device'],
        lr=config['learning_rate']
    )
    
    logger.info("Starting training...")
    trainer.train(num_epochs=config['num_epochs'])
    
    logger.info(" successfully!")

if __name__ == "__main__":
    main()
