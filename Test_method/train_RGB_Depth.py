import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import logging
from tqdm import tqdm
import sys
import signal
import json
from datetime import datetime

from dataset import MetaFood3DRGBDepthDataModule
from model_RGB_Depth import (
    create_model, 
    CalorieLoss, 
    create_rgbd_input
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_mae_mape(predictions: torch.Tensor, targets: torch.Tensor) -> dict:
    with torch.no_grad():
        pred = predictions.cpu().numpy()
        target = targets.cpu().numpy()
        mae = np.mean(np.abs(pred - target))
        epsilon = 1e-8
        mape = np.mean(np.abs((pred - target) / (np.abs(target) + epsilon))) * 100
        
        return {'mae': mae, 'mape': mape}

class RGBDepthTrainer:

    def __init__(self,
                 model,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 nutrition_stats: dict,
                 device: str = 'cuda',
                 lr: float = 1e-4,
                 save_dir: str = './depth_4th_channel_simple'):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.save_dir = save_dir
        self.nutrition_stats = nutrition_stats
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=7
        )
        self.criterion = CalorieLoss(loss_type='mae')
        os.makedirs(save_dir, exist_ok=True)
        self.train_losses = []
        self.val_losses = []
        self.train_maes = []
        self.val_maes = []
        self.train_mapes = []
        self.val_mapes = []
        self.best_val_mae = float('inf')
        self.should_stop = False
        
        self.setup_signal_handlers()
    
    def setup_signal_handlers(self):
        def signal_handler(signum, frame):
            logger.info("\nReceived interrupt signal. Saving checkpoint...")
            self.should_stop = True
        
        signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, signal_handler)
    
    def train_epoch(self) -> dict:
        self.model.train()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(self.train_loader, desc="Training", 
                   bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
        
        for batch_idx, batch in enumerate(pbar):
            if self.should_stop:
                break
            rgb = batch['image'].to(self.device)
            depth = batch['depth'].to(self.device)
            calories = batch['calories'].to(self.device)
            rgbd_input = create_rgbd_input(rgb, depth)
            self.optimizer.zero_grad()
            predictions = self.model(rgbd_input)
            loss = self.criterion(predictions, calories)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            all_predictions.extend(predictions.detach().cpu())
            all_targets.extend(calories.detach().cpu())
            
            pbar.set_postfix({
                'MAE': f'{loss.item():.1f}',
                'Avg': f'{total_loss/(batch_idx+1):.1f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        
        if len(all_predictions) > 0:
            metrics = calculate_mae_mape(
                torch.stack(all_predictions), 
                torch.stack(all_targets)
            )
        else:
            metrics = {'mae': 0, 'mape': 0}
        
        return {
            'loss': avg_loss,
            'mae': metrics['mae'],
            'mape': metrics['mape']
        }
    
    def validate(self) -> dict:
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Validation",
                       bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
            
            for batch_idx, batch in enumerate(pbar):
                if self.should_stop:
                    break
                rgb = batch['image'].to(self.device)
                depth = batch['depth'].to(self.device)
                calories = batch['calories'].to(self.device)
                rgbd_input = create_rgbd_input(rgb, depth)
                
                predictions = self.model(rgbd_input)
                loss = self.criterion(predictions, calories)
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu())
                all_targets.extend(calories.cpu())
                
                pbar.set_postfix({
                    'MAE': f'{loss.item():.1f}',
                    'Avg': f'{total_loss/(batch_idx+1):.1f}'
                })
        
        avg_loss = total_loss / len(self.test_loader)
        
        if len(all_predictions) > 0:
            metrics = calculate_mae_mape(
                torch.stack(all_predictions), 
                torch.stack(all_targets)
            )
        else:
            metrics = {'mae': 0, 'mape': 0}
        
        return {
            'loss': avg_loss,
            'mae': metrics['mae'],
            'mape': metrics['mape']
        }
    
    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_mae': self.best_val_mae,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_maes': self.train_maes,
            'val_maes': self.val_maes,
            'train_mapes': self.train_mapes,
            'val_mapes': self.val_mapes,
            'nutrition_stats': self.nutrition_stats,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        latest_path = os.path.join(self.save_dir, 'latest_depth_4th_channel.pth')
        torch.save(checkpoint, latest_path)
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_depth_4th_channel.pth')
            torch.save(checkpoint, best_path)
            logger.info(f" New best MAE: {metrics['mae']:.2f} kcal")
    
    def train(self, num_epochs: int):
        logger.info(f" Starting training for {num_epochs} epochs")
        logger.info(f" Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        try:
            for epoch in range(num_epochs):
                if self.should_stop:
                    break
                print(f"\n Epoch {epoch + 1}/{num_epochs}")
                print("-" * 50)
                train_metrics = self.train_epoch()
                if self.should_stop:
                    break
                
                val_metrics = self.validate()
                if self.should_stop:
                    break
                old_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step(val_metrics['mae'])
                new_lr = self.optimizer.param_groups[0]['lr']
                
                if new_lr != old_lr:
                    logger.info(f"LR reduced: {old_lr:.2e} -> {new_lr:.2e}")
                
                self.train_losses.append(train_metrics['loss'])
                self.val_losses.append(val_metrics['loss'])
                self.train_maes.append(train_metrics['mae'])
                self.val_maes.append(val_metrics['mae'])
                self.train_mapes.append(train_metrics['mape'])
                self.val_mapes.append(val_metrics['mape'])
                
                is_best = val_metrics['mae'] < self.best_val_mae
                if is_best:
                    self.best_val_mae = val_metrics['mae']
                print(f" Train - MAE: {train_metrics['mae']:6.2f} kcal | MAPE: {train_metrics['mape']:5.1f}%")
                print(f" Val   - MAE: {val_metrics['mae']:6.2f} kcal | MAPE: {val_metrics['mape']:5.1f}%")
                print(f" Best MAE: {self.best_val_mae:.2f} kcal | LR: {new_lr:.2e}")
                
                if is_best:
                    print(f" NEW BEST MODEL!")
                if (epoch + 1) % 5 == 0 or is_best:
                    self.save_checkpoint(epoch, val_metrics, is_best)
                if len(self.val_maes) >= 20:
                    recent_maes = self.val_maes[-15:]
                    if all(mae >= self.best_val_mae for mae in recent_maes):
                        logger.info(" Early stopping: No improvement for 15 epochs")
                        break
        
        except KeyboardInterrupt:
            logger.info(" Training interrupted")
        finally:
            self._save_final_results()
            logger.info(" Training completed!")
    
    def _save_final_results(self):
        results = {
            'model_type': 'Depth_as_4th_Channel_Calorie_Model',
            'paper_reference': 'Nutrition5k',
            'best_val_mae': float(self.best_val_mae),
            'total_epochs': len(self.train_losses),
            'final_train_mae': float(self.train_maes[-1]) if self.train_maes else 0,
            'final_val_mae': float(self.val_maes[-1]) if self.val_maes else 0,
            'final_train_mape': float(self.train_mapes[-1]) if self.train_mapes else 0,
            'final_val_mape': float(self.val_mapes[-1]) if self.val_mapes else 0,
            'paper_comparison': {
                'nutrition5k_depth_4th_channel': '47.6 kcal MAE (18.8% MAPE)',
                'our_result': f'{self.best_val_mae:.2f} kcal MAE'
            },
            'training_history': {
                'train_maes': self.train_maes,
                'val_maes': self.val_maes,
                'train_mapes': self.train_mapes,
                'val_mapes': self.val_mapes
            }
        }
        logger.info("Final Results:")
        logger.info(json.dumps(results, indent=2))
        
        


def main():
    """Main function - simple config"""
    config = {
        'data_root': r"E:\MetaFood3D_Calo",
        'excel_file': r"E:\MetaFood3D_Calo\_MetaFood3D_new_complete_dataset_nutrition_v2 (1).xlsx",
        'batch_size': 8,
        'num_workers': 4,
        'num_epochs': 30,
        'learning_rate': 1e-4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'image_size': 256,
        'dropout': 0.3,
        'save_dir': './depth_4th_channel_simple'
    }
    
    print(" Depth as 4th Channel - Simple Training")
    print("=" * 50)
    print(f"Device: {config['device']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Metrics: MAE + MAPE only")
    print("=" * 50)
    
    logger.info("Loading dataset...")
    data_module = MetaFood3DRGBDepthDataModule(
        root_dir=config['data_root'],
        excel_file=config['excel_file'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        image_size=(config['image_size'], config['image_size']),
        max_points=1024,
        use_depth=True,
        min_match_score=80
    )
    
    train_loader = data_module.get_train_loader()
    test_loader = data_module.get_test_loader()
    nutrition_stats = data_module.get_nutrition_stats()
    
    logger.info("Creating model...")
    model = create_model(pretrained=True, dropout=config['dropout'])
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {total_params:,} parameters")
    
    trainer = RGBDepthTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        nutrition_stats=nutrition_stats,
        device=config['device'],
        lr=config['learning_rate'],
        save_dir=config['save_dir']
    )
    
    trainer.train(num_epochs=config['num_epochs'])
    
    print(f"\nTraining completed!")
    print(f"Best MAE: {trainer.best_val_mae:.2f} kcal")
    print(f"Nutrition5k paper: 47.6 kcal MAE")
    print(f"Saved in: {config['save_dir']}")


if __name__ == "__main__":
    main()
