import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import logging
from tqdm import tqdm
import wandb
import sys
import signal
import traceback
from datetime import datetime
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.rgb2point_model import FoodCalorieEstimationModel,combined_loss,combined_mape_loss
from Data.Dataset import MetaFood3DDataModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EndToEndCalorieTrainer:
    def __init__(self,
                 model: FoodCalorieEstimationModel,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 nutrition_stats: dict,
                 device: str = 'cuda',
                 lr: float = 1e-4,
                 vit_lr: float = 1e-6,
                 feature_lr: float = 1e-5,
                 calorie_weight: float = 1.0,
                 weight_weight: float = 1.0,
                 save_dir: str = './calorie_checkpoints'):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.save_dir = save_dir
        self.calorie_weight = calorie_weight
        self.weight_weight = weight_weight
        
        # Optimizer
        self.optimizer = optim.Adam([
            {'params': self.model.vit_encoder.parameters(), 'lr': vit_lr},
            {'params': self.model.cfi.parameters(), 'lr': lr},
            {'params': self.model.gpm.parameters(), 'lr': lr},
            {'params': self.model.feature_2d_head.parameters(), 'lr': feature_lr},
            {'params': self.model.feature_3d.parameters(), 'lr': feature_lr},
            {'params': self.model.fusion_regressor.parameters(), 'lr': lr * 2}
        ])
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Training tracking
        self.train_losses = []
        self.val_losses = []
        self.train_maes = []
        self.val_maes = []
        self.train_weight_maes = []
        self.val_weight_maes = []
        self.best_val_loss = float('inf')
        self.best_mae = float('inf')
        self.best_weight_mae = float('inf')
        self.best_metrics = None
        self.best_epoch = 0
        self.start_epoch = 0
        self.should_stop = False
        
        self.nutrition_stats = nutrition_stats
        self.setup_signal_handlers()
    def setup_signal_handlers(self):
        def signal_handler(signum, frame):
            logger.info("\n Saving checkpoint...")
            self.should_stop = True
        
        signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, signal_handler)
            
    def denormalize_calories(self, normalized_calories: torch.Tensor) -> torch.Tensor:
        """Denormalize calories."""
        mean_cal = self.nutrition_stats['calories']['mean']
        std_cal = self.nutrition_stats['calories']['std']
        return normalized_calories * std_cal + mean_cal
    
    def denormalize_weight(self, normalized_weight: torch.Tensor) -> torch.Tensor:
        """Denormalize weight."""
        mean_weight = self.nutrition_stats['weight']['mean']
        std_weight = self.nutrition_stats['weight']['std']
        return normalized_weight * std_weight + mean_weight
    
    def train_epoch(self) -> dict:
        self.model.train()
        total_loss = 0.0
        total_calorie_mae = 0.0
        total_weight_mae = 0.0
        total_calorie_mape = 0.0
        total_weight_mape = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            if self.should_stop:
                break
            
            images = batch['image'].to(self.device)
            target_calories = batch['calories'].to(self.device)
            target_weight = batch['weight'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(images, return_point_cloud=False)
            pred_calories_norm = output['calories']
            pred_weight_norm = output['weight']
            
            # Denormalize predictions
            pred_calories_denorm = self.denormalize_calories(pred_calories_norm)
            pred_weight_denorm = self.denormalize_weight(pred_weight_norm)
            
            # Calculate combined loss
            total_loss_val, calorie_loss, weight_loss = combined_loss(
                pred_calories_denorm, target_calories,
                pred_weight_denorm, target_weight,
                self.calorie_weight, self.weight_weight
            )
            
            # Backward pass
            total_loss_val.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Compute metrics
            with torch.no_grad():
                mape_metrics = combined_mape_loss(
                    pred_calories_denorm, target_calories,
                    pred_weight_denorm, target_weight
                )
                
                total_loss += total_loss_val.item()
                total_calorie_mae += calorie_loss.item()
                total_weight_mae += weight_loss.item()
                total_calorie_mape += mape_metrics['calorie_mape'].item()
                total_weight_mape += mape_metrics['weight_mape'].item()
                num_batches += 1
                
                pbar.set_postfix({
                    'Loss': f'{total_loss_val.item():.2f}',
                    'Cal_MAE': f'{calorie_loss.item():.2f}',
                    'Wt_MAE': f'{weight_loss.item():.2f}'
                })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_calorie_mae = total_calorie_mae / num_batches if num_batches > 0 else 0
        avg_weight_mae = total_weight_mae / num_batches if num_batches > 0 else 0
        avg_calorie_mape = total_calorie_mape / num_batches if num_batches > 0 else 0
        avg_weight_mape = total_weight_mape / num_batches if num_batches > 0 else 0
        
        return {
            'loss': avg_loss,
            'calorie_mae': avg_calorie_mae,
            'weight_mae': avg_weight_mae,
            'calorie_mape': avg_calorie_mape,
            'weight_mape': avg_weight_mape
        }
    
    def validate(self) -> dict:
        self.model.eval()
        total_loss = 0.0
        total_calorie_mae = 0.0
        total_weight_mae = 0.0
        total_calorie_mape = 0.0
        total_weight_mape = 0.0
        num_batches = 0
        
        all_calorie_predictions = []
        all_weight_predictions = []
        all_calorie_targets = []
        all_weight_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Validation"):
                if self.should_stop:
                    break
                
                images = batch['image'].to(self.device)
                target_calories = batch['calories'].to(self.device)
                target_weight = batch['weight'].to(self.device)
                
                # Forward pass
                output = self.model(images, return_point_cloud=False)
                pred_calories_norm = output['calories']
                pred_weight_norm = output['weight']
                
                # Denormalize predictions
                pred_calories_denorm = self.denormalize_calories(pred_calories_norm)
                pred_weight_denorm = self.denormalize_weight(pred_weight_norm)
                
                # Calculate losses
                total_loss_val, calorie_loss, weight_loss = combined_loss(
                    pred_calories_denorm, target_calories,
                    pred_weight_denorm, target_weight,
                    self.calorie_weight, self.weight_weight
                )
                
                mape_metrics = combined_mape_loss(
                    pred_calories_denorm, target_calories,
                    pred_weight_denorm, target_weight
                )
                
                total_loss += total_loss_val.item()
                total_calorie_mae += calorie_loss.item()
                total_weight_mae += weight_loss.item()
                total_calorie_mape += mape_metrics['calorie_mape'].item()
                total_weight_mape += mape_metrics['weight_mape'].item()
                num_batches += 1
                
                # Collect predictions for analysis
                all_calorie_predictions.extend(pred_calories_denorm.cpu().numpy())
                all_weight_predictions.extend(pred_weight_denorm.cpu().numpy())
                all_calorie_targets.extend(target_calories.cpu().numpy())
                all_weight_targets.extend(target_weight.cpu().numpy())
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_calorie_mae = total_calorie_mae / num_batches if num_batches > 0 else 0
        avg_weight_mae = total_weight_mae / num_batches if num_batches > 0 else 0
        avg_calorie_mape = total_calorie_mape / num_batches if num_batches > 0 else 0
        avg_weight_mape = total_weight_mape / num_batches if num_batches > 0 else 0
        
        return {
            'loss': avg_loss,
            'calorie_mae': avg_calorie_mae,
            'weight_mae': avg_weight_mae,
            'calorie_mape': avg_calorie_mape,
            'weight_mape': avg_weight_mape
        }

    
    
    def load_checkpoint(self, checkpoint_path: str = None):
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.save_dir, 'latest_calorie.pth')
        
        if not os.path.exists(checkpoint_path):
            logger.info(f" No checkpoint found at {checkpoint_path}")
            return False
        
        try:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Load model and optimizer states
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load epoch and best metrics
            self.start_epoch = checkpoint['epoch'] + 1  
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.best_mae = checkpoint.get('best_mae', float('inf'))
            self.best_weight_mae = checkpoint.get('best_weight_mae', float('inf'))
            
            # Load training history
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.train_maes = checkpoint.get('train_maes', [])
            self.val_maes = checkpoint.get('val_maes', [])
            self.train_weight_maes = checkpoint.get('train_weight_maes', [])
            self.val_weight_maes = checkpoint.get('val_weight_maes', [])
            
            # Load nutrition stats
            self.nutrition_stats = checkpoint.get('nutrition_stats', None)
            
            logger.info(f" Checkpoint loaded successfully!")
            logger.info(f" Resuming from epoch {self.start_epoch} (last completed epoch: {checkpoint['epoch']})")
            logger.info(f" Best Calorie MAE so far: {self.best_mae:.2f} kcal")
            logger.info(f" Best Weight MAE so far: {self.best_weight_mae:.2f} g")
            return True
        except Exception as e:
            logger.error(f" Error loading checkpoint: {e}")
            traceback.print_exc()
            return False
    def load_rgb2point_checkpoint(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            logger.warning(f" RGB2Point checkpoint not found: {checkpoint_path}")
            return False
        
        try:
            logger.info(f" Loading RGB2Point checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model_state = checkpoint['model_state_dict']
            rgb2point_state = {}
            for key, value in model_state.items():
                if key.startswith(('vit_encoder.', 'cfi.', 'gpm.')):
                    rgb2point_state[key] = value
            
            self.model.load_state_dict(rgb2point_state, strict=False)
            logger.info(" RGB2Point weights loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f" Error loading RGB2Point checkpoint: {e}")
            return False     
          
    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        checkpoint = {
            'epoch': epoch, 
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_mae': self.best_mae,
            # ===== THÊM WEIGHT METRICS =====
            'best_weight_mae': self.best_weight_mae,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_maes': self.train_maes,
            'val_maes': self.val_maes,
            # ===== THÊM WEIGHT TRAINING HISTORY =====
            'train_weight_maes': self.train_weight_maes,
            'val_weight_maes': self.val_weight_maes,
            'nutrition_stats': self.nutrition_stats,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(self.save_dir, 'latest_calorie.pth')
        torch.save(checkpoint, latest_path)
        logger.info(f" Saved latest checkpoint: epoch {epoch + 1}")
        
        # Save best model
        if is_best:
            best_checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'nutrition_stats': self.nutrition_stats,
                'metrics': metrics,
                'best_val_loss': self.best_val_loss,
                'best_mae': self.best_mae,
                # ===== THÊM WEIGHT METRICS =====
                'best_weight_mae': self.best_weight_mae,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_maes': self.train_maes,
                'val_maes': self.val_maes,
                # ===== THÊM WEIGHT TRAINING HISTORY =====
                'train_weight_maes': self.train_weight_maes,
                'val_weight_maes': self.val_weight_maes,
                'timestamp': datetime.now().isoformat()
            }
            best_path = os.path.join(self.save_dir, 'best_calorie.pth')
            torch.save(best_checkpoint, best_path)
            # ===== SỬA LOG MESSAGE =====
            logger.info(f" New best model saved - Cal MAE: {metrics['calorie_mae']:.2f} kcal, Wt MAE: {metrics['weight_mae']:.2f} g")

        # Save epoch checkpoints every 10 epochs
        if (epoch + 1) % 10 == 0: 
            epoch_path = os.path.join(self.save_dir, f'epoch_{epoch + 1}_calorie.pth')
            torch.save(checkpoint, epoch_path)
            logger.info(f" Saved epoch checkpoint: {epoch_path}")


    
    def train(self, num_epochs: int, resume: bool = True, rgb2point_checkpoint: str = None):
        resumed_successfully = False
        if resume:
            logger.info(" Attempting to resume.")
            resumed_successfully = self.load_checkpoint() 
            if not resumed_successfully:
                logger.warning(" Resume failed.")
        if not resumed_successfully:
            self.start_epoch = 0 
            if rgb2point_checkpoint and os.path.exists(rgb2point_checkpoint):
                self.load_rgb2point_checkpoint(rgb2point_checkpoint)
            else:
                logger.warning(" No checkpoint found.")
        if not self.nutrition_stats:
            logger.error(" Nutrition missing.")
            return 

        logger.info(f" Starting calorie estimation training from epoch {self.start_epoch + 1}")
        logger.info(f" Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        try:
            for epoch in range(self.start_epoch, num_epochs):
                if self.should_stop:
                    break
                
                logger.info(f" Epoch {epoch + 1}/{num_epochs}")
                
                train_metrics = self.train_epoch()
                if self.should_stop: 
                    break
                
                val_metrics = self.validate()
                if self.should_stop: 
                    break
                
                # Use combined loss for scheduler
                self.scheduler.step(val_metrics['loss'])
                
                # Track metrics
                self.train_losses.append(train_metrics['loss'])
                self.val_losses.append(val_metrics['loss'])
                self.train_maes.append(train_metrics['calorie_mae'])
                self.val_maes.append(val_metrics['calorie_mae'])
                self.train_weight_maes.append(train_metrics['weight_mae'])
                self.val_weight_maes.append(val_metrics['weight_mae'])
                
                # Check for best model 
                combined_mae = val_metrics['calorie_mae'] + val_metrics['weight_mae']
                is_best = combined_mae < (self.best_mae + self.best_weight_mae)
                
                if is_best:
                    self.best_mae = val_metrics['calorie_mae']
                    self.best_weight_mae = val_metrics['weight_mae']
                    self.best_metrics = val_metrics
                    self.best_epoch = epoch + 1
                
                # Logging
                logger.info(f" Train - Loss: {train_metrics['loss']:.4f}, "
                        f"Cal_MAE: {train_metrics['calorie_mae']:.2f} kcal, "
                        f"Wt_MAE: {train_metrics['weight_mae']:.2f} g, "
                        f"Cal_MAPE: {train_metrics['calorie_mape']:.2f}%, "
                        f"Wt_MAPE: {train_metrics['weight_mape']:.2f}%")
                
                logger.info(f" Val - Loss: {val_metrics['loss']:.4f}, "
                        f"Cal_MAE: {val_metrics['calorie_mae']:.2f} kcal, "
                        f"Wt_MAE: {val_metrics['weight_mae']:.2f} g, "
                        f"Cal_MAPE: {val_metrics['calorie_mape']:.2f}%, "
                        f"Wt_MAPE: {val_metrics['weight_mape']:.2f}%")
                
                self.save_checkpoint(epoch, val_metrics, is_best)
                
                # Wandb logging
                try:
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': train_metrics['loss'],
                        'val_loss': val_metrics['loss'],
                        'train_calorie_mae': train_metrics['calorie_mae'],
                        'val_calorie_mae': val_metrics['calorie_mae'],
                        'train_weight_mae': train_metrics['weight_mae'],
                        'val_weight_mae': val_metrics['weight_mae'],
                        'train_calorie_mape': train_metrics['calorie_mape'],
                        'val_calorie_mape': val_metrics['calorie_mape'],
                        'train_weight_mape': train_metrics['weight_mape'],
                        'val_weight_mape': val_metrics['weight_mape'],
                        'lr': self.optimizer.param_groups[0]['lr']
                    })
                except:
                    pass
        
        except KeyboardInterrupt:
            logger.info(" Training interrupted by user.")
        finally:
            logger.info(" Final checkpoint saved.")
            
            if self.best_metrics is not None:
                logger.info(f"\n Best Validation Results (Epoch {self.best_epoch}):")
                logger.info("=" * 60)
                logger.info(f" Calorie MAE: {self.best_metrics['calorie_mae']:.2f} kcal")
                logger.info(f" Weight MAE: {self.best_metrics['weight_mae']:.2f} g")
                logger.info(f" Calorie MAPE: {self.best_metrics['calorie_mape']:.2f}%")
                logger.info(f" Weight MAPE: {self.best_metrics['weight_mape']:.2f}%")
                logger.info("=" * 60)
            
            logger.info(" Training completed!")

def main():
    config = {
        'data_root': "E:/MetaFood3D_Calo",
        'batch_size': 8,
        'num_workers': 4,
        'num_epochs': 200,
        'learning_rate': 1e-5,
        'vit_lr': 1e-6,
        'feature_lr': 1e-4,
        'calorie_weight': 1.0, 
        'weight_weight': 1.0,   
        'num_points': 1024,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'resume_training': True,
        'rgb2point_checkpoint': './checkpoints/best.pth',
        'feat_2d_dim': 256,
        'feat_3d_dim': 256,
        'regression_hidden_dims': [512, 256, 128],
        'dropout': 0.3
    }
    
    print(" Food Calorie Estimation Training - End-to-End")
    print("=" * 50)
    print(f" Device: {config['device']}")
    print(f" Batch size: {config['batch_size']}")
    print(f" Epochs: {config['num_epochs']}")
    print(f" RGB2Point checkpoint: {config['rgb2point_checkpoint']}")
    print("=" * 50)
    
    try:
        wandb.init(project="food-calorie-estimation-end2end", config=config)
    except Exception as e:
        logging.warning(f"Failed to initialize wandb: {e}")
        
    logging.info(" Loading MetaFood3D dataset...")
    data_module = MetaFood3DDataModule(
        root_dir=config['data_root'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        max_points=config['num_points']
    )
    
    train_loader = data_module.get_train_loader()
    test_loader = data_module.get_test_loader()
    nutrition_stats = data_module.get_nutrition_stats()
    logging.info(f" Using nutrition stats for normalization: {nutrition_stats}")
    logging.info(" Creating FoodCalorieEstimationModel...")
    model = FoodCalorieEstimationModel(
        num_points=config['num_points'],
        freeze_vit=True,  
        ff_dim=1024,
        num_heads=4,
        hidden_dim=2048,
        feat_2d_dim=config['feat_2d_dim'],
        feat_3d_dim=config['feat_3d_dim'],
        regression_hidden_dims=config['regression_hidden_dims'],
        dropout=config['dropout']
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f" Model parameters: {trainable_params:,} / {total_params:,} trainable")
    trainer = EndToEndCalorieTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        nutrition_stats=nutrition_stats,
        device=config['device'],
        lr=config['learning_rate'],
        vit_lr=config['vit_lr'],
        feature_lr=config['feature_lr'],
        calorie_weight=config['calorie_weight'],
        weight_weight=config['weight_weight']
    )

    logging.info(" Starting training...")
    trainer.train(
        num_epochs=config['num_epochs'],
        resume=config['resume_training'],
        rgb2point_checkpoint=config['rgb2point_checkpoint']
    )
    
    logging.info(" Training completed")

if __name__ == "__main__":
    main()