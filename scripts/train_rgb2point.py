import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import logging
from tqdm import tqdm
import wandb
import sys
import signal
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.rgb2point_model import RGB2PointModel, chamfer_distance, compute_all_metrics
from Data.Dataset import MetaFood3DDataModule
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RGB2PointTrainer:
    def __init__(self, 
                 model: RGB2PointModel,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 device: str = 'cuda',
                 lr: float = 1e-4,
                 vit_lr: float = 1e-5,  
                 save_dir: str = './checkpoints'):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.save_dir = save_dir
        
        self.optimizer = optim.Adam([
            {'params': self.model.vit_encoder.parameters(), 'lr': vit_lr},
            {'params': self.model.cfi.parameters(), 'lr': lr},
            {'params': self.model.gpm.parameters(), 'lr': lr}
        ])
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
)
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.train_losses = []
        self.test_losses = []
        self.best_loss = float('inf')
        self.start_epoch = 0
        self.should_stop = False
        self.setup_signal_handlers()
    
    def setup_signal_handlers(self):
        def signal_handler(signum, frame):
            logger.info("\n Received interrupt signal. ")
            self.should_stop = True
        signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, signal_handler)
    
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            if self.should_stop:
                logger.info(" Stop training ")
                break
            images = batch['image'].to(self.device)
            if 'pointcloud' in batch:
                gt_pointclouds = batch['pointcloud'].to(self.device)
                
            self.optimizer.zero_grad()
            pred_pointclouds = self.model(images)
            loss = chamfer_distance(pred_pointclouds, gt_pointclouds)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Validation"):
                if self.should_stop:
                    break
                    
                images = batch['image'].to(self.device)
                if 'pointcloud' in batch:
                    gt_pointclouds = batch['pointcloud'].to(self.device)
                pred_pointclouds = self.model(images)
                loss = chamfer_distance(pred_pointclouds, gt_pointclouds)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def load_checkpoint(self, checkpoint_path: str = None):
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.save_dir, 'latest.pth')
        
        if not os.path.exists(checkpoint_path):
            logger.info(f" No checkpoint found at {checkpoint_path}")
            return False
        
        try:
            logger.info(f" Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            self.train_losses = checkpoint.get('train_losses', [])
            self.test_losses = checkpoint.get('test_losses', [])
            logger.info(f" Resumed from epoch {self.start_epoch}")
            logger.info(f" Best loss so far: {self.best_loss:.6f}")
            logger.info(f" Trained for {len(self.train_losses)} epochs")
            return True
        except Exception as e:
            logger.error(f" Error loading checkpoint: {e}")
            return False
    
    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'timestamp': datetime.now().isoformat()
        }
        
        latest_path = os.path.join(self.save_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)
        
        if is_best:
            best_path = os.path.join(self.save_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            logger.info(f" New best model saved with loss: {loss:.6f}")
        
        if epoch % 10 == 0:
            epoch_path = os.path.join(self.save_dir, f'epoch_{epoch}.pth')
            torch.save(checkpoint, epoch_path)
    
    def validate_with_metrics(self) -> dict:
        self.model.eval()
        total_metrics = {
            'chamfer_distance': 0.0,
            'earth_movers_distance': 0.0,
        }
        num_batches = 0
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Validation with metrics"):
                if self.should_stop:
                    break
                    
                images = batch['image'].to(self.device)      
                if 'pointcloud' in batch:
                    gt_pointclouds = batch['pointcloud'].to(self.device)
                pred_pointclouds = self.model(images)
                metrics = compute_all_metrics(pred_pointclouds, gt_pointclouds)
                
                for key, value in metrics.items():
                    if key in total_metrics:
                        total_metrics[key] += value
                num_batches += 1
        
        if num_batches > 0:
            for key in total_metrics:
                total_metrics[key] /= num_batches
        return total_metrics
    
    def train(self, num_epochs: int, resume: bool = True):
        if resume:
            self.load_checkpoint()
        
        logger.info(f" Starting training from epoch {self.start_epoch+1} to {num_epochs}")
        
        try:
            for epoch in range(self.start_epoch, num_epochs):
                if self.should_stop:
                    logger.info(" Training stopped")
                    break
                
                logger.info(f" Epoch {epoch+1}/{num_epochs}")
                train_loss = self.train_epoch()
                if self.should_stop:
                    break
                    
                self.train_losses.append(train_loss)
                test_loss = self.validate()
                if self.should_stop:
                    break
                    
                self.test_losses.append(test_loss)
                
                if epoch % 5 == 0:
                    logger.info(" Computing comprehensive metrics...")
                    metrics = self.validate_with_metrics()
                    logger.info(" Comprehensive Metrics:")
                    logger.info(f"   Chamfer Distance: {metrics['chamfer_distance']:.6f}")
                    logger.info(f"   Earth Mover's Distance: {metrics['earth_movers_distance']:.6f}")

                
                self.scheduler.step(test_loss)
                is_best = test_loss < self.best_loss
                if is_best:
                    self.best_loss = test_loss
                
                logger.info(f" Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
                self.save_checkpoint(epoch, test_loss, is_best)
                
                try:
                    log_data = {
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'test_loss': test_loss,
                        'lr': self.scheduler.get_last_lr()[0]
                    }
                    if epoch % 5 == 0:
                        log_data.update(metrics)
                    wandb.log(log_data)
                except:
                    pass
        
        except KeyboardInterrupt:
            logger.info(" Training interrupted")
        
        finally:
            logger.info(" Final comprehensive evaluation...")
            final_metrics = self.validate_with_metrics()
            logger.info(" FINAL RESULTS:")
            logger.info(f"   Chamfer Distance: {final_metrics['chamfer_distance']:.6f}")
            logger.info(f"   Earth Mover's Distance: {final_metrics['earth_movers_distance']:.6f}")

            
            if len(self.train_losses) > 0:
                final_epoch = self.start_epoch + len(self.train_losses) - 1
                final_loss = self.test_losses[-1] if self.test_losses else float('inf')
                self.save_checkpoint(final_epoch, final_loss)
                logger.info(f" Final checkpoint saved at epoch {final_epoch+1}")
        
        logger.info(" Training completed!")

def main():
    config = {
        'data_root': "E:/MetaFood3D_Calo",
        'batch_size': 8,
        'num_workers': 4,
        'num_epochs': 200,
        'learning_rate': 1e-4,
        'vit_lr': 1e-5, 
        'num_points': 1024,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'resume_training': True
    }
    
    print(" RGB2Point Training Script ")
    print(" stop")
    print(" Checkpoints save in ./checkpoints/")
    print("-" * 50)
    
    try:
        wandb.init(project="rgb2point-metafood3d-improved", config=config)
    except:
        pass
    
    data_module = MetaFood3DDataModule(
        root_dir=config['data_root'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        max_points=config['num_points']
    )
    
    train_loader = data_module.get_train_loader()
    test_loader = data_module.get_test_loader()
    
    model = RGB2PointModel(
        num_points=config['num_points'],
        freeze_vit=False,  
        ff_dim=1024,
        num_heads=4,
        hidden_dim=2048
    )
    
    print(f" Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")
    
    trainer = RGB2PointTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=config['device'],
        lr=config['learning_rate'],
        vit_lr=config['vit_lr']
    )
    
    trainer.train(config['num_epochs'], resume=config['resume_training'])
    

if __name__ == "__main__":
    main()