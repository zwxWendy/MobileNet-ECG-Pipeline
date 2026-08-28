"""
Training script for MobileNet ECG classifier
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
import config
from models import create_mobilenet_ecg
from data import DataManager
from utils import get_logger, MetricsComputer

logger = get_logger('Training', 'train.log')


class Trainer:
    """Trainer for MobileNet ECG model"""
    
    def __init__(self, model, device='cuda', use_mixed_precision=False):
        """
        Args:
            model: PyTorch model
            device: Device to train on ('cuda' or 'cpu')
            use_mixed_precision: Whether to use mixed precision training
        """
        self.model = model.to(device)
        self.device = device
        self.use_mixed_precision = use_mixed_precision and device == 'cuda'
        
        # Mixed precision
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        self.metrics_computer = MetricsComputer()
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
    
    def setup_training(self, learning_rate=config.LEARNING_RATE,
                       optimizer_type=config.OPTIMIZER,
                       weight_decay=config.WEIGHT_DECAY):
        """
        Setup optimizer, scheduler, and loss function
        
        Args:
            learning_rate: Initial learning rate
            optimizer_type: Type of optimizer ('adam' or 'sgd')
            weight_decay: Weight decay for regularization
        """
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        if optimizer_type.lower() == 'adam':
            self.optimizer = Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=config.MOMENTUM,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        # Learning rate scheduler
        self.scheduler = StepLR(
            self.optimizer,
            step_size=config.DECAY_STEP,
            gamma=config.LEARNING_RATE_DECAY
        )
        
        logger.info(f"Setup: optimizer={optimizer_type}, lr={learning_rate}, weight_decay={weight_decay}")
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
        
        Returns:
            dict: Metrics for the epoch
        """
        self.model.train()
        
        total_loss = 0.0
        all_preds_proba = []
        all_labels = []
        num_batches = 0
        
        for batch_idx in range(train_loader['num_batches']):
            # Get batch
            indices = np.random.choice(
                train_loader['indices'],
                size=min(train_loader['batch_size'], len(train_loader['indices'])),
                replace=False
            )
            signals, labels = train_loader['dataset'].get_batch(indices, apply_aug=True)
            
            # Convert to tensors
            signals = torch.FloatTensor(signals).to(self.device)
            labels = torch.LongTensor(labels).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    logits = self.model(signals)
                    loss = self.criterion(logits, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(signals)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            proba = torch.softmax(logits, dim=1).detach().cpu().numpy()
            all_preds_proba.append(proba)
            all_labels.append(labels.cpu().numpy())
            num_batches += 1
            
            if (batch_idx + 1) % config.LOG_INTERVAL == 0:
                logger.info(f"Batch [{batch_idx + 1}/{train_loader['num_batches']}] Loss: {loss.item():.4f}")
        
        # Aggregate metrics
        avg_loss = total_loss / num_batches
        all_preds_proba = np.concatenate(all_preds_proba, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        metrics = self.metrics_computer.compute_metrics(
            all_labels, all_preds_proba, threshold_strategy="max_prob"
        )
        metrics['loss'] = avg_loss
        
        return metrics
    
    def validate(self, val_loader):
        """
        Validate the model
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            dict: Validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        all_preds_proba = []
        all_labels = []
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx in range(val_loader['num_batches']):
                # Get batch
                start_idx = batch_idx * val_loader['batch_size']
                end_idx = min(start_idx + val_loader['batch_size'], len(val_loader['indices']))
                indices = val_loader['indices'][start_idx:end_idx]
                
                signals, labels = val_loader['dataset'].get_batch(indices, apply_aug=False)
                
                # Convert to tensors
                signals = torch.FloatTensor(signals).to(self.device)
                labels = torch.LongTensor(labels).to(self.device)
                
                # Forward pass
                logits = self.model(signals)
                loss = self.criterion(logits, labels)
                
                # Metrics
                total_loss += loss.item()
                proba = torch.softmax(logits, dim=1).cpu().numpy()
                all_preds_proba.append(proba)
                all_labels.append(labels.cpu().numpy())
                num_batches += 1
        
        # Aggregate metrics
        avg_loss = total_loss / num_batches
        all_preds_proba = np.concatenate(all_preds_proba, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        metrics = self.metrics_computer.compute_metrics(
            all_labels, all_preds_proba, threshold_strategy="max_prob"
        )
        metrics['loss'] = avg_loss
        
        return metrics
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch
            metrics: Current metrics
            is_best: Whether this is the best model
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics
        }
        
        checkpoint_path = config.MODELS_DIR / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        if is_best:
            best_path = config.MODELS_DIR / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved: {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint
    
    def train(self, train_loader, val_loader, epochs=config.EPOCHS):
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
        
        Returns:
            dict: Training history
        """
        logger.info(f"Starting training for {epochs} epochs...")
        
        history = {
            'train': [],
            'val': []
        }
        
        for epoch in range(1, epochs + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch [{epoch}/{epochs}]")
            logger.info(f"{'='*60}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            logger.info("Train Metrics:")
            logger.info(self.metrics_computer.format_metrics(train_metrics))
            history['train'].append(train_metrics)
            
            # Validate
            val_metrics = self.validate(val_loader)
            logger.info("Val Metrics:")
            logger.info(self.metrics_computer.format_metrics(val_metrics))
            history['val'].append(val_metrics)
            
            # Learning rate scheduler step
            self.scheduler.step()
            
            # Early stopping and checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.best_epoch = epoch
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                self.patience_counter += 1
                if epoch % config.SAVE_CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(epoch, val_metrics, is_best=False)
            
            logger.info(f"Best val loss: {self.best_val_loss:.4f} (epoch {self.best_epoch})")
            logger.info(f"Patience: {self.patience_counter}/{config.EARLY_STOPPING_PATIENCE}")
            
            # Early stopping
            if self.patience_counter >= config.EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        logger.info(f"\nTraining completed. Best epoch: {self.best_epoch}")
        return history


def main():
    """Main training function"""
    logger.info("="*60)
    logger.info("MobileNet ECG Classification - Training")
    logger.info("="*60)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() and config.DEVICE == 'cuda' else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create model
    logger.info("Creating MobileNet ECG model...")
    model = create_mobilenet_ecg(
        num_classes=config.NUM_CLASSES,
        width_multiplier=config.MOBILENET_WIDTH_MULTIPLIER,
        dropout=config.DROPOUT_RATE
    )
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Prepare data
    logger.info("Preparing datasets...")
    data_manager = DataManager()
    train_dataset, val_dataset, test_dataset = data_manager.prepare_datasets(
        train_split=config.TRAIN_SPLIT,
        val_split=config.VAL_SPLIT,
        test_split=config.TEST_SPLIT,
        random_seed=config.RANDOM_SEED
    )
    
    if train_dataset is None:
        logger.error("Failed to prepare datasets!")
        return
    
    loaders = data_manager.get_data_loaders(batch_size=config.BATCH_SIZE)
    
    # Setup trainer
    trainer = Trainer(model, device=device, use_mixed_precision=config.USE_MIXED_PRECISION)
    trainer.setup_training(
        learning_rate=config.LEARNING_RATE,
        optimizer_type=config.OPTIMIZER,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Train
    history = trainer.train(
        loaders['train'],
        loaders['val'],
        epochs=config.EPOCHS
    )
    
    # Save history
    history_path = config.RESULTS_DIR / 'training_history.json'
    with open(history_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        history_serializable = {
            'train': [{k: float(v) if isinstance(v, (int, float, np.number)) else v 
                      for k, v in metrics.items()} for metrics in history['train']],
            'val': [{k: float(v) if isinstance(v, (int, float, np.number)) else v 
                    for k, v in metrics.items()} for metrics in history['val']]
        }
        json.dump(history_serializable, f, indent=2)
    logger.info(f"Training history saved: {history_path}")
    
    logger.info("="*60)
    logger.info("Training completed!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
