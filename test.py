"""
Testing script for MobileNet ECG classifier
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
import config
from models import create_mobilenet_ecg
from data import DataManager
from utils import get_logger, MetricsComputer

logger = get_logger('Testing', 'test.log')


class Tester:
    """Tester for MobileNet ECG model"""
    
    def __init__(self, model, device='cuda'):
        """
        Args:
            model: PyTorch model
            device: Device to test on
        """
        self.model = model.to(device)
        self.device = device
        self.metrics_computer = MetricsComputer()
    
    def predict(self, test_loader, threshold_strategy="max_prob"):
        """
        Make predictions on test set
        
        Args:
            test_loader: Test data loader
            threshold_strategy: Strategy for thresholding ('max_prob' or 'fixed')
        
        Returns:
            tuple: (all_labels, all_preds, all_proba)
        """
        self.model.eval()
        
        all_labels = []
        all_preds_proba = []
        
        logger.info("Making predictions...")
        
        with torch.no_grad():
            for batch_idx in range(test_loader['num_batches']):
                # Get batch
                start_idx = batch_idx * test_loader['batch_size']
                end_idx = min(start_idx + test_loader['batch_size'], len(test_loader['indices']))
                indices = test_loader['indices'][start_idx:end_idx]
                
                signals, labels = test_loader['dataset'].get_batch(indices, apply_aug=False)
                
                # Convert to tensors
                signals = torch.FloatTensor(signals).to(self.device)
                
                # Forward pass
                logits = self.model(signals)
                proba = torch.softmax(logits, dim=1).cpu().numpy()
                
                all_preds_proba.append(proba)
                all_labels.append(labels)
                
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"Processed {batch_idx + 1}/{test_loader['num_batches']} batches")
        
        # Concatenate all batches
        all_labels = np.concatenate(all_labels, axis=0)
        all_preds_proba = np.concatenate(all_preds_proba, axis=0)
        
        # Get predicted labels based on threshold strategy
        if threshold_strategy == "max_prob":
            all_preds = np.argmax(all_preds_proba, axis=1)
        elif threshold_strategy == "fixed":
            max_probs = np.max(all_preds_proba, axis=1)
            all_preds = np.argmax(all_preds_proba, axis=1)
            # Mark uncertain predictions
            all_preds = np.where(max_probs >= config.THRESHOLD_FIXED, all_preds, -1)
        else:
            all_preds = np.argmax(all_preds_proba, axis=1)
        
        return all_labels, all_preds, all_preds_proba
    
    def evaluate(self, test_loader, threshold_strategy="max_prob"):
        """
        Evaluate model on test set
        
        Args:
            test_loader: Test data loader
            threshold_strategy: Threshold strategy
        
        Returns:
            dict: Evaluation metrics
        """
        all_labels, all_preds, all_preds_proba = self.predict(
            test_loader, threshold_strategy=threshold_strategy
        )
        
        metrics = self.metrics_computer.compute_metrics(
            all_labels, all_preds_proba, all_preds, threshold_strategy=threshold_strategy
        )
        
        return metrics, all_labels, all_preds, all_preds_proba
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from: {checkpoint_path}")
        return checkpoint


def print_metrics_report(metrics, class_names):
    """
    Print detailed metrics report
    
    Args:
        metrics: Metrics dictionary
        class_names: Dictionary mapping class indices to names
    """
    logger.info("\n" + "="*80)
    logger.info("DETAILED METRICS REPORT")
    logger.info("="*80)
    
    # Overall metrics
    logger.info("\n[OVERALL METRICS]")
    logger.info(f"Accuracy:        {metrics['accuracy']:.4f}")
    logger.info(f"Macro F1:        {metrics['macro_f1']:.4f}")
    logger.info(f"Weighted F1:     {metrics['weighted_f1']:.4f}")
    logger.info(f"Weighted AUC:    {metrics['weighted_auc']:.4f}")
    
    # Per-class metrics
    logger.info("\n[PER-CLASS METRICS]")
    logger.info("-" * 80)
    logger.info(f"{'Class':<10} {'AUC':>10} {'F1':>10} {'Precision':>12} {'Recall':>10} {'Specificity':>12}")
    logger.info("-" * 80)
    
    for class_idx, class_name in class_names.items():
        auc = metrics.get(f'auc_{class_name}', 0.0)
        f1 = metrics.get(f'f1_{class_name}', 0.0)
        precision = metrics.get(f'precision_{class_name}', 0.0)
        recall = metrics.get(f'recall_{class_name}', 0.0)
        specificity = metrics.get(f'specificity_{class_name}', 0.0)
        
        logger.info(f"{class_name:<10} {auc:>10.4f} {f1:>10.4f} {precision:>12.4f} {recall:>10.4f} {specificity:>12.4f}")
    
    logger.info("-" * 80)
    logger.info("="*80)


def main():
    """Main testing function"""
    logger.info("="*80)
    logger.info("MobileNet ECG Classification - Testing")
    logger.info("="*80)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() and config.DEVICE == 'cuda' else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Check for best model
    best_model_path = config.MODELS_DIR / 'best_model.pt'
    if not best_model_path.exists():
        logger.error(f"Best model not found at {best_model_path}")
        logger.error("Please train the model first using train.py")
        return
    
    # Create model
    logger.info("Creating MobileNet ECG model...")
    model = create_mobilenet_ecg(
        num_classes=config.NUM_CLASSES,
        width_multiplier=config.MOBILENET_WIDTH_MULTIPLIER,
        dropout=config.DROPOUT_RATE
    )
    
    # Setup tester
    tester = Tester(model, device=device)
    
    # Load checkpoint
    tester.load_checkpoint(best_model_path)
    
    # Prepare test data
    logger.info("Preparing test dataset...")
    data_manager = DataManager()
    train_dataset, val_dataset, test_dataset = data_manager.prepare_datasets(
        train_split=config.TRAIN_SPLIT,
        val_split=config.VAL_SPLIT,
        test_split=config.TEST_SPLIT,
        random_seed=config.RANDOM_SEED
    )
    
    if test_dataset is None:
        logger.error("Failed to prepare test dataset!")
        return
    
    loaders = data_manager.get_data_loaders(batch_size=config.BATCH_SIZE)
    
    # Evaluate
    logger.info("\nEvaluating on test set...")
    metrics, all_labels, all_preds, all_preds_proba = tester.evaluate(
        loaders['test'],
        threshold_strategy=config.THRESHOLD_STRATEGY
    )
    
    # Print report
    class_names = {v: k for k, v in config.CLASSES.items()}
    print_metrics_report(metrics, class_names)
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'threshold_strategy': config.THRESHOLD_STRATEGY,
            'threshold_fixed': config.THRESHOLD_FIXED,
            'batch_size': config.BATCH_SIZE
        },
        'metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) 
                   for k, v in metrics.items()}
    }
    
    results_path = config.RESULTS_DIR / f'test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {results_path}")
    
    # Save predictions
    predictions_path = config.RESULTS_DIR / f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.npz'
    np.savez(
        predictions_path,
        labels=all_labels,
        predictions=all_preds,
        probabilities=all_preds_proba
    )
    logger.info(f"Predictions saved to: {predictions_path}")
    
    logger.info("="*80)
    logger.info("Testing completed!")
    logger.info("="*80)


if __name__ == '__main__':
    main()
