"""
Notebook-style tutorial for ECG classification pipeline
This can be converted to Jupyter notebook format
"""

# ============================================================================
# TUTORIAL: MobileNet ECG Classification Pipeline
# ============================================================================

# STEP 0: Setup and Imports
# ============================================================================

import os
import sys
from pathlib import Path
import numpy as np
import torch
from datetime import datetime

# Add project to path
project_root = Path('.').resolve()
sys.path.insert(0, str(project_root))

import config
from utils import get_logger, MetricsComputer
from data import DataManager, ECGPreprocessor
from models import create_mobilenet_ecg
from train import Trainer
from test import Tester

logger = get_logger('Tutorial', 'tutorial.log')

print("✓ All imports successful")

# ============================================================================
# STEP 1: Check Environment
# ============================================================================

print("\n" + "="*80)
print("STEP 1: Environment Setup")
print("="*80)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
print(f"PyTorch version: {torch.__version__}")

if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Create directories
for dir_path in [config.DATA_DIR, config.MODELS_DIR, config.RESULTS_DIR, config.LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

print(f"Data directory: {config.DATA_DIR}")
print(f"Models directory: {config.MODELS_DIR}")
print(f"Results directory: {config.RESULTS_DIR}")

# ============================================================================
# STEP 2: Prepare Data
# ============================================================================

print("\n" + "="*80)
print("STEP 2: Data Preparation")
print("="*80)

print("Creating DataManager...")
data_manager = DataManager()

print("Preparing train/val/test datasets...")
train_dataset, val_dataset, test_dataset = data_manager.prepare_datasets(
    train_split=config.TRAIN_SPLIT,
    val_split=config.VAL_SPLIT,
    test_split=config.TEST_SPLIT,
    random_seed=config.RANDOM_SEED
)

if train_dataset is None:
    print("ERROR: No datasets available. Please download data first.")
    print("Run: python data/download_dataset.py")
else:
    print(f"✓ Train set: {len(train_dataset)} records")
    print(f"✓ Val set: {len(val_dataset)} records")
    print(f"✓ Test set: {len(test_dataset)} records")
    
    # Get data loaders
    loaders = data_manager.get_data_loaders(batch_size=config.BATCH_SIZE)
    print(f"\nData loaders created")
    print(f"  Train batches: {loaders['train']['num_batches']}")
    print(f"  Val batches: {loaders['val']['num_batches']}")
    print(f"  Test batches: {loaders['test']['num_batches']}")

# ============================================================================
# STEP 3: Create Model
# ============================================================================

print("\n" + "="*80)
print("STEP 3: Model Creation")
print("="*80)

print("Creating MobileNet ECG model...")
model = create_mobilenet_ecg(
    num_classes=config.NUM_CLASSES,
    width_multiplier=config.MOBILENET_WIDTH_MULTIPLIER,
    dropout=config.DROPOUT_RATE
)

model_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model created")
print(f"  Total parameters: {model_params:,}")
print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Test forward pass
print("\nTesting forward pass...")
model.eval()
with torch.no_grad():
    dummy_input = torch.randn(1, config.MOBILENET_INPUT_CHANNELS, config.SIGNAL_LENGTH)
    dummy_input = dummy_input.to(device)
    model = model.to(device)
    output = model(dummy_input)

print(f"✓ Forward pass successful")
print(f"  Input shape: {dummy_input.shape}")
print(f"  Output shape: {output.shape}")

# ============================================================================
# STEP 4: Setup Training
# ============================================================================

print("\n" + "="*80)
print("STEP 4: Training Setup")
print("="*80)

trainer = Trainer(model, device=device, use_mixed_precision=config.USE_MIXED_PRECISION)
trainer.setup_training(
    learning_rate=config.LEARNING_RATE,
    optimizer_type=config.OPTIMIZER,
    weight_decay=config.WEIGHT_DECAY
)

print("✓ Trainer initialized")
print(f"  Optimizer: {config.OPTIMIZER}")
print(f"  Learning rate: {config.LEARNING_RATE}")
print(f"  Loss function: CrossEntropyLoss")
print(f"  Mixed precision: {config.USE_MIXED_PRECISION}")

# ============================================================================
# STEP 5: Train Model (Optional - Skip if already trained)
# ============================================================================

print("\n" + "="*80)
print("STEP 5: Training (This will take a while)")
print("="*80)

best_model_path = config.MODELS_DIR / 'best_model.pt'

if best_model_path.exists():
    print(f"Best model already exists: {best_model_path}")
    response = input("Train again? (y/n): ")
    if response.lower() != 'y':
        print("Skipping training...")
    else:
        print("Starting training...")
        history = trainer.train(loaders['train'], loaders['val'], epochs=config.EPOCHS)
else:
    print("Starting training...")
    if train_dataset is not None:
        history = trainer.train(loaders['train'], loaders['val'], epochs=config.EPOCHS)
    else:
        print("Cannot train without data. Please download data first.")

# ============================================================================
# STEP 6: Test Model
# ============================================================================

print("\n" + "="*80)
print("STEP 6: Testing")
print("="*80)

if not best_model_path.exists():
    print("ERROR: Best model not found. Please train the model first.")
else:
    print("Loading best model...")
    model = create_mobilenet_ecg()
    tester = Tester(model, device=device)
    tester.load_checkpoint(best_model_path)
    
    if test_dataset is not None:
        print("Evaluating on test set...")
        metrics, all_labels, all_preds, all_preds_proba = tester.evaluate(
            loaders['test'],
            threshold_strategy=config.THRESHOLD_STRATEGY
        )
        
        print("\n" + "="*80)
        print("TEST RESULTS")
        print("="*80)
        
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:     {metrics['accuracy']:.4f}")
        print(f"  Macro F1:     {metrics['macro_f1']:.4f}")
        print(f"  Weighted F1:  {metrics['weighted_f1']:.4f}")
        print(f"  Weighted AUC: {metrics['weighted_auc']:.4f}")
        
        print(f"\nPer-Class Metrics:")
        class_names = {v: k for k, v in config.CLASSES.items()}
        for class_idx, class_name in class_names.items():
            print(f"\n  Class {class_name}:")
            print(f"    AUC:        {metrics.get(f'auc_{class_name}', 0.0):.4f}")
            print(f"    F1 Score:   {metrics.get(f'f1_{class_name}', 0.0):.4f}")
            print(f"    Precision:  {metrics.get(f'precision_{class_name}', 0.0):.4f}")
            print(f"    Recall:     {metrics.get(f'recall_{class_name}', 0.0):.4f}")
            print(f"    Specificity:{metrics.get(f'specificity_{class_name}', 0.0):.4f}")

print("\n" + "="*80)
print("TUTORIAL COMPLETED")
print("="*80)
