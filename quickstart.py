"""
Quick start guide and utility functions
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config
from utils import get_logger

logger = get_logger('QuickStart', 'quickstart.log')


def check_environment():
    """
    Check if the environment is properly set up
    
    Returns:
        bool: True if environment is OK
    """
    logger.info("Checking environment...")
    
    checks = {
        'PyTorch': False,
        'CUDA': False,
        'Data directory': False,
        'Checkpoints directory': False,
        'Results directory': False
    }
    
    # Check PyTorch
    try:
        import torch
        checks['PyTorch'] = True
        logger.info(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        logger.error("✗ PyTorch not installed")
    
    # Check CUDA
    if torch.cuda.is_available():
        checks['CUDA'] = True
        logger.info(f"✓ CUDA available (Device: {torch.cuda.get_device_name()})")
    else:
        logger.warning("⚠ CUDA not available, will use CPU")
    
    # Check directories
    for dir_path in [config.DATA_DIR, config.MODELS_DIR, config.RESULTS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
        if dir_path.exists():
            checks[f'{dir_path.name} directory'] = True
            logger.info(f"✓ Directory exists: {dir_path}")
        else:
            logger.error(f"✗ Directory not found: {dir_path}")
    
    all_ok = all(checks.values())
    if all_ok:
        logger.info("✓ Environment check passed")
    else:
        logger.warning("⚠ Some checks failed")
    
    return all_ok


def print_system_info():
    """
    Print system information
    """
    logger.info("\n" + "="*60)
    logger.info("SYSTEM INFORMATION")
    logger.info("="*60)
    
    import platform
    logger.info(f"OS: {platform.system()} {platform.release()}")
    logger.info(f"Python: {platform.python_version()}")
    
    try:
        import torch
        logger.info(f"PyTorch: {torch.__version__}")
        logger.info(f"CUDA: {torch.version.cuda}")
        logger.info(f"cuDNN: {torch.backends.cudnn.version()}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    except Exception as e:
        logger.warning(f"Could not retrieve PyTorch info: {e}")
    
    import numpy as np
    logger.info(f"NumPy: {np.__version__}")
    
    try:
        import sklearn
        logger.info(f"scikit-learn: {sklearn.__version__}")
    except:
        pass
    
    logger.info("="*60)


def print_config_summary():
    """
    Print current configuration summary
    """
    logger.info("\n" + "="*60)
    logger.info("CONFIGURATION SUMMARY")
    logger.info("="*60)
    
    logger.info("\n[DATA]")
    logger.info(f"  Train/Val/Test split: {config.TRAIN_SPLIT}/{config.VAL_SPLIT}/{config.TEST_SPLIT}")
    logger.info(f"  Signal length: {config.SIGNAL_LENGTH} samples ({config.SIGNAL_DURATION}s @ {config.SIGNAL_FREQUENCY}Hz)")
    logger.info(f"  Preprocessing: Filter {config.FILTER_LOWCUT}-{config.FILTER_HIGHCUT}Hz")
    logger.info(f"  Data augmentation: {config.USE_DATA_AUG}")
    
    logger.info("\n[MODEL]")
    logger.info(f"  Architecture: MobileNet (width_multiplier={config.MOBILENET_WIDTH_MULTIPLIER})")
    logger.info(f"  Input channels: {config.MOBILENET_INPUT_CHANNELS}")
    logger.info(f"  Output classes: {config.NUM_CLASSES} (N, A, O, ~)")
    logger.info(f"  Dropout: {config.DROPOUT_RATE}")
    
    logger.info("\n[TRAINING]")
    logger.info(f"  Epochs: {config.EPOCHS}")
    logger.info(f"  Batch size: {config.BATCH_SIZE}")
    logger.info(f"  Learning rate: {config.LEARNING_RATE}")
    logger.info(f"  Optimizer: {config.OPTIMIZER}")
    logger.info(f"  Weight decay: {config.WEIGHT_DECAY}")
    logger.info(f"  Early stopping patience: {config.EARLY_STOPPING_PATIENCE}")
    
    logger.info("\n[TESTING]")
    logger.info(f"  Threshold strategy: {config.THRESHOLD_STRATEGY}")
    logger.info(f"  Fixed threshold: {config.THRESHOLD_FIXED}")
    
    logger.info("="*60)


def download_and_verify():
    """
    Download dataset and verify
    """
    from data import download_challenge_2017_dataset, verify_dataset
    
    logger.info("Downloading dataset...")
    try:
        download_challenge_2017_dataset()
        verify_dataset()
        logger.info("✓ Dataset ready for training")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to download dataset: {e}")
        return False


def quick_test():
    """
    Quick test: load model and run inference on dummy data
    """
    logger.info("\nRunning quick inference test...")
    
    try:
        from models import create_mobilenet_ecg
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = create_mobilenet_ecg()
        model = model.to(device)
        model.eval()
        
        # Dummy input
        dummy_input = torch.randn(1, config.MOBILENET_INPUT_CHANNELS, config.SIGNAL_LENGTH)
        dummy_input = dummy_input.to(device)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        logger.info(f"✓ Model inference successful")
        logger.info(f"  Input shape: {dummy_input.shape}")
        logger.info(f"  Output shape: {output.shape}")
        logger.info(f"  Output (logits): {output.cpu().numpy()}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Inference test failed: {e}")
        return False


def main():
    """
    Run quick start checks and tests
    """
    logger.info("\n" + "#"*60)
    logger.info("MobileNet ECG Classification - Quick Start")
    logger.info("#"*60)
    
    # Print system info
    print_system_info()
    
    # Check environment
    if not check_environment():
        logger.error("Environment check failed. Please run: pip install -r requirements.txt")
        return
    
    # Print config
    print_config_summary()
    
    # Quick test
    if quick_test():
        logger.info("\n" + "="*60)
        logger.info("✓ ALL CHECKS PASSED")
        logger.info("="*60)
        logger.info("\nYou're ready to start!")
        logger.info("\nOptions:")
        logger.info("1. Download data:    python data/download_dataset.py")
        logger.info("2. Train model:      python train.py")
        logger.info("3. Test model:       python test.py")
        logger.info("4. Full pipeline:    python main.py --download --train --test")
    else:
        logger.error("Quick test failed")


if __name__ == '__main__':
    main()
