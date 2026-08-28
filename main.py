"""
Main entry point for the ECG classification pipeline
Running this script will execute the complete pipeline: download -> preprocess -> train -> test
"""

import os
import sys
from pathlib import Path
import argparse
import subprocess

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_logger

logger = get_logger('Main', 'pipeline.log')


def run_pipeline(args):
    """
    Run complete ECG classification pipeline
    
    Args:
        args: Command line arguments
    """
    logger.info("="*80)
    logger.info("ECG CLASSIFICATION PIPELINE")
    logger.info("="*80)
    
    # Step 1: Download dataset
    if args.download:
        logger.info("\n[STEP 1] Downloading dataset...")
        try:
            result = subprocess.run(
                [sys.executable, 'data/download_dataset.py'],
                check=True,
                capture_output=False
            )
            logger.info("✓ Dataset downloaded successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed to download dataset: {e}")
            if not args.force:
                return
    
    # Step 2: Train model
    if args.train:
        logger.info("\n[STEP 2] Training model...")
        try:
            result = subprocess.run(
                [sys.executable, 'train.py'],
                check=True,
                capture_output=False
            )
            logger.info("✓ Model trained successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed to train model: {e}")
            if not args.force:
                return
    
    # Step 3: Test model
    if args.test:
        logger.info("\n[STEP 3] Testing model...")
        try:
            result = subprocess.run(
                [sys.executable, 'test.py'],
                check=True,
                capture_output=False
            )
            logger.info("✓ Model tested successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed to test model: {e}")
            if not args.force:
                return
    
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETED")
    logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='MobileNet ECG Classification Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''\nExamples:
  # Run complete pipeline
  python main.py --download --train --test
  
  # Only train
  python main.py --train
  
  # Only test
  python main.py --test
  
  # Skip download and train (only test)
  python main.py --test --no-download --no-train
        '''
    )
    
    parser.add_argument('--download', action='store_true', default=True,
                       help='Download dataset (default: True)')
    parser.add_argument('--no-download', dest='download', action='store_false',
                       help='Skip dataset download')
    
    parser.add_argument('--train', action='store_true', default=True,
                       help='Train model (default: True)')
    parser.add_argument('--no-train', dest='train', action='store_false',
                       help='Skip training')
    
    parser.add_argument('--test', action='store_true', default=True,
                       help='Test model (default: True)')
    parser.add_argument('--no-test', dest='test', action='store_false',
                       help='Skip testing')
    
    parser.add_argument('--force', action='store_true',
                       help='Continue even if a step fails')
    
    args = parser.parse_args()
    
    run_pipeline(args)


if __name__ == '__main__':
    main()
