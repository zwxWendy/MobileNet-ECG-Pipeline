#!/bin/bash

# MobileNet ECG Classification Pipeline - Setup Script

echo "========================================"
echo "Setting up MobileNet ECG Pipeline"
echo "========================================"

# Check Python version
echo "[1/4] Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

if ! python -c "import sys; sys.exit(0 if sys.version_info >= (3, 7) else 1)"; then
    echo "ERROR: Python 3.7+ is required"
    exit 1
fi

# Install dependencies
echo ""
echo "[2/4] Installing dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

# Create necessary directories
echo ""
echo "[3/4] Creating directories..."
mkdir -p data/challenge2017
mkdir -p checkpoints
mkdir -p results
mkdir -p logs
echo "Directories created successfully"

# Test imports
echo ""
echo "[4/4] Testing imports..."
python -c "
import torch
import numpy as np
import wfdb
import sklearn
import scipy
print('\u2713 All imports successful')
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
" 2>&1

if [ $? -ne 0 ]; then
    echo "ERROR: Import test failed"
    exit 1
fi

echo ""
echo "========================================"
echo "Setup completed successfully!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Run complete pipeline:"
echo "   python main.py --download --train --test"
echo ""
echo "2. Or run individual steps:"
echo "   python data/download_dataset.py  # Download data"
echo "   python train.py                  # Train model"
echo "   python test.py                   # Test model"
echo ""
echo "Edit config.py to adjust hyperparameters"
