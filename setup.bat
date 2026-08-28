#!/bin/bash

# Windows batch script for setup (setup.bat)
# Run this in Command Prompt or PowerShell on Windows

@echo off
echo ========================================
echo Setting up MobileNet ECG Pipeline
echo ========================================

REM Check Python version
echo [1/4] Checking Python version...
python --version

REM Install dependencies
echo.
echo [2/4] Installing dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    exit /b 1
)

REM Create necessary directories
echo.
echo [3/4] Creating directories...
if not exist data\challenge2017 mkdir data\challenge2017
if not exist checkpoints mkdir checkpoints
if not exist results mkdir results
if not exist logs mkdir logs
echo Directories created successfully

REM Test imports
echo.
echo [4/4] Testing imports...
python -c "import torch; import numpy; import wfdb; import sklearn; print('All imports successful'); print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

if errorlevel 1 (
    echo ERROR: Import test failed
    exit /b 1
)

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo Next steps:
echo 1. Run complete pipeline:
echo    python main.py --download --train --test
echo.
echo 2. Or run individual steps:
echo    python data/download_dataset.py
echo    python train.py
echo    python test.py
echo.
echo Edit config.py to adjust hyperparameters
pause
