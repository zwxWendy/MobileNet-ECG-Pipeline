"""
Configuration file for MobileNet ECG Classification Pipeline
"""

import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "challenge2017"
MODELS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATASET CONFIG
# ============================================================================
DATASET_NAME = "challenge-2017"
TRAIN_DATA_DIR = DATA_DIR / "training"
VAL_DATA_DIR = DATA_DIR / "validation"

# ECG Challenge 2017 Classes
CLASSES = {
    "N": 0,      # Normal
    "A": 1,      # Atrial Fibrillation (AFib)
    "O": 2,      # Other Rhythm
    "~": 3       # Noisy
}
NUM_CLASSES = len(CLASSES)

# ECG Signal Parameters
SIGNAL_FREQUENCY = 300  # Hz
SIGNAL_DURATION = 30    # seconds
SIGNAL_LENGTH = SIGNAL_FREQUENCY * SIGNAL_DURATION  # 9000 samples
NUM_LEADS = 1           # Single-lead ECG

# ============================================================================
# DATA PREPROCESSING
# ============================================================================
# Normalize ECG signals
NORMALIZE_MEAN = 0.0
NORMALIZE_STD = 1.0

# Filter parameters
FILTER_LOWCUT = 0.5     # Hz
FILTER_HIGHCUT = 40.0   # Hz
FILTER_ORDER = 4

# Data augmentation
USE_DATA_AUG = True
AUG_NOISE_STD = 0.01    # Gaussian noise
AUG_SCALE_RANGE = (0.95, 1.05)  # Scaling factor

# ============================================================================
# TRAIN/VAL/TEST SPLIT
# ============================================================================
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# ============================================================================
# MODEL CONFIG
# ============================================================================
# MobileNet architecture
MOBILENET_WIDTH_MULTIPLIER = 1.0  # Adjust for model size (0.5, 1.0, 1.5, 2.0)
MOBILENET_DEPTH_MULTIPLIER = 1
MOBILENET_INPUT_CHANNELS = NUM_LEADS
MOBILENET_NUM_FILTERS_FIRST_LAYER = 32

# Regularization
L2_REGULARIZATION = 1e-4
DROPOUT_RATE = 0.3
BATCH_NORM_MOMENTUM = 0.99

# ============================================================================
# TRAINING CONFIG
# ============================================================================
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
LEARNING_RATE_DECAY = 0.95
DECAY_STEP = 10  # epochs

# Optimizer
OPTIMIZER = "adam"  # adam, sgd
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4

# Loss function
LOSS_FUNCTION = "cross_entropy"  # cross_entropy, focal_loss
FOCAL_LOSS_GAMMA = 2.0
FOCAL_LOSS_ALPHA = 0.25

# Early stopping
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_MIN_DELTA = 1e-4

# Checkpoint
SAVE_BEST_ONLY = True
SAVE_CHECKPOINT_INTERVAL = 5

# ============================================================================
# TESTING CONFIG
# ============================================================================
# Threshold strategy for classification (after softmax)
THRESHOLD_STRATEGY = "max_prob"  # "max_prob" or "fixed" (fixed=0.5 for each class)
THRESHOLD_FIXED = 0.5

# ============================================================================
# METRICS CONFIG
# ============================================================================
# Metrics to compute
COMPUTE_METRICS = ["accuracy", "auc", "f1", "sensitivity", "specificity", "precision", "recall"]

# ============================================================================
# DEVICE & PRECISION
# ============================================================================
DEVICE = "cuda"  # cuda or cpu
USE_MIXED_PRECISION = True
NUM_WORKERS = 4

# ============================================================================
# LOGGING
# ============================================================================
LOG_LEVEL = "INFO"
LOG_INTERVAL = 10  # Log every N batches during training
TENSORBOARD_LOG = True

# ============================================================================
# INFERENCE
# ============================================================================
TEST_TIME_AUGMENTATION = False
NUM_TTA_ITERATIONS = 5
