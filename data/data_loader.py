"""
ECG data loader for training and testing
"""

import numpy as np
import sys
from pathlib import Path
from typing import List, Tuple, Optional
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from utils.logger import get_logger
from .download_dataset import get_all_records, load_ecg_record, get_record_annotation
from .preprocess import ECGPreprocessor

logger = get_logger('DataLoader', 'dataloader.log')


class ECGDataset:
    """ECG dataset with preprocessing"""
    
    def __init__(self, record_names: List[str], data_dir=config.TRAIN_DATA_DIR,
                 apply_augmentation=False, preprocessor=None):
        """
        Args:
            record_names: List of record names to load
            data_dir: Directory containing the records
            apply_augmentation: Whether to apply data augmentation
            preprocessor: ECGPreprocessor instance
        """
        self.record_names = record_names
        self.data_dir = data_dir
        self.apply_augmentation = apply_augmentation
        self.preprocessor = preprocessor or ECGPreprocessor()
        
        self.signals = []
        self.labels = []
        self._load_data()
    
    def _load_data(self):
        """Load all records"""
        logger.info(f"Loading {len(self.record_names)} records...")
        
        failed_records = []
        
        for i, record_name in enumerate(self.record_names):
            if (i + 1) % 100 == 0:
                logger.info(f"Loaded {i + 1}/{len(self.record_names)} records")
            
            # Load ECG signal
            ecg_signal, fs, _ = load_ecg_record(record_name, self.data_dir)
            if ecg_signal is None:
                failed_records.append(record_name)
                continue
            
            # Get annotation
            annotation = get_record_annotation(record_name, self.data_dir)
            if annotation is None or annotation not in config.CLASSES:
                failed_records.append(record_name)
                continue
            
            # Preprocess signal
            try:
                preprocessed = self.preprocessor.preprocess(ecg_signal, fs)
                self.signals.append(preprocessed)
                self.labels.append(config.CLASSES[annotation])
            except Exception as e:
                logger.warning(f"Failed to preprocess {record_name}: {str(e)}")
                failed_records.append(record_name)
        
        logger.info(f"✓ Loaded {len(self.signals)} records")
        if failed_records:
            logger.warning(f"✗ Failed to load {len(failed_records)} records")
    
    def apply_augmentation_to_batch(self, signals):
        """
        Apply data augmentation
        
        Args:
            signals: Batch of signals (batch_size, channels, length)
        
        Returns:
            np.array: Augmented signals
        """
        if not self.apply_augmentation:
            return signals
        
        augmented = signals.copy()
        
        # Gaussian noise
        noise = np.random.normal(0, config.AUG_NOISE_STD, augmented.shape)
        augmented = augmented + noise
        
        # Scaling
        scale = np.random.uniform(*config.AUG_SCALE_RANGE)
        augmented = augmented * scale
        
        return augmented
    
    def get_batch(self, indices: np.ndarray, apply_aug=False):
        """
        Get a batch of data
        
        Args:
            indices: Batch indices
            apply_aug: Whether to apply augmentation
        
        Returns:
            tuple: (signals, labels) as numpy arrays
        """
        signals = np.array([self.signals[i] for i in indices])
        labels = np.array([self.labels[i] for i in indices])
        
        if apply_aug:
            signals = self.apply_augmentation_to_batch(signals)
        
        return signals, labels
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]


class DataManager:
    """Manage train/val/test splits"""
    
    def __init__(self, data_dir=config.TRAIN_DATA_DIR):
        self.data_dir = data_dir
        self.preprocessor = ECGPreprocessor()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def prepare_datasets(self, train_split=config.TRAIN_SPLIT,
                        val_split=config.VAL_SPLIT,
                        test_split=config.TEST_SPLIT,
                        random_seed=config.RANDOM_SEED):
        """
        Prepare train/val/test datasets
        
        Args:
            train_split: Fraction for training
            val_split: Fraction for validation
            test_split: Fraction for testing
            random_seed: Random seed for reproducibility
        
        Returns:
            tuple: (train_dataset, val_dataset, test_dataset)
        """
        logger.info("Preparing datasets...")
        
        # Get all records
        all_records = get_all_records(self.data_dir)
        logger.info(f"Total records available: {len(all_records)}")
        
        if len(all_records) == 0:
            logger.error("No records found in data directory!")
            return None, None, None
        
        # First split: train+val vs test
        train_val_records, test_records = train_test_split(
            all_records,
            test_size=test_split,
            random_state=random_seed
        )
        
        # Second split: train vs val
        train_records, val_records = train_test_split(
            train_val_records,
            test_size=val_split / (train_split + val_split),
            random_state=random_seed
        )
        
        logger.info(f"Train set: {len(train_records)} records")
        logger.info(f"Val set: {len(val_records)} records")
        logger.info(f"Test set: {len(test_records)} records")
        
        # Create datasets
        self.train_dataset = ECGDataset(
            train_records, self.data_dir,
            apply_augmentation=config.USE_DATA_AUG,
            preprocessor=self.preprocessor
        )
        
        self.val_dataset = ECGDataset(
            val_records, self.data_dir,
            apply_augmentation=False,
            preprocessor=self.preprocessor
        )
        
        self.test_dataset = ECGDataset(
            test_records, self.data_dir,
            apply_augmentation=False,
            preprocessor=self.preprocessor
        )
        
        return self.train_dataset, self.val_dataset, self.test_dataset
    
    def get_data_loaders(self, batch_size=config.BATCH_SIZE):
        """
        Create data loaders for train/val/test
        
        Returns:
            dict: Dictionary with 'train', 'val', 'test' data loaders
        """
        loaders = {}
        
        for split, dataset in [('train', self.train_dataset),
                               ('val', self.val_dataset),
                               ('test', self.test_dataset)]:
            if dataset is not None:
                num_batches = (len(dataset) + batch_size - 1) // batch_size
                indices = np.arange(len(dataset))
                
                loaders[split] = {
                    'dataset': dataset,
                    'num_batches': num_batches,
                    'batch_size': batch_size,
                    'indices': indices
                }
        
        return loaders
