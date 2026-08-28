"""
Download PhysioNet 2017 ECG Challenge dataset
"""

import os
import sys
from pathlib import Path
import wfdb
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from utils.logger import get_logger

logger = get_logger('DataDownload', 'download.log')


def download_challenge_2017_dataset():
    """
    Download PhysioNet 2017 ECG Challenge dataset
    
    Dataset contains:
    - Training set: 8,528 records
    - Reference annotations with 4 classes: N, A, O, ~
    """
    
    logger.info("Starting dataset download...")
    logger.info(f"Target directory: {config.DATA_DIR}")
    
    try:
        # Download the challenge-2017 database
        logger.info("Downloading challenge-2017 database from PhysioNet...")
        wfdb.dl_database(
            db_name='challenge-2017',
            dl_dir=str(config.DATA_DIR),
            files=['*.mat', '*.hea', 'RECORDS.txt']
        )
        logger.info("✓ Dataset download completed")
        
        # Verify download
        verify_dataset()
        
    except Exception as e:
        logger.error(f"✗ Failed to download dataset: {str(e)}")
        raise


def verify_dataset():
    """
    Verify that dataset was downloaded correctly
    """
    logger.info("Verifying dataset...")
    
    training_dir = config.TRAIN_DATA_DIR
    
    if not training_dir.exists():
        logger.warning(f"Training directory not found: {training_dir}")
        return False
    
    # Count files
    mat_files = list(training_dir.glob('*.mat'))
    hea_files = list(training_dir.glob('*.hea'))
    
    logger.info(f"Found {len(mat_files)} .mat files")
    logger.info(f"Found {len(hea_files)} .hea files")
    
    if len(mat_files) == len(hea_files) and len(mat_files) > 0:
        logger.info(f"✓ Dataset verified: {len(mat_files)} ECG records")
        return True
    else:
        logger.warning("✗ Dataset verification failed")
        return False


def load_ecg_record(record_name: str, data_dir=config.TRAIN_DATA_DIR):
    """
    Load a single ECG record
    
    Args:
        record_name: Name of the record (e.g., 'A00001')
        data_dir: Directory containing the records
    
    Returns:
        tuple: (signal, sampling_rate, metadata)
    """
    try:
        record_path = data_dir / record_name
        record = wfdb.rdrecord(str(record_path))
        return record.p_signal, record.fs, record
    except Exception as e:
        logger.error(f"Failed to load record {record_name}: {str(e)}")
        return None, None, None


def get_record_annotation(record_name: str, data_dir=config.TRAIN_DATA_DIR):
    """
    Read annotation file (.txt) for a record
    
    Args:
        record_name: Name of the record
        data_dir: Directory containing the records
    
    Returns:
        str: Annotation label (N, A, O, or ~)
    """
    try:
        annotation_file = data_dir / f"{record_name}.txt"
        with open(annotation_file, 'r') as f:
            # First line contains the annotation
            first_line = f.readline().strip()
            # Parse annotation - format is "label1,label2,..."
            if ',' in first_line:
                labels = first_line.split(',')[0]
            else:
                labels = first_line
            return labels
    except Exception as e:
        logger.error(f"Failed to read annotation for {record_name}: {str(e)}")
        return None


def get_all_records(data_dir=config.TRAIN_DATA_DIR):
    """
    Get list of all records in the dataset
    
    Returns:
        list: List of record names
    """
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return []
    
    # Get all .hea files
    hea_files = list(data_dir.glob('*.hea'))
    record_names = [f.stem for f in hea_files]
    return sorted(record_names)


if __name__ == '__main__':
    download_challenge_2017_dataset()
