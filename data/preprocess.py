"""
ECG signal preprocessing module
"""

import numpy as np
from scipy import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from utils.logger import get_logger

logger = get_logger('Preprocess', 'preprocess.log')


class ECGPreprocessor:
    """Preprocess ECG signals"""
    
    def __init__(self,
                 target_freq=config.SIGNAL_FREQUENCY,
                 target_length=config.SIGNAL_LENGTH,
                 lowcut=config.FILTER_LOWCUT,
                 highcut=config.FILTER_HIGHCUT,
                 filter_order=config.FILTER_ORDER):
        """
        Args:
            target_freq: Target sampling frequency (Hz)
            target_length: Target signal length (samples)
            lowcut: Low cutoff frequency (Hz)
            highcut: High cutoff frequency (Hz)
            filter_order: Filter order
        """
        self.target_freq = target_freq
        self.target_length = target_length
        self.lowcut = lowcut
        self.highcut = highcut
        self.filter_order = filter_order
        self.sos = None
    
    def _design_filter(self, fs):
        """
        Design Butterworth bandpass filter
        
        Args:
            fs: Sampling frequency of the signal
        """
        nyquist_freq = fs / 2
        low = self.lowcut / nyquist_freq
        high = self.highcut / nyquist_freq
        
        # Ensure frequencies are in valid range (0, 1)
        low = np.clip(low, 0.001, 0.999)
        high = np.clip(high, 0.001, 0.999)
        
        if low >= high:
            high = low + 0.01
            high = np.clip(high, 0.001, 0.999)
        
        self.sos = signal.butter(self.filter_order, [low, high], btype='band', output='sos')
    
    def resample_signal(self, ecg_signal, original_fs):
        """
        Resample ECG signal to target frequency
        
        Args:
            ecg_signal: Input ECG signal
            original_fs: Original sampling frequency
        
        Returns:
            np.array: Resampled signal
        """
        if original_fs == self.target_freq:
            return ecg_signal
        
        num_samples = int(len(ecg_signal) * self.target_freq / original_fs)
        resampled = signal.resample(ecg_signal, num_samples)
        return resampled
    
    def bandpass_filter(self, ecg_signal, fs):
        """
        Apply bandpass filter to remove noise
        
        Args:
            ecg_signal: Input ECG signal
            fs: Sampling frequency
        
        Returns:
            np.array: Filtered signal
        """
        self._design_filter(fs)
        filtered = signal.sosfilt(self.sos, ecg_signal)
        return filtered
    
    def remove_baseline_wander(self, ecg_signal, fs, cutoff=0.5):
        """
        Remove baseline wander using high-pass filter
        
        Args:
            ecg_signal: Input ECG signal
            fs: Sampling frequency
            cutoff: Cutoff frequency for high-pass filter
        
        Returns:
            np.array: Signal without baseline wander
        """
        nyquist_freq = fs / 2
        normal_cutoff = cutoff / nyquist_freq
        normal_cutoff = np.clip(normal_cutoff, 0.001, 0.999)
        
        sos = signal.butter(4, normal_cutoff, btype='high', output='sos')
        filtered = signal.sosfilt(sos, ecg_signal)
        return filtered
    
    def normalize_signal(self, ecg_signal):
        """
        Normalize ECG signal to zero mean and unit variance
        
        Args:
            ecg_signal: Input ECG signal
        
        Returns:
            np.array: Normalized signal
        """
        mean = np.mean(ecg_signal)
        std = np.std(ecg_signal)
        if std == 0:
            std = 1.0
        normalized = (ecg_signal - mean) / std
        return normalized
    
    def pad_or_truncate(self, ecg_signal):
        """
        Pad or truncate signal to target length
        
        Args:
            ecg_signal: Input ECG signal
        
        Returns:
            np.array: Signal with target length
        """
        if len(ecg_signal) >= self.target_length:
            # Truncate from center
            start_idx = (len(ecg_signal) - self.target_length) // 2
            return ecg_signal[start_idx:start_idx + self.target_length]
        else:
            # Pad with zeros (symmetric)
            pad_total = self.target_length - len(ecg_signal)
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            return np.pad(ecg_signal, (pad_left, pad_right), mode='constant', constant_values=0)
    
    def preprocess(self, ecg_signal, original_fs, apply_filter=True):
        """
        Complete preprocessing pipeline
        
        Args:
            ecg_signal: Input ECG signal (can be multi-channel)
            original_fs: Original sampling frequency
            apply_filter: Whether to apply bandpass filter
        
        Returns:
            np.array: Preprocessed signal (1, target_length) for single-lead
        """
        # Handle multi-channel input - take first channel if needed
        if len(ecg_signal.shape) > 1 and ecg_signal.shape[1] > 1:
            ecg_signal = ecg_signal[:, 0]
        elif len(ecg_signal.shape) > 1:
            ecg_signal = ecg_signal.flatten()
        
        # Ensure 1D
        ecg_signal = np.asarray(ecg_signal).flatten()
        
        # Step 1: Resample to target frequency
        ecg_signal = self.resample_signal(ecg_signal, original_fs)
        
        # Step 2: Remove baseline wander
        ecg_signal = self.remove_baseline_wander(ecg_signal, self.target_freq)
        
        # Step 3: Bandpass filter
        if apply_filter:
            ecg_signal = self.bandpass_filter(ecg_signal, self.target_freq)
        
        # Step 4: Normalize
        ecg_signal = self.normalize_signal(ecg_signal)
        
        # Step 5: Pad or truncate to target length
        ecg_signal = self.pad_or_truncate(ecg_signal)
        
        # Return as (1, target_length) for consistency with model input
        return np.expand_dims(ecg_signal, axis=0)
