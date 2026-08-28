"""
MobileNet architecture for ECG classification
Based on: Howard et al. 2017 - MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class DepthwiseSeparableConv1D(nn.Module):
    """
    Depthwise Separable Convolution for 1D signals (ECG)
    
    Splits standard convolution into:
    - Depthwise: Each input channel filtered separately
    - Pointwise: 1x1 convolution to combine channels
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 dilation: int = 1,
                 bias: bool = True,
                 batch_norm: bool = True,
                 activation: str = 'relu',
                 dropout: float = 0.0):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Kernel size for depthwise convolution
            stride: Stride for depthwise convolution
            padding: Padding for depthwise convolution
            dilation: Dilation for depthwise convolution
            bias: Whether to use bias
            batch_norm: Whether to use batch normalization
            activation: Activation function ('relu', 'relu6', or None)
            dropout: Dropout rate
        """
        super(DepthwiseSeparableConv1D, self).__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,  # Key: groups=in_channels makes it depthwise
            bias=False  # Bias is not used before batch norm
        )
        
        # Batch normalization after depthwise
        self.bn_dw = nn.BatchNorm1d(in_channels) if batch_norm else None
        
        # Pointwise convolution (1x1)
        self.pointwise = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        
        # Batch normalization after pointwise
        self.bn_pw = nn.BatchNorm1d(out_channels) if batch_norm else None
        
        # Activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'relu6':
            self.activation = nn.ReLU6(inplace=True)
        else:
            self.activation = None
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, x):
        # Depthwise
        x = self.depthwise(x)
        if self.bn_dw is not None:
            x = self.bn_dw(x)
        
        # Pointwise
        x = self.pointwise(x)
        if self.bn_pw is not None:
            x = self.bn_pw(x)
        
        # Activation
        if self.activation is not None:
            x = self.activation(x)
        
        # Dropout
        if self.dropout is not None:
            x = self.dropout(x)
        
        return x


class MobileNetBlock(nn.Module):
    """
    MobileNet block with depthwise separable convolution
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 dropout: float = 0.0):
        super(MobileNetBlock, self).__init__()
        
        self.block = nn.Sequential(
            DepthwiseSeparableConv1D(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                activation='relu',
                dropout=dropout
            )
        )
    
    def forward(self, x):
        return self.block(x)


class MobileNetECG(nn.Module):
    """
    MobileNet for ECG classification
    
    Architecture:
    - Input layer
    - Multiple depthwise separable convolutional blocks
    - Global average pooling
    - Classification head
    """
    
    def __init__(self,
                 in_channels: int = config.MOBILENET_INPUT_CHANNELS,
                 num_classes: int = config.NUM_CLASSES,
                 width_multiplier: float = config.MOBILENET_WIDTH_MULTIPLIER,
                 depth_multiplier: int = config.MOBILENET_DEPTH_MULTIPLIER,
                 dropout: float = config.DROPOUT_RATE,
                 base_filters: int = config.MOBILENET_NUM_FILTERS_FIRST_LAYER):
        """
        Args:
            in_channels: Number of input channels (1 for single-lead ECG)
            num_classes: Number of output classes
            width_multiplier: Multiplier to reduce/increase filters (0.5-2.0)
            depth_multiplier: Multiplier for depth dimension
            dropout: Dropout rate
            base_filters: Base number of filters for first layer
        """
        super(MobileNetECG, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.width_multiplier = width_multiplier
        self.dropout = dropout
        
        # Calculate number of filters
        def _make_divisible(v, divisor=8):
            new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
            return new_v
        
        filters = lambda x: _make_divisible(int(x * width_multiplier))
        
        # Initial convolution layer (standard 1D convolution)
        self.initial_conv = nn.Sequential(
            nn.Conv1d(
                in_channels,
                filters(base_filters),
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm1d(filters(base_filters)),
            nn.ReLU(inplace=True)
        )
        
        # MobileNet blocks configuration: (out_channels, kernel_size, stride)
        mobilenet_config = [
            (filters(64), 3, 1),
            (filters(128), 3, 2),
            (filters(128), 3, 1),
            (filters(256), 3, 2),
            (filters(256), 3, 1),
            (filters(512), 3, 2),
            (filters(512), 3, 1),
            (filters(512), 3, 1),
            (filters(512), 3, 1),
            (filters(512), 3, 1),
            (filters(512), 3, 1),
            (filters(1024), 3, 2),
            (filters(1024), 3, 1),
        ]
        
        # Build blocks
        in_ch = filters(base_filters)
        self.blocks = nn.ModuleList()
        
        for out_ch, kernel_size, stride in mobilenet_config:
            self.blocks.append(
                MobileNetBlock(
                    in_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    stride=stride,
                    dropout=dropout
                )
            )
            in_ch = out_ch
        
        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_ch, filters(1024)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(filters(1024), num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, signal_length)
        
        Returns:
            logits: Output logits of shape (batch_size, num_classes)
        """
        # Initial convolution
        x = self.initial_conv(x)
        
        # MobileNet blocks
        for block in self.blocks:
            x = block(x)
        
        # Global average pooling
        x = self.avg_pool(x)  # (batch_size, channels, 1)
        x = x.view(x.size(0), -1)  # (batch_size, channels)
        
        # Classifier
        logits = self.classifier(x)
        
        return logits
    
    def get_embeddings(self, x):
        """
        Get embeddings before classifier head
        
        Args:
            x: Input tensor
        
        Returns:
            embeddings: Feature embeddings
        """
        x = self.initial_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.avg_pool(x)
        embeddings = x.view(x.size(0), -1)
        return embeddings


def create_mobilenet_ecg(num_classes: int = config.NUM_CLASSES,
                        width_multiplier: float = config.MOBILENET_WIDTH_MULTIPLIER,
                        dropout: float = config.DROPOUT_RATE) -> MobileNetECG:
    """
    Factory function to create MobileNet ECG model
    
    Args:
        num_classes: Number of output classes
        width_multiplier: Model size multiplier
        dropout: Dropout rate
    
    Returns:
        MobileNetECG: Initialized model
    """
    model = MobileNetECG(
        in_channels=config.MOBILENET_INPUT_CHANNELS,
        num_classes=num_classes,
        width_multiplier=width_multiplier,
        dropout=dropout
    )
    return model
