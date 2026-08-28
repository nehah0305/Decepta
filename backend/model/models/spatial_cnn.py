"""
Custom Spatial Convolutional Neural Network (Spatial CNN).

Extracts 256-dimensional spatial visual artifact representations from
canonical 224x224x3 aligned face crops.

Architecture:
- Block 1: Conv 3x3 (32) -> BN -> ReLU -> Conv 3x3 (32) -> BN -> ReLU -> MaxPool 2x2
- Block 2: Conv 3x3 (64) -> BN -> ReLU -> Conv 3x3 (64) -> BN -> ReLU -> MaxPool 2x2
- Block 3: Conv 3x3 (128) -> BN -> ReLU -> Conv 3x3 (128) -> BN -> ReLU -> MaxPool 2x2
- Block 4: Conv 3x3 (256) -> BN -> ReLU
- Global Average Pooling (GAP)
- Linear Projection -> 256-D Feature Vector
"""

import torch
import torch.nn as nn


class SpatialCNN(nn.Module):
    """
    Scratch-built Spatial CNN for facial manipulation artifact detection.
    Outputs 256-D feature vectors (not final classifications).
    """

    def __init__(self, in_channels: int = 3, feature_dim: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim

        # Block 1: 224x224 -> 112x112
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 2: 112x112 -> 56x56
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 3: 56x56 -> 28x28
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 4: 28x28 -> 28x28
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Global Average Pooling and Projection
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.LayerNorm(feature_dim)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming normal initialization for conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input face batch (B, 3, 224, 224)

        Returns:
            Spatial feature tensor (B, 256)
        """
        # Ensure float tensor
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)

        out = self.gap(out)          # (B, 256, 1, 1)
        out = torch.flatten(out, 1)  # (B, 256)
        out = self.fc(out)           # (B, 256)
        return out
