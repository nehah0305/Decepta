"""
Custom 2D Audio Spectrogram Convolutional Neural Network (2D Audio CNN).

Processes Log-Mel spectrograms (128 mel frequency bins) using a 2D convolutional
architecture over the frequency-time spectrogram representation, preserving the
temporal sequence dimension while pooling frequency.

Canonical Architecture:
Input: (B, 1, 128, T)
  ↓
Conv2D (1 -> 32, k=3x3, p=1) -> BatchNorm2d(32) -> ReLU -> MaxPool2D(2, 2)   # (B, 32, 64, T/2)
  ↓
Conv2D (32 -> 64, k=3x3, p=1) -> BatchNorm2d(64) -> ReLU -> MaxPool2D(2, 2)  # (B, 64, 32, T/4)
  ↓
Conv2D (64 -> 128, k=3x3, p=1) -> BatchNorm2d(128) -> ReLU -> MaxPool2D(2, 1) # (B, 128, 16, T/4)
  ↓
Conv2D (128 -> 256, k=3x3, p=1) -> BatchNorm2d(256) -> ReLU                  # (B, 256, 16, T/4)
  ↓
Frequency Pooling (AdaptiveAvgPool2d((1, None)))                             # (B, 256, 1, T')
  ↓
Reshape & Linear Projection (256 -> 768)                                     # (B, T', 768)

Outputs temporal sequence of audio tokens A_t ∈ R^(B × T' × 768).
"""

import torch
import torch.nn as nn


class Audio2DCNN(nn.Module):
    """
    Scratch-built 2D Spectrogram CNN for acoustic manipulation artifact extraction.
    Preserves the temporal sequence of audio tokens while compressing frequency.
    """

    def __init__(self, in_channels: int = 1, out_dim: int = 768):
        """
        Args:
            in_channels: Number of spectrogram input channels (default 1).
            out_dim: Embedding dimension of output temporal tokens (default 768).
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim

        # Block 1: (B, 1, 128, T) -> (B, 32, 64, T/2)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        # Block 2: (B, 32, 64, T/2) -> (B, 64, 32, T/4)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        # Block 3: (B, 64, 32, T/4) -> (B, 128, 16, T/4)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )

        # Block 4: (B, 128, 16, T/4) -> (B, 256, 16, T/4)
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Frequency pooling: Pool frequency axis (H) to 1 while preserving time axis (W)
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))

        # Linear projection: 256 -> 768
        self.proj = nn.Sequential(
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True)
        )

        self._initialize_weights()

    def _initialize_weights(self):
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

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel_spec: Log-Mel spectrogram tensor.
                      Can be (B, 1, 128, T), (B, 128, T), (1, 128, T), or (128, T).

        Returns:
            temporal_tokens: (B, T', 768) where T' is the temporal sequence length.
        """
        is_unbatched = False

        if mel_spec.dim() == 2:
            # (128, T) -> (1, 1, 128, T)
            mel_spec = mel_spec.unsqueeze(0).unsqueeze(0)
            is_unbatched = True
        elif mel_spec.dim() == 3:
            # (B, 128, T) -> (B, 1, 128, T)
            mel_spec = mel_spec.unsqueeze(1)
            is_unbatched = False
        elif mel_spec.dim() == 4:
            # (B, 1, 128, T) - already 4D
            is_unbatched = False
        else:
            raise ValueError(f"Expected 2D, 3D, or 4D tensor, got shape {mel_spec.shape}")

        x = self.block1(mel_spec)  # (B, 32, 64, T/2)
        x = self.block2(x)         # (B, 64, 32, T/4)
        x = self.block3(x)         # (B, 128, 16, T/4)
        x = self.block4(x)         # (B, 256, 16, T/4)

        # Frequency pooling: (B, 256, 16, T') -> (B, 256, 1, T')
        x = self.freq_pool(x)

        # Squeeze frequency dimension: (B, 256, T')
        x = x.squeeze(2)

        # Permute to (B, T', 256) for sequence modeling
        x = x.permute(0, 2, 1)

        # Project to (B, T', 768)
        temporal_tokens = self.proj(x)

        if is_unbatched:
            temporal_tokens = temporal_tokens.squeeze(0)

        return temporal_tokens


# Aliases for backward compatibility
Audio1DCNN = Audio2DCNN
AudioCNN = Audio2DCNN
