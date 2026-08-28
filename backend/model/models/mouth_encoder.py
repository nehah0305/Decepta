"""
Custom Lightweight Mouth ROI CNN Encoder.

Extracts 256-dimensional motion-oriented mouth embeddings from canonical
112x112x3 mouth crops localized from MTCNN landmarks.

Architecture:
- Conv 3x3 (32) -> BN -> ReLU -> MaxPool 2x2 (56x56)
- Conv 3x3 (64) -> BN -> ReLU -> MaxPool 2x2 (28x28)
- Conv 3x3 (128) -> BN -> ReLU -> MaxPool 2x2 (14x14)
- Conv 3x3 (256) -> BN -> ReLU
- Global Average Pooling (GAP)
- Linear Projection -> 256-D Mouth Embedding M_t
"""

import torch
import torch.nn as nn


class MouthROIEncoder(nn.Module):
    """
    Scratch-built lightweight CNN encoder extracting motion features from mouth ROIs.
    Outputs 256-D temporal mouth embeddings M_t for audio-visual synchronization.
    """

    def __init__(self, in_channels: int = 3, embedding_dim: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim

        # Block 1: 112x112 -> 56x56
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 2: 56x56 -> 28x28
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 3: 28x28 -> 14x14
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 4: 14x14 -> 14x14
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # GAP and Linear Projection
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
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

    def forward(self, mouth_images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mouth_images: (N, 3, 112, 112) or (B, N, 3, 112, 112)

        Returns:
            mouth_embeddings: (N, 256) or (B, N, 256)
        """
        if mouth_images.dtype == torch.uint8:
            mouth_images = mouth_images.float() / 255.0

        if mouth_images.dim() == 4:
            # (N, 3, 112, 112)
            out = self.block1(mouth_images)
            out = self.block2(out)
            out = self.block3(out)
            out = self.block4(out)
            out = self.gap(out)
            out = torch.flatten(out, 1)
            embeddings = self.proj(out)  # (N, 256)
            return embeddings

        elif mouth_images.dim() == 5:
            # (B, N, 3, 112, 112)
            B, N, C, H, W = mouth_images.shape
            flat_crops = mouth_images.view(B * N, C, H, W)
            out = self.block1(flat_crops)
            out = self.block2(out)
            out = self.block3(out)
            out = self.block4(out)
            out = self.gap(out)
            out = torch.flatten(out, 1)
            flat_embeddings = self.proj(out)  # (B * N, 256)
            return flat_embeddings.view(B, N, self.embedding_dim)
        else:
            raise ValueError(f"Expected 4D or 5D tensor, got {mouth_images.shape}")
