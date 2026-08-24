import torch
from torch import nn


class FrameCNN(nn.Module):
    def __init__(self, frames_per_video: int = 8) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 2),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        batch_size, frames, channels, height, width = video.shape
        features = self.features(video.reshape(batch_size * frames, channels, height, width))
        features = features.reshape(batch_size, frames, -1).mean(dim=1)
        return self.classifier(features)