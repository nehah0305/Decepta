# What: This file implements the model module in backend/Datasets/src/phase1/model.py.
# How: It defines the module logic, declarations, and integrations used by its callers.
# Input: Imported values, function arguments, configuration, and runtime events supplied by callers.
# Output: Exported values, returned results, rendered UI, or runtime side effects produced by this module.

import torch
from torch import nn


class FrameCNN(nn.Module):
    """
    What: This class provides the behavior declared by its name.
    How: It validates and transforms its inputs through the statements in its body.
    Input: Constructor arguments and class attributes supplied by callers.
    Output: Instances exposing the class attributes and methods.
    """
    def __init__(self, frames_per_video: int = 8) -> None:
        """
        What: This function provides the behavior declared by its name.
        How: It validates and transforms its inputs through the statements in its body.
        Input: Arguments: self, frames_per_video.
        Output: Side effects performed by the function; no value is explicitly returned.
        """
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
        """
        What: This function provides the behavior declared by its name.
        How: It validates and transforms its inputs through the statements in its body.
        Input: Arguments: self, video.
        Output: The returned value, or side effects performed by the function.
        """
        batch_size, frames, channels, height, width = video.shape
        features = self.features(video.reshape(batch_size * frames, channels, height, width))
        features = features.reshape(batch_size, frames, -1).mean(dim=1)
        return self.classifier(features)