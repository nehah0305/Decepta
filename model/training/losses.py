"""
Loss Functions for End-to-End Visual Deepfake Detection Training.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepfakeDetectionLoss(nn.Module):
    """
    Binary Cross-Entropy Loss with Logits for video-level deepfake classification,
    supporting label smoothing and optional focal modulation.
    """

    def __init__(
        self,
        label_smoothing: float = 0.05,
        pos_weight: Optional[float] = None,
        focal_gamma: float = 0.0
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor([pos_weight]))
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Predicted logits from VisualDeepfakeDetector (B, 1) or (B,)
            targets: Binary ground truth labels (B, 1) or (B,) [0 = Real, 1 = Fake]

        Returns:
            Scalar loss tensor.
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        # Apply label smoothing if configured: y_smooth = y * (1 - e) + 0.5 * e
        if self.label_smoothing > 0.0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        if self.focal_gamma > 0.0:
            # Focal loss modulation
            bce_loss = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=self.pos_weight, reduction="none"
            )
            probs = torch.sigmoid(logits)
            p_t = probs * targets + (1 - probs) * (1 - targets)
            focal_weight = (1.0 - p_t) ** self.focal_gamma
            loss = (focal_weight * bce_loss).mean()
        else:
            loss = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=self.pos_weight
            )

        return loss
