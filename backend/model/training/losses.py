"""
Loss Functions for End-to-End Multimodal Deepfake Detection Training.

Includes:
1. Binary Cross-Entropy Loss with Logits (L_classification)
2. Temperature-Scaled InfoNCE Synchronization Loss (L_sync) with τ = 0.07
3. Legacy Cosine Margin Synchronization Loss (for backward compatibility)
4. Total Compound Multimodal Objective: L_total = L_classification + λ_sync * L_sync
"""

from typing import List, Optional, Tuple, Union
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
            logits: Predicted logits from model (B, 1) or (B,)
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


class InfoNCESyncLoss(nn.Module):
    """
    Canonical Temperature-Scaled InfoNCE Loss for Audio-Visual Synchronization Pretraining.

    Formulation:
      L_sync = -log( exp(sim(M, A_pos) / τ) / Σ_j exp(sim(M, A_j) / τ) )

    Uses cosine similarity between L2-normalized mouth and audio embeddings with τ = 0.07.
    Supports within-batch cross-modal negatives and temporally shifted negative pairs.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = max(1e-4, temperature)

    def forward(
        self,
        mouth_embeddings: Optional[torch.Tensor] = None,
        audio_embeddings: Optional[torch.Tensor] = None,
        pos_similarities: Optional[torch.Tensor] = None,
        neg_similarities: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None
    ) -> torch.Tensor:
        """
        Computes InfoNCE loss using either raw embeddings or precomputed similarity tensors.

        Args:
            mouth_embeddings: (B, D) or (B, K, D) normalized/unnormalized mouth representations.
            audio_embeddings: (B, D) or (B, K, D) normalized/unnormalized audio representations.
            pos_similarities: (B, K) or (B, 1) or (B,) cosine similarity for in-sync pairs.
            neg_similarities: (B, N_neg) or (B, K) cosine similarity for shifted/mismatched pairs.

        Returns:
            Scalar InfoNCE loss tensor.
        """
        # Mode 1: Compute from raw embeddings across batch
        if mouth_embeddings is not None and audio_embeddings is not None:
            # Flatten / average sequence dimension if 3D
            if mouth_embeddings.dim() == 3:
                m_vec = F.normalize(mouth_embeddings.mean(dim=1), p=2, dim=-1)  # (B, D)
            else:
                m_vec = F.normalize(mouth_embeddings, p=2, dim=-1)

            if audio_embeddings.dim() == 3:
                a_vec = F.normalize(audio_embeddings.mean(dim=1), p=2, dim=-1)  # (B, D)
            else:
                a_vec = F.normalize(audio_embeddings, p=2, dim=-1)

            B = m_vec.size(0)

            # Cosine similarity matrix: (B, B)
            sim_matrix = torch.matmul(m_vec, a_vec.T) / self.temperature

            if B > 1:
                labels = torch.arange(B, device=mouth_embeddings.device)
                loss_m2a = F.cross_entropy(sim_matrix, labels)
                loss_a2m = F.cross_entropy(sim_matrix.T, labels)
                return 0.5 * (loss_m2a + loss_a2m)
            else:
                # Single sample in batch: fall back to negative log-sigmoid
                diag_sim = torch.diagonal(sim_matrix)
                return -F.logsigmoid(diag_sim).mean()

        # Mode 2: Compute from precomputed pos_similarities and optional neg_similarities
        elif pos_similarities is not None:
            pos = pos_similarities.mean(dim=-1, keepdim=True) if pos_similarities.dim() > 1 else pos_similarities.view(-1, 1)

            if neg_similarities is not None:
                if isinstance(neg_similarities, list):
                    neg_tensors = [n.mean(dim=-1, keepdim=True) if n.dim() > 1 else n.view(-1, 1) for n in neg_similarities]
                    neg = torch.cat(neg_tensors, dim=-1)
                elif isinstance(neg_similarities, torch.Tensor):
                    neg = neg_similarities.mean(dim=-1, keepdim=True) if neg_similarities.dim() > 2 else neg_similarities.view(pos.size(0), -1)
                else:
                    neg = pos - 0.5

                # Logits: [pos (index 0), neg_1, neg_2, ...] / tau
                logits = torch.cat([pos, neg], dim=-1) / self.temperature  # (B, 1 + N_neg)
                labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
                return F.cross_entropy(logits, labels)
            else:
                # Positive-only alignment: minimize distance to +1.0
                return (1.0 - pos.mean()) ** 2
        else:
            raise ValueError("Either (mouth_embeddings, audio_embeddings) or pos_similarities must be provided.")


class AudioVisualSyncLoss(nn.Module):
    """
    Unified Audio-Visual Synchronization Loss supporting both InfoNCE (canonical default)
    and Cosine Margin (legacy option).
    """

    def __init__(
        self,
        loss_type: str = "infonce",
        temperature: float = 0.07,
        margin: float = 0.5
    ):
        super().__init__()
        self.loss_type = loss_type.lower()
        self.infonce = InfoNCESyncLoss(temperature=temperature)
        self.margin = margin

    def forward(
        self,
        pos_similarities: Optional[torch.Tensor] = None,
        neg_similarities: Optional[torch.Tensor] = None,
        mouth_embeddings: Optional[torch.Tensor] = None,
        audio_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.loss_type == "infonce":
            return self.infonce(
                mouth_embeddings=mouth_embeddings,
                audio_embeddings=audio_embeddings,
                pos_similarities=pos_similarities,
                neg_similarities=neg_similarities
            )
        else:
            # Legacy Cosine Margin
            pos_mean = pos_similarities.mean() if pos_similarities is not None else torch.tensor(0.5)
            pos_loss = (1.0 - pos_mean) ** 2
            if neg_similarities is not None:
                neg_mean = neg_similarities.mean()
                margin_loss = F.relu(self.margin - (pos_mean - neg_mean))
                return pos_loss + margin_loss
            return pos_loss


class MultimodalCompoundLoss(nn.Module):
    """
    Combined Multimodal Loss: L_total = L_classification + λ_sync * L_sync
    """

    def __init__(
        self,
        lambda_sync: float = 0.5,
        label_smoothing: float = 0.05,
        sync_loss_type: str = "infonce",
        sync_temperature: float = 0.07,
        sync_margin: float = 0.5
    ):
        super().__init__()
        self.lambda_sync = lambda_sync
        self.cls_loss = DeepfakeDetectionLoss(label_smoothing=label_smoothing)
        self.sync_loss = AudioVisualSyncLoss(
            loss_type=sync_loss_type,
            temperature=sync_temperature,
            margin=sync_margin
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        pos_sync_sims: Optional[torch.Tensor] = None,
        neg_sync_sims: Optional[torch.Tensor] = None,
        mouth_embeddings: Optional[torch.Tensor] = None,
        audio_embeddings: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            (total_loss, classification_loss, sync_loss)
        """
        l_cls = self.cls_loss(logits, targets)

        if pos_sync_sims is not None or (mouth_embeddings is not None and audio_embeddings is not None):
            l_sync = self.sync_loss(
                pos_similarities=pos_sync_sims,
                neg_similarities=neg_sync_sims,
                mouth_embeddings=mouth_embeddings,
                audio_embeddings=audio_embeddings
            )
        else:
            l_sync = torch.tensor(0.0, device=logits.device)

        l_total = l_cls + (self.lambda_sync * l_sync)
        return l_total, l_cls, l_sync
