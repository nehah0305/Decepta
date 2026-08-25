"""
Audio-Visual Synchronization Branch (Branch B).

Determines temporal correspondence between speaker mouth movements and speech audio:
1. Projects Temporal Audio Tokens (768-D) to Audio Sync Embeddings (256-D).
2. Performs Timestamp-Based Temporal Alignment and Resampling.
3. Computes Normalized Cross-Modal Cosine Similarity: S_t = cosine_similarity(M_t, A_t).
4. Processes concatenated multimodal temporal representations [M_t, A_t, S_t] (513-D)
   through a Learnable Sync Temporal Module.
5. Aggregates over time to produce the 256-D Synchronization Feature F_sync.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .temporal_transformer import AttentionPooling


@dataclass
class SyncBranchOutput:
    """Output data bundle from the audio-visual synchronization branch."""
    sync_feature: torch.Tensor          # (B, 256) or (256,) aggregated synchronization representation
    sync_score: torch.Tensor            # (B, 1) or (1,) scalar alignment confidence score
    temporal_similarities: torch.Tensor # (B, K) or (K,) pointwise cosine similarities over time
    aligned_mouth_embeddings: torch.Tensor # (B, K, 256)
    aligned_audio_embeddings: torch.Tensor # (B, K, 256)


class AudioVisualSyncBranch(nn.Module):
    """
    Learned Audio-Visual Synchronization module mapping temporal mouth motion
    and speech acoustic tokens into a shared synchronization space.
    """

    def __init__(
        self,
        audio_token_dim: int = 768,
        mouth_dim: int = 256,
        sync_dim: int = 256,
        target_seq_len: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.audio_token_dim = audio_token_dim
        self.mouth_dim = mouth_dim
        self.sync_dim = sync_dim
        self.target_seq_len = target_seq_len

        # 1. Project 768-D Audio Tokens to 256-D Sync Embedding
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_token_dim, sync_dim),
            nn.BatchNorm1d(sync_dim),
            nn.ReLU(inplace=True)
        )

        # 2. Linear projection for mouth embeddings
        self.mouth_proj = nn.Sequential(
            nn.Linear(mouth_dim, sync_dim),
            nn.BatchNorm1d(sync_dim),
            nn.ReLU(inplace=True)
        )

        # 3. Learnable Sync Temporal Module: Input is [M_t (256), A_t (256), S_t (1)] = 513
        combined_in_dim = sync_dim + sync_dim + 1
        self.sync_temporal_conv = nn.Sequential(
            nn.Conv1d(combined_in_dim, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Conv1d(256, sync_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(sync_dim),
            nn.ReLU(inplace=True)
        )

        # 4. Temporal Attention Pooling -> 256-D Sync Feature
        self.sync_pooling = AttentionPooling(d_model=sync_dim)

        # 5. Scalar Sync Confidence Head
        self.sync_score_head = nn.Sequential(
            nn.Linear(sync_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def _align_sequences(
        self,
        mouth_seq: torch.Tensor,
        audio_seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Resamples variable-length temporal sequences to a common uniform time grid
        using 1D linear interpolation.

        Args:
            mouth_seq: (B, N_visual, D)
            audio_seq: (B, T_audio, D)

        Returns:
            aligned_mouth: (B, K, D)
            aligned_audio: (B, K, D)
        """
        # Determine common length K
        K = max(self.target_seq_len, min(mouth_seq.size(1), audio_seq.size(1)))

        # (B, N, D) -> (B, D, N) for 1D interpolation
        m_perm = mouth_seq.permute(0, 2, 1)
        a_perm = audio_seq.permute(0, 2, 1)

        m_resampled = F.interpolate(m_perm, size=K, mode="linear", align_corners=False)
        a_resampled = F.interpolate(a_perm, size=K, mode="linear", align_corners=False)

        # (B, D, K) -> (B, K, D)
        aligned_mouth = m_resampled.permute(0, 2, 1)
        aligned_audio = a_resampled.permute(0, 2, 1)

        return aligned_mouth, aligned_audio

    def forward(
        self,
        mouth_embeddings: torch.Tensor,
        audio_tokens: torch.Tensor
    ) -> SyncBranchOutput:
        """
        Args:
            mouth_embeddings: (N, 256) or (B, N, 256)
            audio_tokens: (T, 768) or (B, T, 768)

        Returns:
            SyncBranchOutput containing 256-D sync feature, scalar sync score,
            and temporal cosine similarities.
        """
        is_unbatched = (mouth_embeddings.dim() == 2)
        if is_unbatched:
            mouth_embeddings = mouth_embeddings.unsqueeze(0)  # (1, N, 256)
        if audio_tokens.dim() == 2:
            audio_tokens = audio_tokens.unsqueeze(0)          # (1, T, 768)

        B, N_v, _ = mouth_embeddings.shape
        _, T_a, _ = audio_tokens.shape

        # Step 1: Project mouth and audio to sync dimension
        # Apply BatchNorm over flat batch*time
        flat_mouth = mouth_embeddings.view(B * N_v, -1)
        proj_mouth = self.mouth_proj(flat_mouth).view(B, N_v, self.sync_dim)

        flat_audio = audio_tokens.view(B * T_a, -1)
        proj_audio = self.audio_proj(flat_audio).view(B, T_a, self.sync_dim)

        # Step 2: Temporal alignment and grid resampling
        aligned_mouth, aligned_audio = self._align_sequences(proj_mouth, proj_audio)  # (B, K, 256)

        # Step 3: Compute Pointwise Normalized Cosine Similarity
        norm_mouth = F.normalize(aligned_mouth, p=2, dim=-1)
        norm_audio = F.normalize(aligned_audio, p=2, dim=-1)
        # S_t = dot product: (B, K)
        cosine_sim = (norm_mouth * norm_audio).sum(dim=-1, keepdim=True)  # (B, K, 1)

        # Step 4: Concatenate [M_t, A_t, S_t] -> (B, K, 513)
        sync_input = torch.cat([aligned_mouth, aligned_audio, cosine_sim], dim=-1)  # (B, K, 513)

        # Step 5: Conv1D Temporal Processing
        sync_conv_in = sync_input.permute(0, 2, 1)  # (B, 513, K)
        sync_conv_out = self.sync_temporal_conv(sync_conv_in).permute(0, 2, 1)  # (B, K, 256)

        # Step 6: Attention Pooling -> (B, 256)
        sync_feature, _ = self.sync_pooling(sync_conv_out)

        # Step 7: Scalar Sync Confidence Score
        sync_score = self.sync_score_head(sync_feature)  # (B, 1)

        similarities = cosine_sim.squeeze(-1)  # (B, K)

        if is_unbatched:
            sync_feature = sync_feature.squeeze(0)
            sync_score = sync_score.squeeze(0)
            similarities = similarities.squeeze(0)
            aligned_mouth = aligned_mouth.squeeze(0)
            aligned_audio = aligned_audio.squeeze(0)

        return SyncBranchOutput(
            sync_feature=sync_feature,
            sync_score=sync_score,
            temporal_similarities=similarities,
            aligned_mouth_embeddings=aligned_mouth,
            aligned_audio_embeddings=aligned_audio
        )
