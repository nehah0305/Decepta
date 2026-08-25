"""
Temporal Transformer Module for Sequence-Level Video Aggregation.

Processes the sequence of chronological frame-level features:
1. Linear Projection: 256 -> 768 (d_model)
2. Positional Encoding (preserving chronological temporal ordering)
3. 2-layer TransformerEncoder (d_model=768, nhead=8, num_layers=2)
4. Attention-weighted pooling to produce a unified 768-D Video Feature.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding injected before Transformer attention layers.
    """

    def __init__(self, d_model: int = 768, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, d_model) or (N, d_model)
        """
        is_unbatched = (x.dim() == 2)
        if is_unbatched:
            x = x.unsqueeze(0)

        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        x = self.dropout(x)

        if is_unbatched:
            x = x.squeeze(0)
        return x


class AttentionPooling(nn.Module):
    """
    Learnable attention pooling over sequence length N -> aggregated vector (d_model).
    """

    def __init__(self, d_model: int = 768):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model, 1))
        self.scale = 1.0 / math.sqrt(d_model)
        nn.init.xavier_uniform_(self.query)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, N, d_model) or (N, d_model)
            mask: (B, N) or (N,) boolean mask where True indicates padded/invalid positions.

        Returns:
            pooled: (B, d_model) or (d_model,)
            weights: (B, N) or (N,) attention weights
        """
        is_unbatched = (x.dim() == 2)
        if is_unbatched:
            x = x.unsqueeze(0)
            if mask is not None and mask.dim() == 1:
                mask = mask.unsqueeze(0)

        # Compute raw scores: (B, N, d_model) x (d_model, 1) -> (B, N, 1)
        scores = torch.matmul(x, self.query) * self.scale
        scores = scores.squeeze(-1)  # (B, N)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        weights = torch.softmax(scores, dim=-1)  # (B, N)
        # Handle edge case where all tokens are masked
        weights = torch.nan_to_num(weights, nan=0.0)

        # Weighted sum: (B, 1, N) x (B, N, d_model) -> (B, 1, d_model)
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (B, d_model)

        if is_unbatched:
            pooled = pooled.squeeze(0)
            weights = weights.squeeze(0)

        return pooled, weights


class TemporalTransformer(nn.Module):
    """
    Processes high-coverage sequences of fused frame features into a video-level representation.
    """

    def __init__(
        self,
        in_dim: int = 256,
        d_model: int = 768,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 4096
    ):
        super().__init__()
        self.in_dim = in_dim
        self.d_model = d_model

        # Linear projection: 256 -> 768
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True)
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_seq_len, dropout=dropout)

        # 2-layer TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Attention pooling
        self.pooling = AttentionPooling(d_model=d_model)

    def forward(
        self,
        frame_features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            frame_features: Sequence of frame features: (B, N, 256) or (N, 256)
            padding_mask: (B, N) boolean tensor (True = ignore/pad)

        Returns:
            video_feature: (B, 768)
            attention_weights: (B, N)
        """
        is_unbatched = (frame_features.dim() == 2)
        if is_unbatched:
            frame_features = frame_features.unsqueeze(0)  # (1, N, 256)

        # Step 1: Project 256 -> 768
        x = self.proj(frame_features)  # (B, N, 768)

        # Step 2: Inject Positional Encoding
        x = self.pos_encoder(x)

        # Step 3: Transformer Encoder layers
        if padding_mask is not None:
            all_masked = padding_mask.all(dim=-1, keepdim=True)
            if all_masked.any():
                safe_mask = padding_mask.clone()
                # If a row is 100% padded, unmask all positions in that row so Transformer handles it gracefully
                safe_mask[all_masked.squeeze(-1)] = False
                x = self.transformer_encoder(x, src_key_padding_mask=safe_mask)
            else:
                x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        else:
            x = self.transformer_encoder(x)

        # Step 4: Attention Pooling -> (B, 768)
        video_feature, attn_weights = self.pooling(x, mask=padding_mask)

        if is_unbatched:
            video_feature = video_feature.squeeze(0)
            attn_weights = attn_weights.squeeze(0)

        return video_feature, attn_weights
