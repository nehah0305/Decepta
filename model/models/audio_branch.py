"""
Audio Authenticity Branch (Branch A).

Extracts 768-D discriminative acoustic representations from 16 kHz audio:
1. 1D Audio CNN -> Temporal Audio Tokens A ∈ R^(T' × 768)
2. Positional Encoding
3. Transformer Self-Attention (2 layers, 8 heads, d_model=768)
4. Attention Pooling -> 768-D Window Feature
5. Multi-Window Temporal Aggregation -> 768-D Global Audio Feature F_audio

Also exposes unpooled temporal tokens for the Audio-Visual Synchronization branch.
"""

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn

from .audio_cnn import Audio2DCNN, Audio1DCNN
from .temporal_transformer import AttentionPooling, PositionalEncoding


@dataclass
class AudioBranchOutput:
    """Output data bundle from the audio authenticity branch."""
    audio_feature: torch.Tensor             # (B, 768) or (768,) global video audio representation
    temporal_tokens: torch.Tensor           # (B, T', 768) or (T', 768) frame-level temporal tokens
    window_features: Optional[torch.Tensor] # (B, W, 768) or (W, 768) window representations
    attention_weights: Optional[torch.Tensor] # Attention pooling weights


class AudioAuthenticityBranch(nn.Module):
    """
    Complete Audio Authenticity Branch with Custom 2D Spectrogram CNN and Transformer Self-Attention.
    """

    def __init__(
        self,
        in_mels: int = 128,
        d_model: int = 768,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        use_self_attention: bool = True
    ):
        super().__init__()
        self.in_mels = in_mels
        self.d_model = d_model
        self.use_self_attention = use_self_attention

        # 1. Custom 2D Spectrogram Audio CNN
        self.audio_cnn = Audio2DCNN(in_channels=1, out_dim=d_model)

        # 2. Positional Encoding & Self-Attention
        if self.use_self_attention:
            self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=2048, dropout=dropout)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="relu",
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            self.pos_encoder = None
            self.transformer_encoder = None

        # 3. Within-Window Temporal Attention Pooling
        self.window_pooling = AttentionPooling(d_model=d_model)

        # 4. Across-Window Sequence Aggregation (for multi-window videos)
        self.window_sequence_pooling = AttentionPooling(d_model=d_model)

    def forward_single_window(
        self,
        mel_spec: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Processes a single (B, 128, T) or (128, T) Log-Mel spectrogram window.

        Returns:
            window_feat: (B, 768)
            temporal_tokens: (B, T', 768)
            attn_weights: (B, T')
        """
        is_unbatched = (mel_spec.dim() == 2)
        if is_unbatched:
            mel_spec = mel_spec.unsqueeze(0)  # (1, 128, T)

        # 1. 1D CNN extraction -> (B, T', 768)
        tokens = self.audio_cnn(mel_spec)

        # 2. Self-Attention
        if self.use_self_attention:
            tokens_pe = self.pos_encoder(tokens)
            tokens_trans = self.transformer_encoder(tokens_pe)
        else:
            tokens_trans = tokens

        # 3. Attention Pooling -> (B, 768)
        window_feat, attn_weights = self.window_pooling(tokens_trans)

        if is_unbatched:
            window_feat = window_feat.squeeze(0)
            tokens_trans = tokens_trans.squeeze(0)
            attn_weights = attn_weights.squeeze(0)

        return window_feat, tokens_trans, attn_weights

    def forward(
        self,
        mel_windows: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> AudioBranchOutput:
        """
        Processes one or multiple audio windows for a video or batch of videos.

        Args:
            mel_windows: Either (128, T), (W, 128, T), (B, 128, T), or (B, W, 128, T).
            padding_mask: (B, W) boolean mask if multiple windows are padded.

        Returns:
            AudioBranchOutput with global 768-D audio feature and temporal tokens.
        """
        if mel_windows.dim() == 2:
            # Single window unbatched: (128, T)
            w_feat, tokens, attn = self.forward_single_window(mel_windows)
            return AudioBranchOutput(
                audio_feature=w_feat,
                temporal_tokens=tokens,
                window_features=w_feat.unsqueeze(0),
                attention_weights=attn
            )

        elif mel_windows.dim() == 3:
            # Could be (B, 128, T) for batch of 1-window samples
            # OR (W, 128, T) for 1 video with W windows.
            # We treat (W, 128, T) as multiple windows of a single video
            W, M, T = mel_windows.shape
            w_feats, tokens, attns = self.forward_single_window(mel_windows)  # (W, 768), (W, T', 768)

            if W == 1:
                global_feat = w_feats[0]
            else:
                # Aggregate across windows: (1, W, 768) -> (1, 768)
                global_feat, _ = self.window_sequence_pooling(w_feats.unsqueeze(0))
                global_feat = global_feat.squeeze(0)

            # Flatten temporal tokens across windows for sync branch: (W * T', 768)
            flat_tokens = tokens.view(-1, self.d_model)

            return AudioBranchOutput(
                audio_feature=global_feat,
                temporal_tokens=flat_tokens,
                window_features=w_feats,
                attention_weights=attns
            )

        elif mel_windows.dim() == 4:
            # Batched multi-window: (B, W, 128, T)
            B, W, M, T = mel_windows.shape
            flat_mels = mel_windows.view(B * W, M, T)
            flat_w_feats, flat_tokens, _ = self.forward_single_window(flat_mels)

            batch_w_feats = flat_w_feats.view(B, W, self.d_model)  # (B, W, 768)
            batch_tokens = flat_tokens.view(B, W, -1, self.d_model) # (B, W, T', 768)

            if W == 1:
                global_feats = batch_w_feats.squeeze(1)
            else:
                global_feats, _ = self.window_sequence_pooling(batch_w_feats, mask=padding_mask)

            # Flatten temporal tokens per batch item: (B, W * T', 768)
            flat_batch_tokens = batch_tokens.view(B, -1, self.d_model)

            return AudioBranchOutput(
                audio_feature=global_feats,
                temporal_tokens=flat_batch_tokens,
                window_features=batch_w_feats,
                attention_weights=None
            )
        else:
            raise ValueError(f"Unexpected tensor shape for mel_windows: {mel_windows.shape}")
