"""
End-to-End Visual Deepfake Detection Model.

Integrates:
1. Custom Spatial CNN (256-D)
2. PyTorch Dynamic 2D FFT Log-Magnitude Module
3. Custom Frequency CNN (256-D)
4. Gated Fusion Module (Convex adaptive gate + 256-D fused feature)
5. Frame-Level Feature & Gate Retention
6. Temporal Transformer (256 -> 768 Projection, Positional Encoding, 2-layer Transformer, Attention Pooling)
7. Final Video-Level Classifier (768 -> 1 Logits)

Includes memory-safe chunked inference/training for arbitrarily high frame counts
and supports full ablation experiments.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from .spatial_cnn import SpatialCNN
from .fft_module import FFT2DModule
from .frequency_cnn import FrequencyCNN
from .gated_fusion import GatedFusion
from .temporal_transformer import TemporalTransformer


@dataclass
class VisualModelOutput:
    """Complete output data bundle from visual deepfake detection model."""
    logits: torch.Tensor                     # (B, 1) or (1,) video-level classification logit
    probability: torch.Tensor                # (B, 1) or (1,) real(0) vs fake(1) probability
    video_feature: torch.Tensor              # (B, 768) aggregated video representation
    frame_fused_features: torch.Tensor       # (B, N, 256) or (N, 256) frame-level representations
    spatial_features: Optional[torch.Tensor] # (B, N, 256) or (N, 256)
    frequency_features: Optional[torch.Tensor] # (B, N, 256) or (N, 256)
    gate_values: Optional[torch.Tensor]      # (B, N, 1) or (N, 1) gate values per frame
    attention_weights: Optional[torch.Tensor] # (B, N) or (N,) transformer attention weights


class VisualDeepfakeDetector(nn.Module):
    """
    Unified end-to-end Visual Deepfake Detection model.
    """

    def __init__(
        self,
        spatial_dim: int = 256,
        frequency_dim: int = 256,
        fusion_hidden_dim: int = 128,
        fused_dim: int = 256,
        transformer_dim: int = 768,
        transformer_heads: int = 8,
        transformer_layers: int = 2,
        dropout: float = 0.1,
        mode: str = "full",
        frame_chunk_size: int = 32
    ):
        """
        Args:
            spatial_dim: Spatial CNN output dimension (256).
            frequency_dim: Frequency CNN output dimension (256).
            fusion_hidden_dim: Gated fusion hidden dimension (128).
            fused_dim: Fused representation dimension (256).
            transformer_dim: Temporal transformer model dimension (768).
            transformer_heads: Attention heads (8).
            transformer_layers: Transformer encoder layers (2).
            dropout: Dropout probability.
            mode: "full" | "spatial_only" | "frequency_only" | "no_gate" | "frame_average".
            frame_chunk_size: Max frame batch size for GPU CNN extraction.
        """
        super().__init__()
        self.mode = mode.lower()
        self.frame_chunk_size = frame_chunk_size

        # Modality Branches
        self.spatial_cnn = SpatialCNN(in_channels=3, feature_dim=spatial_dim)
        self.fft_module = FFT2DModule()
        self.frequency_cnn = FrequencyCNN(in_channels=1, feature_dim=frequency_dim)

        # Fusion
        if self.mode == "no_gate":
            # Simple linear projection without gating
            self.no_gate_proj = nn.Linear(spatial_dim + frequency_dim, fused_dim)
            self.gated_fusion = None
        else:
            self.gated_fusion = GatedFusion(
                spatial_dim=spatial_dim,
                frequency_dim=frequency_dim,
                hidden_dim=fusion_hidden_dim
            )

        # Transformer in-dimension
        if self.mode == "spatial_only":
            trans_in_dim = spatial_dim
        elif self.mode == "frequency_only":
            trans_in_dim = frequency_dim
        else:
            trans_in_dim = fused_dim

        # Temporal Sequence Processing
        if self.mode == "frame_average":
            self.temporal_transformer = None
            self.linear_proj = nn.Linear(trans_in_dim, transformer_dim)
        else:
            self.temporal_transformer = TemporalTransformer(
                in_dim=trans_in_dim,
                d_model=transformer_dim,
                nhead=transformer_heads,
                num_layers=transformer_layers,
                dropout=dropout
            )

        # Final Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(transformer_dim, 1)
        )

    def extract_frame_features_chunked(
        self,
        face_frames: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Processes face frames in memory-safe chunks (e.g. 32 frames per batch)
        to handle hundreds of frames without exceeding GPU VRAM.

        Args:
            face_frames: (N, 3, 224, 224) tensor of aligned face frames.

        Returns:
            fused_features: (N, 256)
            spatial_features: (N, 256) or None
            frequency_features: (N, 256) or None
            gate_values: (N, 1) or None
        """
        total_frames = face_frames.size(0)
        all_fused: List[torch.Tensor] = []
        all_spatial: List[torch.Tensor] = []
        all_freq: List[torch.Tensor] = []
        all_gates: List[torch.Tensor] = []

        for i in range(0, total_frames, self.frame_chunk_size):
            chunk = face_frames[i:i + self.frame_chunk_size]

            # 1. Spatial branch
            if self.mode != "frequency_only":
                fs = self.spatial_cnn(chunk)  # (C, 256)
                all_spatial.append(fs)
            else:
                fs = None

            # 2. Frequency branch
            if self.mode != "spatial_only":
                fft_map = self.fft_module(chunk)     # (C, 1, 224, 224)
                ff = self.frequency_cnn(fft_map)     # (C, 256)
                all_freq.append(ff)
            else:
                ff = None

            # 3. Fusion logic
            if self.mode == "spatial_only":
                fused = fs
                gate = None
            elif self.mode == "frequency_only":
                fused = ff
                gate = None
            elif self.mode == "no_gate":
                concat = torch.cat([fs, ff], dim=-1)
                fused = self.no_gate_proj(concat)
                gate = None
            else:
                fused, gate = self.gated_fusion(fs, ff)
                all_gates.append(gate)

            all_fused.append(fused)

        fused_tensor = torch.cat(all_fused, dim=0)  # (N, 256)
        spatial_tensor = torch.cat(all_spatial, dim=0) if all_spatial else None
        freq_tensor = torch.cat(all_freq, dim=0) if all_freq else None
        gates_tensor = torch.cat(all_gates, dim=0) if all_gates else None

        return fused_tensor, spatial_tensor, freq_tensor, gates_tensor

    def forward(
        self,
        face_frames: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> VisualModelOutput:
        """
        Forward pass of the visual deepfake detection system.

        Args:
            face_frames: Either (N, 3, 224, 224) for single video sequence
                         or (B, N, 3, 224, 224) for batched video training.
            padding_mask: (B, N) boolean mask for padded frames.

        Returns:
            VisualModelOutput containing prediction logits, probabilities,
            video feature, frame-level features, gate values, and attention weights.
        """
        if face_frames.dim() == 4:
            # Single video: (N, 3, 224, 224)
            fused_feat, spatial_feat, freq_feat, gates = self.extract_frame_features_chunked(face_frames)

            if self.mode == "frame_average":
                # Average pooling over frames without transformer
                proj_feat = self.linear_proj(fused_feat)  # (N, 768)
                video_feat = torch.mean(proj_feat, dim=0, keepdim=True)  # (1, 768)
                attn_weights = None
            else:
                # Temporal Transformer: (N, 256) -> (1, 768)
                video_feat, attn_weights = self.temporal_transformer(fused_feat)
                if video_feat.dim() == 1:
                    video_feat = video_feat.unsqueeze(0)  # (1, 768)

            logits = self.classifier(video_feat)          # (1, 1)
            probs = torch.sigmoid(logits)                 # (1, 1)

            return VisualModelOutput(
                logits=logits.squeeze(0),
                probability=probs.squeeze(0),
                video_feature=video_feat.squeeze(0),
                frame_fused_features=fused_feat,
                spatial_features=spatial_feat,
                frequency_features=freq_feat,
                gate_values=gates,
                attention_weights=attn_weights
            )

        elif face_frames.dim() == 5:
            # Batched training: (B, N, 3, 224, 224)
            B, N, C, H, W = face_frames.shape
            flat_frames = face_frames.view(B * N, C, H, W)

            # Chunked processing for the flat batch of frames
            flat_fused, flat_spatial, flat_freq, flat_gates = self.extract_frame_features_chunked(flat_frames)

            batch_fused = flat_fused.view(B, N, -1)
            batch_spatial = flat_spatial.view(B, N, -1) if flat_spatial is not None else None
            batch_freq = flat_freq.view(B, N, -1) if flat_freq is not None else None
            batch_gates = flat_gates.view(B, N, -1) if flat_gates is not None else None

            if self.mode == "frame_average":
                proj_feat = self.linear_proj(batch_fused)  # (B, N, 768)
                if padding_mask is not None:
                    # Masked mean
                    mask_expanded = (~padding_mask).unsqueeze(-1).float()
                    video_feat = (proj_feat * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-6)
                else:
                    video_feat = torch.mean(proj_feat, dim=1)
                attn_weights = None
            else:
                video_feat, attn_weights = self.temporal_transformer(batch_fused, padding_mask=padding_mask)

            logits = self.classifier(video_feat)  # (B, 1)
            probs = torch.sigmoid(logits)         # (B, 1)

            return VisualModelOutput(
                logits=logits,
                probability=probs,
                video_feature=video_feat,
                frame_fused_features=batch_fused,
                spatial_features=batch_spatial,
                frequency_features=batch_freq,
                gate_values=batch_gates,
                attention_weights=attn_weights
            )
        else:
            raise ValueError(f"Expected 4D or 5D tensor for face_frames, got {face_frames.shape}")
