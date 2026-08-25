"""
Master End-to-End Multimodal Deepfake Detection System.

Integrates:
1. Visual Branch (High-Coverage Spatial CNN + FFT Frequency CNN + Gated Fusion + Temporal Transformer -> 768-D)
2. Audio Authenticity Branch (16 kHz Log-Mel + 1D Audio CNN + Self-Attention -> 768-D)
3. Audio-Visual Synchronization Branch (Mouth ROI CNN + Temporal Token Alignment + Learnable Sync Module -> 256-D)
4. Adaptive Modality Attention Fusion (α_v, α_a, α_s with missing-modality masking)
5. Final Classification MLP (768 -> 256 -> 1)

Produces all 10 required outputs and supports all ablation configurations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from .visual_model import VisualDeepfakeDetector
from .audio_branch import AudioAuthenticityBranch, AudioBranchOutput
from .mouth_encoder import MouthROIEncoder
from .sync_branch import AudioVisualSyncBranch, SyncBranchOutput
from .multimodal_fusion import AdaptiveModalityAttention, MultimodalFusionOutput


@dataclass
class MultimodalDetectorOutput:
    """Master output bundle containing all 10 required outputs and intermediate representations."""
    logits: torch.Tensor                     # Final video classification logit
    probability: torch.Tensor                # Final Real(0) vs Fake(1) probability
    prediction: str                          # "Real" or "Fake"
    visual_feature: torch.Tensor             # (768-D)
    audio_feature: torch.Tensor              # (768-D)
    sync_feature: torch.Tensor               # (256-D)
    alpha_v: torch.Tensor                    # Visual modality attention weight
    alpha_a: torch.Tensor                    # Audio modality attention weight
    alpha_s: torch.Tensor                    # Sync modality attention weight
    sync_score: torch.Tensor                 # Scalar audio-visual synchronization score
    fused_feature: torch.Tensor              # (768-D) Multimodal aggregated representation
    temporal_similarities: Optional[torch.Tensor] = None # Cosine similarity curve over time
    frame_gate_values: Optional[torch.Tensor] = None     # Visual spatial/frequency gate values


class MultimodalDeepfakeDetector(nn.Module):
    """
    Unified Multimodal Deepfake Detection model with Adaptive Modality Attention.
    """

    def __init__(
        self,
        visual_dim: int = 768,
        audio_dim: int = 768,
        sync_dim: int = 256,
        fusion_dim: int = 768,
        mode: str = "full",
        dropout: float = 0.1,
        frame_chunk_size: int = 32
    ):
        super().__init__()
        self.mode = mode.lower()
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim
        self.sync_dim = sync_dim
        self.fusion_dim = fusion_dim
        self.frame_chunk_size = frame_chunk_size

        # 1. Visual Branch (Spatial CNN + Frequency CNN + Gated Fusion + Transformer)
        self.visual_branch = VisualDeepfakeDetector(
            spatial_dim=256,
            frequency_dim=256,
            fused_dim=256,
            transformer_dim=visual_dim,
            transformer_heads=8,
            transformer_layers=2,
            dropout=dropout,
            mode="full",
            frame_chunk_size=frame_chunk_size
        )

        # 2. Audio Authenticity Branch (1D Audio CNN + Transformer Self-Attention)
        self.audio_branch = AudioAuthenticityBranch(
            in_mels=128,
            d_model=audio_dim,
            nhead=8,
            num_layers=2,
            dropout=dropout,
            use_self_attention=True
        )

        # 3. Mouth ROI Encoder & Audio-Visual Sync Branch
        self.mouth_encoder = MouthROIEncoder(in_channels=3, embedding_dim=256)
        self.sync_branch = AudioVisualSyncBranch(
            audio_token_dim=audio_dim,
            mouth_dim=256,
            sync_dim=sync_dim,
            dropout=dropout
        )

        # 4. Adaptive Multimodal Fusion
        if self.mode == "concat_fusion":
            self.concat_proj = nn.Sequential(
                nn.Linear(visual_dim + audio_dim + sync_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ReLU(inplace=True)
            )
            self.modality_attention = None
        else:
            self.modality_attention = AdaptiveModalityAttention(
                visual_dim=visual_dim,
                audio_dim=audio_dim,
                sync_dim=sync_dim,
                fusion_dim=fusion_dim,
                hidden_dim=256,
                dropout=dropout
            )

        # 5. Final Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1)
        )

    def forward(
        self,
        face_frames: Optional[torch.Tensor] = None,
        mouth_crops: Optional[torch.Tensor] = None,
        mel_windows: Optional[torch.Tensor] = None,
        modality_mask: Optional[torch.Tensor] = None,
        padding_mask_v: Optional[torch.Tensor] = None,
        padding_mask_a: Optional[torch.Tensor] = None
    ) -> MultimodalDetectorOutput:
        """
        Forward pass for single video or batch of videos.

        Args:
            face_frames: (N, 3, 224, 224) or (B, N, 3, 224, 224)
            mouth_crops: (N, 3, 112, 112) or (B, N, 3, 112, 112)
            mel_windows: (128, T), (W, 128, T), (B, 128, T), or (B, W, 128, T)
            modality_mask: (3,) or (B, 3) boolean mask [has_visual, has_audio, has_sync]
            padding_mask_v: (B, N)
            padding_mask_a: (B, W)
        """
        device = next(self.parameters()).device
        is_batched = (face_frames is not None and face_frames.dim() == 5) or \
                     (mel_windows is not None and mel_windows.dim() == 4)
        batch_size = face_frames.size(0) if (face_frames is not None and face_frames.dim() == 5) else (
            mel_windows.size(0) if (mel_windows is not None and mel_windows.dim() == 4) else 1
        )

        has_v = (face_frames is not None)
        has_a = (mel_windows is not None)
        has_s = (has_v and has_a and mouth_crops is not None)

        if modality_mask is None:
            if is_batched:
                modality_mask = torch.tensor([[has_v, has_a, has_s]] * batch_size, dtype=torch.bool, device=device)
            else:
                modality_mask = torch.tensor([has_v, has_a, has_s], dtype=torch.bool, device=device)

        # ---------------------------------------------------------------------
        # 1. Visual Branch Execution
        # ---------------------------------------------------------------------
        gate_values = None
        if has_v and self.mode not in ["audio_only", "sync_only"]:
            v_out = self.visual_branch(face_frames, padding_mask=padding_mask_v)
            f_visual = v_out.video_feature
            gate_values = v_out.gate_values
        else:
            f_visual = torch.zeros((batch_size, self.visual_dim) if is_batched else self.visual_dim, device=device)

        # ---------------------------------------------------------------------
        # 2. Audio Authenticity Branch Execution
        # ---------------------------------------------------------------------
        if has_a and self.mode not in ["visual_only", "sync_only"]:
            a_out = self.audio_branch(mel_windows, padding_mask=padding_mask_a)
            f_audio = a_out.audio_feature
            audio_tokens = a_out.temporal_tokens
        else:
            f_audio = torch.zeros((batch_size, self.audio_dim) if is_batched else self.audio_dim, device=device)
            audio_tokens = torch.zeros((batch_size, 64, self.audio_dim) if is_batched else (64, self.audio_dim), device=device)

        # ---------------------------------------------------------------------
        # 3. Audio-Visual Synchronization Branch Execution
        # ---------------------------------------------------------------------
        if has_s and self.mode not in ["visual_only", "audio_only", "visual_audio"]:
            mouth_emb = self.mouth_encoder(mouth_crops)
            sync_out = self.sync_branch(mouth_emb, audio_tokens)
            f_sync = sync_out.sync_feature
            sync_score = sync_out.sync_score
            temp_sims = sync_out.temporal_similarities
        else:
            f_sync = torch.zeros((batch_size, self.sync_dim) if is_batched else self.sync_dim, device=device)
            sync_score = torch.tensor([[0.5]] * batch_size if is_batched else [0.5], device=device)
            temp_sims = None

        # ---------------------------------------------------------------------
        # 4. Multimodal Fusion Execution
        # ---------------------------------------------------------------------
        if self.mode == "concat_fusion":
            # Baseline concatenation
            if not is_batched:
                concat_in = torch.cat([f_visual, f_audio, f_sync], dim=-1).unsqueeze(0)
            else:
                concat_in = torch.cat([f_visual, f_audio, f_sync], dim=-1)

            f_fused = self.concat_proj(concat_in)
            if not is_batched:
                f_fused = f_fused.squeeze(0)
            alpha_v = torch.tensor(0.3333, device=device)
            alpha_a = torch.tensor(0.3333, device=device)
            alpha_s = torch.tensor(0.3333, device=device)
        else:
            # Production Adaptive Modality Attention
            fusion_out = self.modality_attention(
                f_visual=f_visual,
                f_audio=f_audio,
                f_sync=f_sync,
                modality_mask=modality_mask
            )
            f_fused = fusion_out.fused_feature
            alpha_v = fusion_out.alpha_v
            alpha_a = fusion_out.alpha_a
            alpha_s = fusion_out.alpha_s

        # ---------------------------------------------------------------------
        # 5. Final Classification MLP
        # ---------------------------------------------------------------------
        classifier_in = f_fused if is_batched else f_fused.unsqueeze(0)
        logits = self.classifier(classifier_in)
        probs = torch.sigmoid(logits)

        if not is_batched:
            logits = logits.squeeze(0)
            probs = probs.squeeze(0)
            pred_str = "Fake" if probs.item() >= 0.5 else "Real"
        else:
            pred_str = "Batch"

        return MultimodalDetectorOutput(
            logits=logits,
            probability=probs,
            prediction=pred_str,
            visual_feature=f_visual,
            audio_feature=f_audio,
            sync_feature=f_sync,
            alpha_v=alpha_v,
            alpha_a=alpha_a,
            alpha_s=alpha_s,
            sync_score=sync_score,
            fused_feature=f_fused,
            temporal_similarities=temp_sims,
            frame_gate_values=gate_values
        )
