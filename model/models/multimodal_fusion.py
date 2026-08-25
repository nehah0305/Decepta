"""
Adaptive Multimodal Modality Attention Fusion Module.

Dynamically computes sample-specific attention weights (α_v, α_a, α_s)
across Visual, Audio, and Sync representations with strict missing-modality handling.

Inputs:
- F_visual ∈ R^768
- F_audio  ∈ R^768
- F_sync   ∈ R^256

Projections:
- P_v = Proj_v(F_visual) ∈ R^768
- P_a = Proj_a(F_audio)  ∈ R^768
- P_s = Proj_s(F_sync)   ∈ R^768

Modality Attention:
- e_i = AttentionScore(P_i)
- [α_v, α_a, α_s] = Softmax(masked([e_v, e_a, e_s])) where α_v + α_a + α_s = 1.0
- F_fused = α_v P_v + α_a P_a + α_s P_s ∈ R^768
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MultimodalFusionOutput:
    """Output bundle from adaptive multimodal fusion."""
    fused_feature: torch.Tensor       # (B, 768) or (768,) weighted multimodal representation
    alpha_v: torch.Tensor             # (B, 1) or (1,) visual modality attention weight
    alpha_a: torch.Tensor             # (B, 1) or (1,) audio modality attention weight
    alpha_s: torch.Tensor             # (B, 1) or (1,) sync modality attention weight
    projected_visual: torch.Tensor    # (B, 768)
    projected_audio: torch.Tensor     # (B, 768)
    projected_sync: torch.Tensor      # (B, 768)


class AdaptiveModalityAttention(nn.Module):
    """
    Learns dynamic sample-specific modality weights (α_v, α_a, α_s)
    with strict masking and renormalization for unavailable modalities.
    """

    def __init__(
        self,
        visual_dim: int = 768,
        audio_dim: int = 768,
        sync_dim: int = 256,
        fusion_dim: int = 768,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim
        self.sync_dim = sync_dim
        self.fusion_dim = fusion_dim

        # 1. Linear Projections to Common Fusion Dimension (768)
        self.proj_v = nn.Sequential(
            nn.Linear(visual_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True)
        )
        self.proj_a = nn.Sequential(
            nn.Linear(audio_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True)
        )
        self.proj_s = nn.Sequential(
            nn.Linear(sync_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True)
        )

        # 2. Modality Attention Scoring Network
        self.attn_net = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        f_visual: torch.Tensor,
        f_audio: torch.Tensor,
        f_sync: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None
    ) -> MultimodalFusionOutput:
        """
        Args:
            f_visual: (B, 768) or (768,)
            f_audio: (B, 768) or (768,)
            f_sync: (B, 256) or (256,)
            modality_mask: (B, 3) boolean tensor [mask_v, mask_a, mask_s] where
                           True = available, False = missing/unavailable.

        Returns:
            MultimodalFusionOutput with F_fused and weights (α_v, α_a, α_s).
        """
        is_unbatched = (f_visual.dim() == 1)
        if is_unbatched:
            f_visual = f_visual.unsqueeze(0)
            f_audio = f_audio.unsqueeze(0)
            f_sync = f_sync.unsqueeze(0)
            if modality_mask is not None and modality_mask.dim() == 1:
                modality_mask = modality_mask.unsqueeze(0)

        B = f_visual.size(0)

        # Step 1: Project each modality to common fusion space (B, 768)
        p_v = self.proj_v(f_visual)
        p_a = self.proj_a(f_audio)
        p_s = self.proj_s(f_sync)

        # Stack into tensor: (B, 3, 768)
        stacked_modalities = torch.stack([p_v, p_a, p_s], dim=1)

        # Step 2: Compute Modality Attention Energy Scores: (B, 3, 1) -> (B, 3)
        scores = self.attn_net(stacked_modalities).squeeze(-1)

        # Step 3: Apply Missing-Modality Masking before Softmax
        if modality_mask is not None:
            # modality_mask: (B, 3) with True for available, False for missing
            scores = scores.masked_fill(~modality_mask, -1e9)

        # Step 4: Softmax normalization over the 3 modalities -> (B, 3)
        weights = torch.softmax(scores, dim=-1)
        # Numerical protection if all modalities were masked out
        weights = torch.nan_to_num(weights, nan=0.3333)

        alpha_v = weights[:, 0:1]  # (B, 1)
        alpha_a = weights[:, 1:2]  # (B, 1)
        alpha_s = weights[:, 2:3]  # (B, 1)

        # Step 5: Adaptive Weighted Fusion: F_fused = α_v P_v + α_a P_a + α_s P_s
        f_fused = (alpha_v * p_v) + (alpha_a * p_a) + (alpha_s * p_s)  # (B, 768)

        if is_unbatched:
            f_fused = f_fused.squeeze(0)
            alpha_v = alpha_v.squeeze(0)
            alpha_a = alpha_a.squeeze(0)
            alpha_s = alpha_s.squeeze(0)
            p_v = p_v.squeeze(0)
            p_a = p_a.squeeze(0)
            p_s = p_s.squeeze(0)

        return MultimodalFusionOutput(
            fused_feature=f_fused,
            alpha_v=alpha_v,
            alpha_a=alpha_a,
            alpha_s=alpha_s,
            projected_visual=p_v,
            projected_audio=p_a,
            projected_sync=p_s
        )
