"""
Gated Fusion Module for Spatial and Frequency Representations.

Dynamically balances the contribution of spatial domain and frequency domain
forensic representations per frame using an adaptive gating mechanism.

Formulation:
  Fs ∈ R^256 (Spatial Feature)
  Ff ∈ R^256 (Frequency Feature)
  [Fs, Ff] ∈ R^512 (Concatenation)
  gate = Sigmoid(Linear_2(ReLU(Linear_1([Fs, Ff])))) ∈ [0, 1]
  Fused = gate * Fs + (1 - gate) * Ff ∈ R^256
"""

from typing import Tuple
import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """
    Learns an input-dependent gating coefficient balancing spatial and spectral features.
    """

    def __init__(
        self,
        spatial_dim: int = 256,
        frequency_dim: int = 256,
        hidden_dim: int = 128
    ):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.frequency_dim = frequency_dim
        self.concat_dim = spatial_dim + frequency_dim

        # Gating network: 512 -> 128 -> 1
        self.gate_network = nn.Sequential(
            nn.Linear(self.concat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.gate_network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, fs: torch.Tensor, ff: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            fs: Spatial feature tensor (B, 256)
            ff: Frequency feature tensor (B, 256)

        Returns:
            fused: Fused feature tensor (B, 256)
            gate: Gate scalar tensor (B, 1)
        """
        # Step 1: Concatenate along feature dimension
        concat_features = torch.cat([fs, ff], dim=-1)  # (B, 512)

        # Step 2: Compute gate scalar in range [0, 1]
        gate = self.gate_network(concat_features)      # (B, 1)

        # Step 3: Convex combination of spatial and frequency features
        fused = (gate * fs) + ((1.0 - gate) * ff)      # (B, 256)

        return fused, gate
