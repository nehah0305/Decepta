"""
2D Fast Fourier Transform (FFT) Log-Magnitude Extraction Module.

Computes 2D spatial frequency spectrum dynamically using PyTorch:
1. 2D FFT: torch.fft.fft2()
2. Quadrant Shift: torch.fft.fftshift() (centers DC frequency)
3. Magnitude: torch.abs()
4. Log Compression: torch.log1p()
5. Channel Averaging: Mean across RGB channels -> (B, 1, 224, 224)
"""

import torch
import torch.nn as nn


class FFT2DModule(nn.Module):
    """
    Differentiable 2D FFT module that dynamically computes log-magnitude
    frequency spectra from aligned RGB face tensors.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transforms (B, 3, H, W) RGB face image tensor into (B, 1, H, W) log-magnitude spectrum.

        Args:
            x: Input face tensor (B, 3, 224, 224) in range [0, 1] or [0, 255].

        Returns:
            Log-magnitude frequency tensor (B, 1, 224, 224).
        """
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        # Step 1: 2D FFT over spatial dimensions (H, W)
        fft_complex = torch.fft.fft2(x, dim=(-2, -1))

        # Step 2: FFT Shift to move zero-frequency component to center
        fft_shifted = torch.fft.fftshift(fft_complex, dim=(-2, -1))

        # Step 3: Spectral Magnitude
        magnitude = torch.abs(fft_shifted)

        # Step 4: Dynamic Range Compression: log(1 + magnitude)
        log_magnitude = torch.log1p(magnitude + self.eps)

        # Step 5: Mean across RGB channels -> (B, 1, 224, 224)
        freq_map = torch.mean(log_magnitude, dim=1, keepdim=True)

        # Optional instance-level min-max normalization for numerical stability
        min_val = freq_map.amin(dim=(-2, -1), keepdim=True)
        max_val = freq_map.amax(dim=(-2, -1), keepdim=True)
        norm_freq_map = (freq_map - min_val) / (max_val - min_val + self.eps)

        return norm_freq_map
