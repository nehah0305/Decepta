"""
High-Coverage Adaptive Frame Sampling Module.

Implements the high-coverage frame sampling strategy prioritizing comprehensive
temporal coverage (60-80% of video frames) rather than sparse random sampling.

Preserves strict chronological order, avoids frame duplication, and partitions
candidate frames into GPU-safe chunks without discarding any frames.
"""

from dataclasses import dataclass
import logging
import math
from typing import List, Optional, Sequence
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SamplingPlan:
    """Detailed plan and statistics for video frame sampling."""
    total_usable_frames: int
    target_coverage_ratio: float
    num_candidate_frames: int
    candidate_indices: List[int]
    chunk_size: int
    chunks: List[List[int]]
    actual_coverage_ratio: float

    def summary(self) -> str:
        return (
            f"Sampling Plan: Total={self.total_usable_frames}, "
            f"Sampled={self.num_candidate_frames} ({self.actual_coverage_ratio:.2%}), "
            f"Chunks={len(self.chunks)} (chunk_size={self.chunk_size})"
        )


class HighCoverageFrameSampler:
    """
    Computes high-coverage candidate frame sequences and GPU-safe processing chunks.
    """

    def __init__(
        self,
        coverage_ratio: float = 0.70,
        min_frames: int = 32,
        max_frames: Optional[int] = None,
        chunk_size: int = 32,
        allow_random: bool = False
    ):
        """
        Args:
            coverage_ratio: Fraction of usable frames to sample (0.10 to 1.0, default 0.70).
            min_frames: Minimum candidate frames to sample (default 32).
            max_frames: Optional upper limit on sampled frames (None = no limit).
            chunk_size: Max frames per CNN processing batch (default 32).
            allow_random: If True, uses random sampling; otherwise deterministic equidistant.
        """
        if not (0.0 < coverage_ratio <= 1.0):
            raise ValueError(f"coverage_ratio must be in (0.0, 1.0], got {coverage_ratio}")
        if min_frames <= 0:
            raise ValueError(f"min_frames must be positive, got {min_frames}")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")

        self.coverage_ratio = coverage_ratio
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.chunk_size = chunk_size
        self.allow_random = allow_random

    def compute_candidate_indices(self, total_frames: int) -> List[int]:
        """
        Calculates the exact chronological frame indices to be analyzed.

        Formula:
            num_candidate = max(min_frames, int(total_frames * coverage_ratio))
            capped at total_frames (and max_frames if set).

        Args:
            total_frames: Total usable frames in the input video.

        Returns:
            Strictly increasing, non-duplicate list of frame indices.
        """
        if total_frames <= 0:
            return []

        # If video is smaller than min_frames, take all available frames
        if total_frames <= self.min_frames:
            return list(range(total_frames))

        # Calculate candidate count
        num_candidates = max(self.min_frames, int(math.ceil(total_frames * self.coverage_ratio)))

        # Cap by total frames
        num_candidates = min(num_candidates, total_frames)

        # Cap by max_frames if configured
        if self.max_frames is not None and self.max_frames > 0:
            num_candidates = min(num_candidates, self.max_frames)

        if self.allow_random:
            # Deterministic random subset with temporal sorting
            indices = np.random.choice(total_frames, size=num_candidates, replace=False)
            sorted_indices = sorted(indices.tolist())
            return sorted_indices

        # Uniform, equidistant sampling preserving chronological progression
        # Uses np.linspace across [0, total_frames - 1] to avoid clustering
        raw_indices = np.linspace(0, total_frames - 1, num=num_candidates, endpoint=True)
        int_indices = [int(round(x)) for x in raw_indices]

        # Ensure strictly unique and monotonic
        unique_indices = sorted(list(dict.fromkeys(int_indices)))

        # If duplicate rounding reduced count slightly below num_candidates, fill gaps
        if len(unique_indices) < num_candidates and len(unique_indices) < total_frames:
            seen = set(unique_indices)
            for idx in range(total_frames):
                if idx not in seen:
                    unique_indices.append(idx)
                    seen.add(idx)
                if len(unique_indices) >= num_candidates:
                    break
            unique_indices.sort()

        return unique_indices

    def chunk_indices(self, indices: Sequence[int]) -> List[List[int]]:
        """
        Partitions candidate indices into GPU-safe chunks of size <= chunk_size.
        Retains ALL frames across chunks.
        """
        return [list(indices[i:i + self.chunk_size]) for i in range(0, len(indices), self.chunk_size)]

    def create_sampling_plan(self, total_frames: int) -> SamplingPlan:
        """
        Generates the complete high-coverage sampling plan for a video.
        """
        candidates = self.compute_candidate_indices(total_frames)
        chunks = self.chunk_indices(candidates)
        actual_ratio = len(candidates) / total_frames if total_frames > 0 else 0.0

        return SamplingPlan(
            total_usable_frames=total_frames,
            target_coverage_ratio=self.coverage_ratio,
            num_candidate_frames=len(candidates),
            candidate_indices=candidates,
            chunk_size=self.chunk_size,
            chunks=chunks,
            actual_coverage_ratio=actual_ratio
        )
