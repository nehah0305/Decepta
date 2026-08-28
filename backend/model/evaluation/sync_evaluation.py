"""
Dedicated Audio-Visual Synchronization Evaluation Module.

Meets Section 33 Requirements:
Evaluates synchronization performance separately:
- Synchronized-pair accuracy & cosine similarity
- Misaligned-pair accuracy & cosine similarity under temporal offsets (±0.25s, ±0.5s, ±1.0s, ±2.0s)
- Similarity distribution across temporal shifts
"""

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from models.sync_branch import AudioVisualSyncBranch
from models.mouth_encoder import MouthROIEncoder

logger = logging.getLogger(__name__)


@dataclass
class SyncOffsetEvaluationResult:
    """Evaluation result for a specific temporal shift offset."""
    offset_seconds: float
    mean_similarity: float
    sync_score: float
    classified_as_sync: bool
    accuracy: float


@dataclass
class SyncEvaluationReport:
    """Consolidated synchronization evaluation report across multiple temporal shifts."""
    video_id: str
    synchronized_similarity: float
    synchronized_sync_score: float
    offset_results: List[SyncOffsetEvaluationResult]

    def print_summary(self):
        """Prints formatted sync evaluation report."""
        sep = "=" * 70
        print("\n" + sep)
        print("          AUDIO-VISUAL SYNCHRONIZATION EVALUATION")
        print(sep)
        print(f"Video ID:                          {self.video_id}")
        print(f"Synchronized Cosine Similarity:    {self.synchronized_similarity:>6.4f}")
        print(f"Synchronized Sync Confidence:      {self.synchronized_sync_score * 100:>6.2f}%")
        print("-" * 70)
        print("TEMPORAL OFFSET SENSITIVITY ANALYSIS (+/- 0.25s, +/- 0.5s, +/- 1.0s, +/- 2.0s):")
        print("-" * 70)

        rows = []
        for r in self.offset_results:
            rows.append({
                "Offset (sec)": f"{r.offset_seconds:+.2f}s",
                "Mean Similarity": f"{r.mean_similarity:.4f}",
                "Sync Confidence": f"{r.sync_score * 100:.2f}%",
                "Classification": "IN-SYNC" if r.classified_as_sync else "MISALIGNED",
                "Alignment Accuracy": f"{r.accuracy * 100:.1f}%"
            })
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        print(sep + "\n")


def evaluate_synchronization_offsets(
    sync_branch: AudioVisualSyncBranch,
    mouth_embeddings: torch.Tensor,
    audio_tokens: torch.Tensor,
    sample_rate_tokens: float = 62.75, # approx tokens/sec
    offsets_sec: List[float] = [-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0],
    threshold: float = 0.5
) -> SyncEvaluationReport:
    """
    Evaluates synchronization robustness and sensitivity across positive (0.0s)
    and temporally shifted audio streams.
    """
    sync_branch.eval()
    device = next(sync_branch.parameters()).device

    if mouth_embeddings.dim() == 2:
        mouth_embeddings = mouth_embeddings.unsqueeze(0).to(device) # (1, N, 256)
    if audio_tokens.dim() == 2:
        audio_tokens = audio_tokens.unsqueeze(0).to(device)         # (1, T, 768)

    T_audio = audio_tokens.size(1)
    offset_results: List[SyncOffsetEvaluationResult] = []

    with torch.no_grad():
        # 1. Evaluate In-Sync (0.0s offset)
        sync_out_0 = sync_branch(mouth_embeddings, audio_tokens)
        sim_0 = float(sync_out_0.temporal_similarities.mean().item())
        score_0 = float(sync_out_0.sync_score.item())

        # 2. Evaluate all offsets
        for offset in offsets_sec:
            shift_tokens = int(round(offset * sample_rate_tokens))

            if shift_tokens == 0:
                shifted_audio = audio_tokens
            elif shift_tokens > 0:
                # Shift audio forward: pad start, trim end
                pad = torch.zeros(1, shift_tokens, audio_tokens.size(-1), device=device)
                shifted_audio = torch.cat([pad, audio_tokens[:, :-shift_tokens, :]], dim=1) if shift_tokens < T_audio else pad
            else:
                # Shift audio backward: trim start, pad end
                shift_abs = abs(shift_tokens)
                pad = torch.zeros(1, shift_abs, audio_tokens.size(-1), device=device)
                shifted_audio = torch.cat([audio_tokens[:, shift_abs:, :], pad], dim=1) if shift_abs < T_audio else pad

            out = sync_branch(mouth_embeddings, shifted_audio)
            mean_sim = float(out.temporal_similarities.mean().item())
            sync_sc = float(out.sync_score.item())

            is_in_sync = (sync_sc >= threshold)
            # For 0.0s offset, ground truth is True (in-sync); for offset != 0.0s, ground truth is False (misaligned)
            if offset == 0.0:
                acc = 1.0 if is_in_sync else 0.0
            else:
                acc = 1.0 if not is_in_sync else 0.0

            offset_results.append(
                SyncOffsetEvaluationResult(
                    offset_seconds=offset,
                    mean_similarity=round(mean_sim, 4),
                    sync_score=round(sync_sc, 4),
                    classified_as_sync=is_in_sync,
                    accuracy=acc
                )
            )

    return SyncEvaluationReport(
        video_id="evaluated_sequence",
        synchronized_similarity=round(sim_0, 4),
        synchronized_sync_score=round(score_0, 4),
        offset_results=offset_results
    )
