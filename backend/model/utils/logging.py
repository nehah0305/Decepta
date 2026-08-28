"""
Logging, Frame-Level Feature Serialization, and Frame Coverage Reporting Module.

Handles:
1. Formatted Frame Coverage Report generation (Section 12 requirement).
2. Per-frame metadata and feature vector serialization to CSV and JSON (Section 9 requirement).
"""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FrameCoverageReport:
    """Detailed coverage statistics for an analyzed video."""
    video_path: str
    video_id: str
    duration_seconds: float
    fps: float
    total_video_frames: int
    candidate_sampled_frames: int
    frames_with_face: int
    frames_processed_by_cnn: int
    frames_without_face: int
    frames_rejected_as_unusable: int
    coverage_percentage: float
    avg_face_confidence: float
    avg_gate_value: float
    prediction: str  # "Real" or "Fake"
    confidence: float

    def print_summary(self):
        """Prints exact standardized Frame Coverage Report to console."""
        sep = "=" * 65
        print("\n" + sep)
        print("          VISUAL DEEPFAKE DETECTION & COVERAGE REPORT")
        print(sep)
        print(f"Video File:                   {Path(self.video_path).name}")
        print(f"Video ID:                     {self.video_id}")
        print(f"Duration / FPS:               {self.duration_seconds:.2f}s @ {self.fps:.2f} FPS")
        print("-" * 65)
        print(f"Total video frames:           {self.total_video_frames:>6d}")
        print(f"Candidate sampled frames:     {self.candidate_sampled_frames:>6d}")
        print(f"Faces detected:               {self.frames_with_face:>6d}")
        print(f"CNN processed:                {self.frames_processed_by_cnn:>6d}")
        print(f"No face detected:             {self.frames_without_face:>6d}")
        print(f"Unusable frames:              {self.frames_rejected_as_unusable:>6d}")
        print("-" * 65)
        print(f"Effective face-analysis coverage: {self.coverage_percentage:>6.2f}%")
        print(f"Average face confidence:          {self.avg_face_confidence:>6.4f}")
        print(f"Average spatial gate value:       {self.avg_gate_value:>6.4f}")
        print("-" * 65)
        status_text = "[FAKE]" if self.prediction.upper() == "FAKE" else "[REAL]"
        print(f"FINAL PREDICTION:             {self.prediction.upper()} {status_text}")
        print(f"Confidence:                   {self.confidence * 100:.2f}%")
        print(sep + "\n")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def save_frame_metadata_csv_json(
    frame_records: List[Dict[str, Any]],
    output_csv_path: Path,
    output_json_path: Path
):
    """
    Saves frame-level metadata and representations to CSV and JSON files for research.
    """
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. JSON Export (stores full feature vectors if present)
    serializable_records = []
    for r in frame_records:
        rec = dict(r)
        for k, v in rec.items():
            if isinstance(v, (np.ndarray, np.generic)):
                rec[k] = v.tolist()
        serializable_records.append(rec)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(serializable_records, f, indent=2)

    # 2. CSV Export (stores scalar metadata, summarizes vector lengths)
    csv_rows = []
    for r in frame_records:
        row = {
            "video_id": r.get("video_id"),
            "frame_index": r.get("frame_index"),
            "timestamp": r.get("timestamp"),
            "quality_status": r.get("quality_status"),
            "face_detected": r.get("face_detected"),
            "face_confidence": r.get("face_confidence"),
            "gate_value": r.get("gate_value"),
            "spatial_dim": len(r.get("spatial_feature", [])) if r.get("spatial_feature") is not None else 0,
            "frequency_dim": len(r.get("frequency_feature", [])) if r.get("frequency_feature") is not None else 0,
            "fused_dim": len(r.get("fused_feature", [])) if r.get("fused_feature") is not None else 0,
        }
        csv_rows.append(row)

    df = pd.DataFrame(csv_rows)
    df.to_csv(output_csv_path, index=False)
    logger.info(f"Saved frame metadata to {output_csv_path} and {output_json_path}")
