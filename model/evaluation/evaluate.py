"""
Video Evaluation and Inference Engine.

Performs high-coverage visual deepfake detection on single videos or video batches:
1. Video Metadata Extraction
2. High-Coverage Adaptive Frame Sampling (60-80% coverage)
3. Lightweight Frame Quality Filtering
4. MTCNN Face Detection & Canonical 224x224 Alignment
5. Chunked Spatial & Frequency CNN Processing
6. Dynamic Gated Fusion with Gate Tracking
7. Positional Temporal Transformer Aggregation
8. Final Classification & Confidence Estimation
9. Detailed Frame Coverage Reporting & Frame Metadata Export
"""

from dataclasses import asdict
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from config import DEFAULT_CONFIG, VisualPipelineConfig
from models.visual_model import VisualDeepfakeDetector, VisualModelOutput
from preprocessing.video_reader import VideoReader
from preprocessing.frame_sampler import HighCoverageFrameSampler
from preprocessing.frame_quality import FrameQualityFilter
from preprocessing.face_alignment import FaceAlignmentPipeline
from utils.logging import FrameCoverageReport, save_frame_metadata_csv_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class VisualDeepfakeEvaluator:
    """
    High-coverage visual deepfake detector evaluating full videos end-to-end.
    """

    def __init__(
        self,
        model: Optional[VisualDeepfakeDetector] = None,
        config: VisualPipelineConfig = DEFAULT_CONFIG,
        checkpoint_path: Optional[Union[str, Path]] = None
    ):
        self.config = config
        self.device = torch.device(config.DEVICE)

        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = VisualDeepfakeDetector(
                spatial_dim=config.SPATIAL_FEATURE_DIM,
                frequency_dim=config.FREQUENCY_FEATURE_DIM,
                fusion_hidden_dim=config.FUSION_HIDDEN_DIM,
                fused_dim=config.FUSED_FEATURE_DIM,
                transformer_dim=config.TRANSFORMER_DIM,
                transformer_heads=config.TRANSFORMER_HEADS,
                transformer_layers=config.TRANSFORMER_LAYERS,
                dropout=config.TRANSFORMER_DROPOUT,
                mode=config.MODEL_MODE,
                frame_chunk_size=config.FRAME_BATCH_SIZE
            ).to(self.device)

            if checkpoint_path is not None:
                chk = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(chk["model_state_dict"])
                logger.info(f"Loaded evaluator weights from {checkpoint_path}")

        self.model.eval()

        # Preprocessing Modules
        self.sampler = HighCoverageFrameSampler(
            coverage_ratio=config.FRAME_COVERAGE_RATIO,
            min_frames=config.MIN_FRAMES,
            max_frames=config.MAX_FRAMES,
            chunk_size=config.FRAME_BATCH_SIZE,
            allow_random=config.ALLOW_RANDOM_SAMPLING
        )
        self.quality_filter = FrameQualityFilter(
            min_brightness=config.QUALITY_MIN_BRIGHTNESS,
            max_brightness=config.QUALITY_MAX_BRIGHTNESS,
            blur_floor=config.QUALITY_BLUR_THRESHOLD,
            non_aggressive=config.NON_AGGRESSIVE_FILTERING
        )
        self.face_aligner = FaceAlignmentPipeline(
            target_size=config.FACE_SIZE,
            min_face_size=config.MTCNN_MIN_FACE_SIZE,
            thresholds=config.MTCNN_THRESHOLDS,
            device=self.device
        )

    def evaluate_video(
        self,
        video_path: Union[str, Path],
        coverage_ratio: Optional[float] = None,
        save_metadata: bool = True
    ) -> Tuple[FrameCoverageReport, List[Dict[str, Any]]]:
        """
        Executes full high-coverage deepfake detection on an input video.

        Args:
            video_path: Path to input video file.
            coverage_ratio: Optional override for sampling coverage (e.g. 0.30, 0.50, 0.70, 0.90).
            save_metadata: If True, writes per-frame records to CSV and JSON.

        Returns:
            Tuple of (FrameCoverageReport, List of frame-level record dictionaries).
        """
        vpath = Path(video_path).resolve()
        if not vpath.exists():
            raise FileNotFoundError(f"Video file not found: {vpath}")

        vid_id = vpath.stem
        start_time = time.time()

        # Step 1: Video Metadata Extraction
        reader = VideoReader(vpath)
        meta = reader.metadata
        total_frames = meta.total_frames

        if total_frames == 0:
            logger.error(f"Video {vpath.name} contains 0 readable frames.")
            rep = FrameCoverageReport(
                video_path=str(vpath),
                video_id=vid_id,
                duration_seconds=0.0,
                fps=0.0,
                total_video_frames=0,
                candidate_sampled_frames=0,
                frames_with_face=0,
                frames_processed_by_cnn=0,
                frames_without_face=0,
                frames_rejected_as_unusable=0,
                coverage_percentage=0.0,
                avg_face_confidence=0.0,
                avg_gate_value=0.5,
                prediction="Unknown",
                confidence=0.0
            )
            return rep, []

        # Step 2: High-Coverage Adaptive Frame Sampling
        current_coverage = coverage_ratio if coverage_ratio is not None else self.config.FRAME_COVERAGE_RATIO
        sampler = HighCoverageFrameSampler(
            coverage_ratio=current_coverage,
            min_frames=self.config.MIN_FRAMES,
            max_frames=self.config.MAX_FRAMES,
            chunk_size=self.config.FRAME_BATCH_SIZE
        )
        sampling_plan = sampler.create_sampling_plan(total_frames)
        candidate_indices = sampling_plan.candidate_indices
        num_candidates = len(candidate_indices)

        logger.info(
            f"Analyzing video: {vpath.name} (Total: {total_frames} frames, "
            f"Target Coverage: {current_coverage:.0%}, Candidate Frames: {num_candidates})"
        )

        # Tracking counters
        frames_with_face = 0
        frames_without_face = 0
        frames_unusable = 0
        valid_face_tensors: List[torch.Tensor] = []
        valid_frame_metadata: List[Dict[str, Any]] = []

        self.face_aligner.reset_tracking()

        # Step 3 & 4: Frame Reading, Quality Filtering, MTCNN Detection & Alignment
        for frame_idx, success, frame_rgb, ts in reader.read_frames_by_indices(candidate_indices):
            if not success or frame_rgb is None:
                frames_unusable += 1
                continue

            # Quality Check
            q_res = self.quality_filter.evaluate_frame(frame_rgb, frame_idx, ts)
            if not q_res.is_usable:
                frames_unusable += 1
                continue

            # Face Detection & Alignment
            face_res = self.face_aligner.process_frame(frame_rgb, frame_idx, ts)
            if face_res.face_detected and face_res.aligned_face is not None:
                frames_with_face += 1
                # Format tensor: (3, 224, 224) normalized to [0, 1]
                tensor_crop = torch.from_numpy(face_res.aligned_face).permute(2, 0, 1).float() / 255.0
                valid_face_tensors.append(tensor_crop)

                valid_frame_metadata.append({
                    "video_id": vid_id,
                    "frame_index": frame_idx,
                    "timestamp": round(ts, 4),
                    "quality_status": q_res.quality_status,
                    "face_detected": True,
                    "face_confidence": face_res.face_confidence,
                    "bbox": face_res.bbox,
                })
            else:
                frames_without_face += 1

        cnn_processed = len(valid_face_tensors)
        effective_coverage = (cnn_processed / total_frames * 100.0) if total_frames > 0 else 0.0

        if cnn_processed == 0:
            logger.warning(f"No usable faces found in candidate frames for {vpath.name}")
            rep = FrameCoverageReport(
                video_path=str(vpath),
                video_id=vid_id,
                duration_seconds=meta.duration_seconds,
                fps=meta.fps,
                total_video_frames=total_frames,
                candidate_sampled_frames=num_candidates,
                frames_with_face=0,
                frames_processed_by_cnn=0,
                frames_without_face=frames_without_face,
                frames_rejected_as_unusable=frames_unusable,
                coverage_percentage=0.0,
                avg_face_confidence=0.0,
                avg_gate_value=0.5,
                prediction="Real",
                confidence=0.5
            )
            return rep, []

        # Step 5 to 8: End-to-End Neural Processing (Chunked CNNs + Gated Fusion + Temporal Transformer)
        stacked_faces = torch.stack(valid_face_tensors, dim=0).to(self.device)  # (N, 3, 224, 224)

        with torch.no_grad():
            outputs: VisualModelOutput = self.model(stacked_faces)

        logit = float(outputs.logits.item() if outputs.logits.numel() == 1 else outputs.logits[0].item())
        prob = float(outputs.probability.item() if outputs.probability.numel() == 1 else outputs.probability[0].item())
        is_fake = (prob >= 0.5)
        prediction_label = "Fake" if is_fake else "Real"
        confidence = prob if is_fake else (1.0 - prob)

        # Average Gate Value
        if outputs.gate_values is not None and outputs.gate_values.numel() > 0:
            gates_np = outputs.gate_values.view(-1).cpu().numpy()
            avg_gate = float(np.mean(gates_np))
        else:
            gates_np = np.full(cnn_processed, 0.5)
            avg_gate = 0.5

        # Average Face Confidence
        avg_face_conf = float(np.mean([r["face_confidence"] for r in valid_frame_metadata]))

        # Assemble full frame-level records (Section 9 requirement)
        spatial_feats_np = outputs.spatial_features.cpu().numpy() if outputs.spatial_features is not None else None
        freq_feats_np = outputs.frequency_features.cpu().numpy() if outputs.frequency_features is not None else None
        fused_feats_np = outputs.frame_fused_features.cpu().numpy() if outputs.frame_fused_features is not None else None

        for i, meta_rec in enumerate(valid_frame_metadata):
            meta_rec["gate_value"] = round(float(gates_np[i]), 4)
            if spatial_feats_np is not None:
                meta_rec["spatial_feature"] = spatial_feats_np[i]
            if freq_feats_np is not None:
                meta_rec["frequency_feature"] = freq_feats_np[i]
            if fused_feats_np is not None:
                meta_rec["fused_feature"] = fused_feats_np[i]

        # Step 9: Frame Coverage Report Construction
        report = FrameCoverageReport(
            video_path=str(vpath),
            video_id=vid_id,
            duration_seconds=meta.duration_seconds,
            fps=meta.fps,
            total_video_frames=total_frames,
            candidate_sampled_frames=num_candidates,
            frames_with_face=frames_with_face,
            frames_processed_by_cnn=cnn_processed,
            frames_without_face=frames_without_face,
            frames_rejected_as_unusable=frames_unusable,
            coverage_percentage=round(effective_coverage, 2),
            avg_face_confidence=round(avg_face_conf, 4),
            avg_gate_value=round(avg_gate, 4),
            prediction=prediction_label,
            confidence=round(confidence, 4)
        )

        # Save metadata to disk
        if save_metadata:
            csv_path = self.config.METADATA_DIR / f"{vid_id}_frame_metadata.csv"
            json_path = self.config.METADATA_DIR / f"{vid_id}_frame_metadata.json"
            save_frame_metadata_csv_json(valid_frame_metadata, csv_path, json_path)

            feat_path = self.config.FEATURES_DIR / f"{vid_id}_features.npz"
            np.savez_compressed(
                feat_path,
                video_feature=outputs.video_feature.cpu().numpy(),
                frame_fused_features=fused_feats_np,
                gate_values=gates_np
            )

        elapsed = time.time() - start_time
        logger.info(f"Video evaluation completed in {elapsed:.2f}s")
        return report, valid_frame_metadata
