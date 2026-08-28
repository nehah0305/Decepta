"""
Comprehensive Multimodal Video Evaluation Engine.

Produces all 10 canonical outputs:
1. 768-D Visual Feature
2. 768-D Audio Feature
3. 256-D Sync Feature
4. α_v (Visual modality attention weight)
5. α_a (Audio modality attention weight)
6. α_s (Sync modality attention weight)
7. Synchronization Score
8. Final Real Probability
9. Final Fake Probability
10. Final Prediction (REAL or FAKE)

Maintains full timestamped metadata for Visual, Audio, and Sync modalities.
"""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch

from config import DEFAULT_CONFIG, VisualPipelineConfig
from models.multimodal_detector import MultimodalDeepfakeDetector, MultimodalDetectorOutput
from preprocessing.video_reader import VideoReader
from preprocessing.frame_sampler import HighCoverageFrameSampler
from preprocessing.frame_quality import FrameQualityFilter
from preprocessing.face_alignment import FaceAlignmentPipeline
from preprocessing.mouth_extractor import MouthExtractor
from preprocessing.audio_windowing import AudioWindowExtractor, AudioWindowData, VideoAudioResult

logger = logging.getLogger(__name__)


@dataclass
class MultimodalEvaluationReport:
    """Master report containing all 10 canonical outputs and metadata summary."""
    video_path: str
    video_id: str
    duration_seconds: float
    fps: float
    total_video_frames: int
    candidate_frames: int
    faces_detected: int
    mouths_extracted: int
    visual_coverage_pct: float
    has_audio: bool
    num_audio_windows: int
    # The 10 required outputs:
    visual_feature_dim: int
    audio_feature_dim: int
    sync_feature_dim: int
    alpha_v: float
    alpha_a: float
    alpha_s: float
    sync_score: float
    real_probability: float
    fake_probability: float
    prediction: str
    confidence: float

    def print_summary(self):
        """Prints exact standardized multimodal report."""
        sep = "=" * 70
        print("\n" + sep)
        print("          MULTIMODAL DEEPFAKE DETECTION MASTER REPORT")
        print(sep)
        print(f"Video File:                   {Path(self.video_path).name}")
        print(f"Video ID:                     {self.video_id}")
        print(f"Duration / FPS:               {self.duration_seconds:.2f}s @ {self.fps:.2f} FPS")
        print("-" * 70)
        print(" 1. MODALITY COVERAGE SUMMARY:")
        print(f"    - Total Video Frames:     {self.total_video_frames:>6d}")
        print(f"    - Visual Sampled Frames:  {self.candidate_frames:>6d} ({self.visual_coverage_pct:.1f}% coverage)")
        print(f"    - Aligned Faces:          {self.faces_detected:>6d}")
        print(f"    - Mouth ROIs (112x112):   {self.mouths_extracted:>6d}")
        print(f"    - Audio Stream Available: {'YES (16 kHz Mono)' if self.has_audio else 'NO (Missing Audio)'}")
        print(f"    - Audio Windows (4s/2s):  {self.num_audio_windows:>6d}")
        print("-" * 70)
        print(" 2. EXTRACTED REPRESENTATIONS & MODALITY ATTENTION:")
        print(f"    - Visual Feature:         {self.visual_feature_dim:>4d}-D  [Weight alpha_v = {self.alpha_v:.4f}]")
        print(f"    - Audio Feature:          {self.audio_feature_dim:>4d}-D  [Weight alpha_a = {self.alpha_a:.4f}]")
        print(f"    - Sync Feature:           {self.sync_feature_dim:>4d}-D  [Weight alpha_s = {self.alpha_s:.4f}]")
        print(f"    - Audio-Visual Sync Score: {self.sync_score * 100:>6.2f}%")
        print("-" * 70)
        status_tag = "[FAKE]" if self.prediction.upper() == "FAKE" else "[REAL]"
        print(f" 3. FINAL PREDICTION:         {self.prediction.upper()} {status_tag}")
        print(f"    - Real Probability:       {self.real_probability * 100:.2f}%")
        print(f"    - Fake Probability:       {self.fake_probability * 100:.2f}%")
        print(f"    - Decision Confidence:    {self.confidence * 100:.2f}%")
        print(sep + "\n")


class MultimodalDeepfakeEvaluator:
    """
    Evaluator performing high-coverage visual, audio authenticity,
    and audio-visual sync deepfake detection.
    """

    def __init__(
        self,
        model: Optional[MultimodalDeepfakeDetector] = None,
        config: VisualPipelineConfig = DEFAULT_CONFIG,
        checkpoint_path: Optional[Union[str, Path]] = None
    ):
        self.config = config
        self.device = torch.device(config.DEVICE)

        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = MultimodalDeepfakeDetector(
                visual_dim=config.TRANSFORMER_DIM,
                audio_dim=config.AUDIO_FEATURE_DIM,
                sync_dim=config.SYNC_FEATURE_DIM,
                fusion_dim=config.FUSION_DIM,
                mode=config.MODEL_MODE,
                dropout=config.TRANSFORMER_DROPOUT,
                frame_chunk_size=config.FRAME_BATCH_SIZE
            ).to(self.device)

            if checkpoint_path is not None:
                chk = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(chk["model_state_dict"])

        self.model.eval()

        # Preprocessors
        self.sampler = HighCoverageFrameSampler(
            coverage_ratio=config.FRAME_COVERAGE_RATIO,
            min_frames=config.MIN_FRAMES,
            max_frames=config.MAX_FRAMES,
            chunk_size=config.FRAME_BATCH_SIZE
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
        self.mouth_extractor = MouthExtractor(
            mouth_roi_size=(config.MOUTH_ROI_SIZE, config.MOUTH_ROI_SIZE)
        )
        self.audio_extractor = AudioWindowExtractor(
            sample_rate=config.AUDIO_SAMPLE_RATE,
            window_seconds=config.AUDIO_WINDOW_SECONDS,
            hop_seconds=config.AUDIO_HOP_SECONDS,
            n_mels=config.AUDIO_N_MELS,
            n_fft=config.AUDIO_N_FFT,
            hop_length=config.AUDIO_HOP_LENGTH
        )

    def evaluate_video(
        self,
        video_path: Union[str, Path],
        coverage_ratio: Optional[float] = None,
        save_metadata: bool = True
    ) -> Tuple[MultimodalEvaluationReport, MultimodalDetectorOutput, Dict[str, Any]]:
        """
        Executes end-to-end multimodal deepfake detection on an input video.
        """
        vpath = Path(video_path).resolve()
        if not vpath.exists():
            raise FileNotFoundError(f"Video file not found: {vpath}")

        vid_id = vpath.stem
        start_t = time.time()

        # 1. Video Metadata & Sampling
        reader = VideoReader(vpath)
        meta = reader.metadata
        total_frames = meta.total_frames
        ratio = coverage_ratio if coverage_ratio is not None else self.config.FRAME_COVERAGE_RATIO

        sampling_plan = self.sampler.create_sampling_plan(total_frames)
        candidate_indices = sampling_plan.candidate_indices

        # 2. Visual & Mouth Processing
        valid_faces: List[torch.Tensor] = []
        valid_mouths: List[torch.Tensor] = []
        visual_metadata: List[Dict[str, Any]] = []
        visual_timestamps: List[Tuple[int, float]] = []

        self.face_aligner.reset_tracking()

        for frame_idx, success, frame_rgb, ts in reader.read_frames_by_indices(candidate_indices):
            if not success or frame_rgb is None:
                continue

            q_res = self.quality_filter.evaluate_frame(frame_rgb, frame_idx, ts)
            if not q_res.is_usable:
                continue

            face_res = self.face_aligner.process_frame(frame_rgb, frame_idx, ts)
            if face_res.face_detected and face_res.aligned_face is not None:
                face_t = torch.from_numpy(face_res.aligned_face).permute(2, 0, 1).float() / 255.0
                valid_faces.append(face_t)
                visual_timestamps.append((frame_idx, ts))

                # Mouth extraction
                if face_res.landmarks:
                    box = self.mouth_extractor.compute_mouth_box(
                        face_res.landmarks,
                        (self.config.FACE_SIZE, self.config.FACE_SIZE)
                    )
                    if box:
                        mouth_np = self.mouth_extractor.crop_mouth_roi(face_res.aligned_face, box)
                        mouth_t = torch.from_numpy(mouth_np).permute(2, 0, 1).float() / 255.0
                        valid_mouths.append(mouth_t)
                        mouth_status = "extracted"
                    else:
                        valid_mouths.append(torch.zeros(3, self.config.MOUTH_ROI_SIZE, self.config.MOUTH_ROI_SIZE))
                        mouth_status = "failed_box"
                else:
                    valid_mouths.append(torch.zeros(3, self.config.MOUTH_ROI_SIZE, self.config.MOUTH_ROI_SIZE))
                    mouth_status = "no_landmarks"

                visual_metadata.append({
                    "video_id": vid_id,
                    "frame_index": frame_idx,
                    "timestamp": round(ts, 4),
                    "quality_status": q_res.quality_status,
                    "face_detected": True,
                    "face_confidence": face_res.face_confidence,
                    "mouth_status": mouth_status
                })

        has_visual = len(valid_faces) > 0
        face_tensor = torch.stack(valid_faces, dim=0).to(self.device) if has_visual else None
        mouth_tensor = torch.stack(valid_mouths, dim=0).to(self.device) if has_visual else None

        # 3. Audio Extraction & Windowing
        audio_result = self.audio_extractor.process_video_audio(vpath, visual_timestamps=visual_timestamps)
        has_audio = audio_result.audio_available and len(audio_result.windows) > 0

        if has_audio:
            mel_list = [torch.from_numpy(w.mel_spectrogram) for w in audio_result.windows]
            mel_tensor = torch.stack(mel_list, dim=0).to(self.device)  # (W, 128, T)
        else:
            mel_tensor = None

        has_sync = (has_visual and has_audio)
        modality_mask = torch.tensor([has_visual, has_audio, has_sync], dtype=torch.bool, device=self.device)

        # 4. Multimodal Deepfake Detector Forward Pass
        with torch.no_grad():
            outputs: MultimodalDetectorOutput = self.model(
                face_frames=face_tensor,
                mouth_crops=mouth_tensor,
                mel_windows=mel_tensor,
                modality_mask=modality_mask
            )

        prob_fake = float(outputs.probability.item())
        prob_real = 1.0 - prob_fake
        is_fake = (prob_fake >= 0.5)
        pred_label = "Fake" if is_fake else "Real"
        conf = prob_fake if is_fake else prob_real

        a_v = float(outputs.alpha_v.item()) if outputs.alpha_v.numel() == 1 else 0.0
        a_a = float(outputs.alpha_a.item()) if outputs.alpha_a.numel() == 1 else 0.0
        a_s = float(outputs.alpha_s.item()) if outputs.alpha_s.numel() == 1 else 0.0
        s_score = float(outputs.sync_score.item()) if outputs.sync_score.numel() == 1 else 0.5

        coverage_pct = (len(valid_faces) / total_frames * 100.0) if total_frames > 0 else 0.0

        # Construct Master Report
        report = MultimodalEvaluationReport(
            video_path=str(vpath),
            video_id=vid_id,
            duration_seconds=meta.duration_seconds,
            fps=meta.fps,
            total_video_frames=total_frames,
            candidate_frames=len(candidate_indices),
            faces_detected=len(valid_faces),
            mouths_extracted=len(valid_mouths),
            visual_coverage_pct=round(coverage_pct, 2),
            has_audio=has_audio,
            num_audio_windows=len(audio_result.windows) if has_audio else 0,
            visual_feature_dim=outputs.visual_feature.size(-1),
            audio_feature_dim=outputs.audio_feature.size(-1),
            sync_feature_dim=outputs.sync_feature.size(-1),
            alpha_v=round(a_v, 4),
            alpha_a=round(a_a, 4),
            alpha_s=round(a_s, 4),
            sync_score=round(s_score, 4),
            real_probability=round(prob_real, 4),
            fake_probability=round(prob_fake, 4),
            prediction=pred_label,
            confidence=round(conf, 4)
        )

        # Metadata bundle
        metadata_bundle = {
            "report": report.to_dict() if hasattr(report, "to_dict") else asdict(report),
            "visual_frames": visual_metadata,
            "audio_windows": [w.to_dict() for w in audio_result.windows]
        }

        # Save metadata to disk
        if save_metadata:
            csv_path = self.config.METADATA_DIR / f"{vid_id}_multimodal_frames.csv"
            json_path = self.config.METADATA_DIR / f"{vid_id}_multimodal_metadata.json"
            pd.DataFrame(visual_metadata).to_csv(csv_path, index=False)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metadata_bundle, f, indent=2)

        return report, outputs, metadata_bundle
