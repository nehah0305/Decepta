"""
Dataset Preprocessor Module for Multimodal Deepfake Detection.

Orchestrates sequential PNG frame ingestion, MTCNN face detection, 5-point landmark caching,
canonical face alignment, and mouth ROI extraction with temporal coordinate smoothing.
"""

import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from .face_aligner import AlignedFaceResult, FaceAligner
from .face_detector import FaceDetection, FaceDetector, FrameFaceResult
from .mouth_extractor import MouthExtractor, MouthROIResult


@dataclass
class VideoPreprocessingOutput:
    """Consolidated result of video face and mouth preprocessing."""
    video_id: str
    label: str
    output_directory: str
    num_frames: int
    frame_indices: List[int]
    face_detected: List[bool]
    face_bbox: List[Optional[List[float]]]
    face_confidence: List[Optional[float]]
    facial_landmarks: List[Optional[Dict[str, List[float]]]]
    alignment_status: List[str]
    mouth_roi_coordinates: List[Optional[List[int]]]
    mouth_roi_status: List[str]
    frames_paths: List[str]
    aligned_faces_paths: List[Optional[str]]
    mouth_rois_paths: List[Optional[str]]
    landmarks_file_path: str
    metadata_file_path: str

    def to_dict(self) -> Dict[str, Any]:
        """Converts result to schema matching specification."""
        return {
            "video_id": self.video_id,
            "label": self.label,
            "num_frames": self.num_frames,
            "frame_indices": self.frame_indices,
            "face_detected": self.face_detected,
            "face_bbox": self.face_bbox,
            "face_confidence": self.face_confidence,
            "facial_landmarks": self.facial_landmarks,
            "alignment_status": self.alignment_status,
            "mouth_roi_coordinates": self.mouth_roi_coordinates,
            "mouth_roi_status": self.mouth_roi_status
        }


class VideoFacePreprocessor:
    """
    End-to-End Face and Mouth ROI Preprocessor for deepfake video frame sequences.
    """

    def __init__(
        self,
        face_size: Tuple[int, int] = (224, 224),
        mouth_size: Tuple[int, int] = (96, 96),
        min_face_size: int = 40,
        device: Optional[str] = None
    ):
        """
        Args:
            face_size: Output canonical face dimension (width, height).
            mouth_size: Output mouth ROI dimension (width, height).
            min_face_size: Minimum face detection size.
            device: PyTorch device ('cuda' or 'cpu').
        """
        self.face_detector = FaceDetector(min_face_size=min_face_size, device=device)
        self.face_aligner = FaceAligner(output_size=face_size)
        self.mouth_extractor = MouthExtractor(mouth_roi_size=mouth_size)

    def process_frames_directory(
        self,
        frames_dir: Union[str, Path],
        output_dir: Union[str, Path],
        video_id: str = "video_001",
        label: str = "unknown",
        save_original_frames: bool = True
    ) -> VideoPreprocessingOutput:
        """
        Processes a directory of sampled PNG frames.

        Args:
            frames_dir: Path to directory containing sampled PNG frames (e.g. input/frames/).
            output_dir: Path to output directory (e.g. processed_dataset/video_001/).
            video_id: Identifier for the video.
            label: Classification label (e.g. 'real', 'fake', 0, 1).
            save_original_frames: Whether to copy input frames to output_dir/frames/.

        Returns:
            VideoPreprocessingOutput object with complete forensic metadata.
        """
        in_frames_path = Path(frames_dir).resolve()
        out_root = Path(output_dir).resolve()

        if not in_frames_path.is_dir():
            raise FileNotFoundError(f"Input frames directory not found: {in_frames_path}")

        # Collect all PNG files in chronological natural order
        frame_files = sorted(
            [f for f in in_frames_path.glob("*.png") if f.is_file()],
            key=lambda x: x.name
        )

        if not frame_files:
            raise ValueError(f"No PNG frames found in input directory: {in_frames_path}")

        # Setup output directory structure:
        # output_dir/
        # ├── frames/
        # ├── aligned_faces/
        # ├── mouth_rois/
        # ├── landmarks/
        # └── metadata.json
        out_frames_dir = out_root / "frames"
        out_faces_dir = out_root / "aligned_faces"
        out_mouth_dir = out_root / "mouth_rois"
        out_landmarks_dir = out_root / "landmarks"

        out_frames_dir.mkdir(parents=True, exist_ok=True)
        out_faces_dir.mkdir(parents=True, exist_ok=True)
        out_mouth_dir.mkdir(parents=True, exist_ok=True)
        out_landmarks_dir.mkdir(parents=True, exist_ok=True)

        num_frames = len(frame_files)
        frame_indices: List[int] = []
        face_detected: List[bool] = []
        face_bbox: List[Optional[List[float]]] = []
        face_confidence: List[Optional[float]] = []
        facial_landmarks: List[Optional[Dict[str, List[float]]]] = []
        alignment_status: List[str] = []
        mouth_roi_coordinates: List[Optional[List[int]]] = []
        mouth_roi_status: List[str] = []

        frames_paths: List[str] = []
        aligned_faces_paths: List[Optional[str]] = []
        mouth_rois_paths: List[Optional[str]] = []

        raw_images: List[Image.Image] = []
        detection_results: List[FrameFaceResult] = []
        aligned_face_results: List[AlignedFaceResult] = []
        aligned_face_arrays: List[Optional[np.ndarray]] = []
        aligned_landmarks_list: List[Optional[Dict[str, List[float]]]] = []

        # -------------------------------------------------------------
        # Stage 1: Load Frames & MTCNN Face Detection / Landmark Extraction
        # -------------------------------------------------------------
        prev_primary: Optional[FaceDetection] = None

        for idx, frame_file in enumerate(frame_files):
            frame_indices.append(idx)

            # Standardized output naming matching source frame number or sequential index
            stem = frame_file.stem
            # e.g. frame_01 -> face_01, mouth_01 or face_000001
            suffix_num = stem.replace("frame_", "").replace("frame", "")
            if not suffix_num:
                suffix_num = f"{idx + 1:02d}"

            face_filename = f"face_{suffix_num}.png"
            mouth_filename = f"mouth_{suffix_num}.png"
            frame_dest = out_frames_dir / frame_file.name

            # Copy or retain original frame without modification
            if save_original_frames and str(frame_file) != str(frame_dest):
                shutil.copy2(frame_file, frame_dest)
            frames_paths.append(str(frame_dest))

            # Open image in RGB
            pil_img = Image.open(frame_file).convert("RGB")
            raw_images.append(pil_img)

            # Detect face & 5 landmarks
            det_res = self.face_detector.process_frame(
                pil_img,
                frame_index=idx,
                frame_filename=frame_file.name,
                prev_primary=prev_primary
            )
            detection_results.append(det_res)

            if det_res.detected and det_res.primary_face is not None:
                prev_primary = det_res.primary_face
                face_detected.append(True)
                face_bbox.append(det_res.primary_face.bbox)
                face_confidence.append(det_res.primary_face.confidence)
                facial_landmarks.append(det_res.primary_face.landmarks)
            else:
                face_detected.append(False)
                face_bbox.append(None)
                face_confidence.append(None)
                facial_landmarks.append(None)

        # -------------------------------------------------------------
        # Stage 2: Face Alignment Transformation
        # -------------------------------------------------------------
        for idx, det_res in enumerate(detection_results):
            pil_img = raw_images[idx]
            stem = frame_files[idx].stem
            suffix_num = stem.replace("frame_", "").replace("frame", "")
            if not suffix_num:
                suffix_num = f"{idx + 1:02d}"
            face_filename = f"face_{suffix_num}.png"
            face_dest = out_faces_dir / face_filename

            align_res = self.face_aligner.align(
                image=pil_img,
                detection=det_res.primary_face
            )
            aligned_face_results.append(align_res)
            alignment_status.append(align_res.status)

            if align_res.success and align_res.aligned_face_image is not None:
                # Save aligned full face strictly as lossless PNG
                face_pil = Image.fromarray(align_res.aligned_face_image)
                face_pil.save(face_dest, format="PNG", compress_level=2)
                aligned_faces_paths.append(str(face_dest))
                aligned_face_arrays.append(align_res.aligned_face_image)
                aligned_landmarks_list.append(align_res.aligned_landmarks)
            else:
                aligned_faces_paths.append(None)
                aligned_face_arrays.append(None)
                aligned_landmarks_list.append(None)

        # -------------------------------------------------------------
        # Stage 3: Mouth ROI Extraction with Temporal Smoothing
        # -------------------------------------------------------------
        mouth_results = self.mouth_extractor.extract_sequence(
            aligned_faces=aligned_face_arrays,
            aligned_landmarks_list=aligned_landmarks_list
        )

        for idx, mouth_res in enumerate(mouth_results):
            stem = frame_files[idx].stem
            suffix_num = stem.replace("frame_", "").replace("frame", "")
            if not suffix_num:
                suffix_num = f"{idx + 1:02d}"
            mouth_filename = f"mouth_{suffix_num}.png"
            mouth_dest = out_mouth_dir / mouth_filename

            mouth_roi_status.append(mouth_res.status)
            mouth_roi_coordinates.append(mouth_res.smoothed_coordinates)

            if mouth_res.success and mouth_res.mouth_image is not None:
                # Save mouth crop strictly as lossless PNG
                mouth_pil = Image.fromarray(mouth_res.mouth_image)
                mouth_pil.save(mouth_dest, format="PNG", compress_level=2)
                mouth_rois_paths.append(str(mouth_dest))
            else:
                mouth_rois_paths.append(None)

        # -------------------------------------------------------------
        # Stage 4: Cache Landmarks to landmarks/landmarks.json
        # -------------------------------------------------------------
        landmarks_file = out_landmarks_dir / "landmarks.json"
        landmarks_data = {
            "video_id": video_id,
            "num_frames": num_frames,
            "frames": [r.to_dict() for r in detection_results]
        }
        with open(landmarks_file, "w", encoding="utf-8") as f:
            json.dump(landmarks_data, f, indent=2)

        # -------------------------------------------------------------
        # Stage 5: Generate and Write metadata.json
        # -------------------------------------------------------------
        metadata_file = out_root / "metadata.json"
        output_obj = VideoPreprocessingOutput(
            video_id=video_id,
            label=label,
            output_directory=str(out_root),
            num_frames=num_frames,
            frame_indices=frame_indices,
            face_detected=face_detected,
            face_bbox=face_bbox,
            face_confidence=face_confidence,
            facial_landmarks=facial_landmarks,
            alignment_status=alignment_status,
            mouth_roi_coordinates=mouth_roi_coordinates,
            mouth_roi_status=mouth_roi_status,
            frames_paths=frames_paths,
            aligned_faces_paths=aligned_faces_paths,
            mouth_rois_paths=mouth_rois_paths,
            landmarks_file_path=str(landmarks_file),
            metadata_file_path=str(metadata_file)
        )

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(output_obj.to_dict(), f, indent=2)

        return output_obj


def preprocess_frame_sequence(
    frames_dir: Union[str, Path],
    output_dir: Union[str, Path],
    video_id: str = "video_001",
    label: str = "unknown",
    face_size: Tuple[int, int] = (224, 224),
    mouth_size: Tuple[int, int] = (96, 96),
    device: Optional[str] = None
) -> VideoPreprocessingOutput:
    """
    Convenience function to preprocess a directory of sampled PNG frames.
    """
    preprocessor = VideoFacePreprocessor(
        face_size=face_size,
        mouth_size=mouth_size,
        device=device
    )
    return preprocessor.process_frames_directory(
        frames_dir=frames_dir,
        output_dir=output_dir,
        video_id=video_id,
        label=label
    )
