"""
Mouth ROI Extraction and Temporal Smoothing Module for Deepfake Detection.

Extracts canonical mouth region from aligned full face canvases with optional
temporal coordinate smoothing across the video frame sequence.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image


@dataclass
class MouthROIResult:
    """Result of mouth ROI extraction for a single frame."""
    success: bool
    status: str  # "success" | "failed_no_face" | "failed_landmark_error"
    mouth_image: Optional[np.ndarray] = None  # RGB image array
    raw_coordinates: Optional[List[int]] = None  # [x1, y1, x2, y2]
    smoothed_coordinates: Optional[List[int]] = None  # [x1, y1, x2, y2]
    target_size: Tuple[int, int] = (96, 96)


class MouthExtractor:
    """
    Localizes and extracts mouth regions from aligned faces, with temporal coordinate smoothing.
    """

    def __init__(
        self,
        mouth_roi_size: Tuple[int, int] = (96, 96),
        scale_factor: float = 1.6,
        vertical_offset_ratio: float = 0.05,
        smoothing_window: int = 3,
        smoothing_alpha: float = 0.7
    ):
        """
        Args:
            mouth_roi_size: Target size (width, height) of the extracted mouth crop (e.g. 96x96 or 112x112).
            scale_factor: Multiplier applied to inter-mouth-corner distance for ROI bounding box width.
            vertical_offset_ratio: Slight vertical downward adjustment to ensure full lower lip capture.
            smoothing_window: Moving average window size for coordinate temporal smoothing (odd integer).
            smoothing_alpha: EMA smoothing factor (0.0 < alpha <= 1.0).
        """
        self.mouth_roi_size = mouth_roi_size
        self.scale_factor = scale_factor
        self.vertical_offset_ratio = vertical_offset_ratio
        self.smoothing_window = max(1, smoothing_window)
        self.smoothing_alpha = smoothing_alpha

    def compute_mouth_box(
        self,
        aligned_landmarks: Dict[str, List[float]],
        face_shape: Tuple[int, int]
    ) -> Optional[List[int]]:
        """
        Calculates raw mouth bounding box [x1, y1, x2, y2] from aligned landmarks.
        """
        if "mouth_left" not in aligned_landmarks or "mouth_right" not in aligned_landmarks:
            return None

        ml_x, ml_y = aligned_landmarks["mouth_left"]
        mr_x, mr_y = aligned_landmarks["mouth_right"]

        # Center point between mouth corners
        center_x = (ml_x + mr_x) / 2.0
        center_y = (ml_y + mr_y) / 2.0 + (self.vertical_offset_ratio * self.mouth_roi_size[1])

        # Width based on mouth corner distance with margin
        mouth_width = max(abs(mr_x - ml_x) * self.scale_factor, float(self.mouth_roi_size[0]))
        mouth_height = mouth_width * (self.mouth_roi_size[1] / self.mouth_roi_size[0])

        x1 = int(round(center_x - mouth_width / 2.0))
        y1 = int(round(center_y - mouth_height / 2.0))
        x2 = int(round(x1 + mouth_width))
        y2 = int(round(y1 + mouth_height))

        return [x1, y1, x2, y2]

    def smooth_coordinates_sequence(
        self,
        raw_boxes: List[Optional[List[int]]]
    ) -> List[Optional[List[int]]]:
        """
        Applies lightweight temporal smoothing across consecutive mouth bounding box coordinates
        to eliminate crop jitter without altering image pixel data.
        """
        n_frames = len(raw_boxes)
        smoothed: List[Optional[List[int]]] = [None] * n_frames

        # Extract continuous segments of valid detections
        valid_indices = [i for i, b in enumerate(raw_boxes) if b is not None]
        if not valid_indices:
            return smoothed

        # Apply Exponential Moving Average (EMA) forward-pass
        last_box: Optional[np.ndarray] = None
        for idx in valid_indices:
            current_box = np.array(raw_boxes[idx], dtype=np.float64)
            if last_box is None:
                smoothed_box = current_box
            else:
                smoothed_box = (
                    self.smoothing_alpha * current_box + (1.0 - self.smoothing_alpha) * last_box
                )
            last_box = smoothed_box
            smoothed[idx] = [int(round(c)) for c in smoothed_box]

        return smoothed

    def crop_mouth_roi(
        self,
        aligned_face: np.ndarray,
        box: List[int]
    ) -> np.ndarray:
        """
        Safely crops the mouth ROI from the aligned face image, applying padding if out of bounds.
        """
        x1, y1, x2, y2 = box
        fh, fw = aligned_face.shape[:2]

        # Calculate padding needed if bounding box falls outside face canvas
        pad_top = max(0, -y1)
        pad_bottom = max(0, y2 - fh)
        pad_left = max(0, -x1)
        pad_right = max(0, x2 - fw)

        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            padded_face = cv2.copyMakeBorder(
                aligned_face,
                pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_REFLECT_101
            )
            crop_y1 = y1 + pad_top
            crop_y2 = y2 + pad_top
            crop_x1 = x1 + pad_left
            crop_x2 = x2 + pad_left
            crop = padded_face[crop_y1:crop_y2, crop_x1:crop_x2]
        else:
            crop = aligned_face[y1:y2, x1:x2]

        if crop.size == 0:
            crop = np.zeros((self.mouth_roi_size[1], self.mouth_roi_size[0], 3), dtype=np.uint8)

        # Standardize to target size using high-fidelity LANCZOS4 interpolation
        if crop.shape[1] != self.mouth_roi_size[0] or crop.shape[0] != self.mouth_roi_size[1]:
            crop = cv2.resize(crop, self.mouth_roi_size, interpolation=cv2.INTER_LANCZOS4)

        return crop

    def extract_sequence(
        self,
        aligned_faces: List[Optional[np.ndarray]],
        aligned_landmarks_list: List[Optional[Dict[str, List[float]]]]
    ) -> List[MouthROIResult]:
        """
        Extracts temporal mouth ROIs across a sequence of aligned faces.

        Args:
            aligned_faces: List of aligned face image arrays (or None if detection failed).
            aligned_landmarks_list: List of aligned landmark dictionaries (or None).

        Returns:
            List of MouthROIResult for every frame in the sequence.
        """
        n_frames = len(aligned_faces)
        raw_boxes: List[Optional[List[int]]] = [None] * n_frames

        # Step 1: Compute raw bounding boxes
        for i in range(n_frames):
            face = aligned_faces[i]
            lms = aligned_landmarks_list[i]
            if face is not None and lms is not None:
                box = self.compute_mouth_box(lms, (face.shape[0], face.shape[1]))
                raw_boxes[i] = box

        # Step 2: Temporal Coordinate Smoothing across the sequence
        smoothed_boxes = self.smooth_coordinates_sequence(raw_boxes)

        # Step 3: Extract final mouth crops
        results: List[MouthROIResult] = []
        for i in range(n_frames):
            face = aligned_faces[i]
            raw_b = raw_boxes[i]
            smooth_b = smoothed_boxes[i]

            if face is None or smooth_b is None:
                results.append(
                    MouthROIResult(
                        success=False,
                        status="failed_no_face" if face is None else "failed_landmark_error",
                        target_size=self.mouth_roi_size
                    )
                )
            else:
                mouth_img = self.crop_mouth_roi(face, smooth_b)
                results.append(
                    MouthROIResult(
                        success=True,
                        status="success",
                        mouth_image=mouth_img,
                        raw_coordinates=raw_b,
                        smoothed_coordinates=smooth_b,
                        target_size=self.mouth_roi_size
                    )
                )

        return results
