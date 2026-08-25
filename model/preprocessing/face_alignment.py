"""
Face Detection and Canonical Alignment Module using MTCNN.

Performs:
1. Multi-task Cascaded Convolutional Networks (MTCNN) face detection.
2. 5-point facial landmark localization (left eye, right eye, nose, mouth left, mouth right).
3. Canonical 2D affine similarity transformation to (224, 224, 3) preserving pixel fidelity.
4. Consistent primary-face tracking across consecutive frames.
5. Fault-tolerant execution (missing faces in individual frames do not abort the pipeline).
"""

from dataclasses import asdict, dataclass
import logging
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image
import torch

logger = logging.getLogger(__name__)


@dataclass
class AlignedFaceData:
    """Aligned face result and metadata for a single frame."""
    frame_index: int
    timestamp: float
    face_detected: bool
    face_confidence: float
    bbox: Optional[List[float]]
    landmarks: Optional[Dict[str, List[float]]]
    aligned_face: Optional[np.ndarray]  # (224, 224, 3) RGB uint8
    status: str  # "success", "no_face", "alignment_failed", "unusable_frame"

    def to_dict(self) -> dict:
        d = asdict(self)
        # Exclude large raw numpy array from dict for lightweight metadata serialization
        d.pop("aligned_face", None)
        return d


class FaceAlignmentPipeline:
    """
    Handles face detection, primary face selection/tracking, and similarity alignment to 224x224.
    """

    def __init__(
        self,
        target_size: int = 224,
        min_face_size: int = 40,
        thresholds: Tuple[float, float, float] = (0.6, 0.7, 0.7),
        device: Optional[Union[str, torch.device]] = None
    ):
        """
        Args:
            target_size: Output canonical square dimension (default 224).
            min_face_size: MTCNN minimum detectable face size.
            thresholds: P-Net, R-Net, O-Net detection thresholds.
            device: torch device for MTCNN inference.
        """
        from facenet_pytorch import MTCNN

        self.target_size = target_size
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.mtcnn = MTCNN(
            keep_all=True,
            min_face_size=min_face_size,
            thresholds=list(thresholds),
            post_process=False,
            device=self.device
        )

        # Canonical eye coordinates (fraction of target canvas)
        self.desired_left_eye = (0.35 * target_size, 0.38 * target_size)
        self.desired_right_eye = (0.65 * target_size, 0.38 * target_size)
        self.desired_eye_dist = self.desired_right_eye[0] - self.desired_left_eye[0]

        # Temporal tracking memory
        self.prev_primary_bbox: Optional[List[float]] = None

    def reset_tracking(self):
        """Resets temporal face tracking for a new video sequence."""
        self.prev_primary_bbox = None

    def compute_similarity_transform(
        self,
        left_eye: Tuple[float, float],
        right_eye: Tuple[float, float]
    ) -> np.ndarray:
        """
        Computes 2x3 affine similarity transform matrix based on eye coordinates.
        """
        lx, ly = float(left_eye[0]), float(left_eye[1])
        rx, ry = float(right_eye[0]), float(right_eye[1])

        eye_center = ((lx + rx) / 2.0, (ly + ry) / 2.0)
        dx = rx - lx
        dy = ry - ly
        current_dist = np.sqrt(dx ** 2 + dy ** 2)

        if current_dist < 1e-4:
            angle = 0.0
            scale = 1.0
        else:
            angle = float(np.degrees(np.arctan2(dy, dx)))
            scale = float(self.desired_eye_dist / current_dist)

        M = cv2.getRotationMatrix2D(eye_center, angle, scale)

        desired_center_x = (self.desired_left_eye[0] + self.desired_right_eye[0]) / 2.0
        desired_center_y = (self.desired_left_eye[1] + self.desired_right_eye[1]) / 2.0

        M[0, 2] += (desired_center_x - eye_center[0])
        M[1, 2] += (desired_center_y - eye_center[1])

        return M

    def _select_primary_face(
        self,
        boxes: np.ndarray,
        probs: np.ndarray,
        landmarks: np.ndarray,
        frame_shape: Tuple[int, int]
    ) -> Tuple[List[float], float, Dict[str, List[float]]]:
        """
        Selects primary face using confidence, bounding box area, and temporal spatial proximity.
        """
        num_faces = len(boxes)
        if num_faces == 1:
            best_idx = 0
        else:
            best_idx = 0
            best_score = -float("inf")
            h, w = frame_shape[:2]

            for i in range(num_faces):
                box = boxes[i]
                prob = probs[i] if probs is not None else 1.0
                area = max(0.0, (box[2] - box[0]) * (box[3] - box[1])) / (w * h + 1e-6)

                # Proximity score to previous frame's primary face
                proximity_score = 0.0
                if self.prev_primary_bbox is not None:
                    prev_center = (
                        (self.prev_primary_bbox[0] + self.prev_primary_bbox[2]) / 2.0,
                        (self.prev_primary_bbox[1] + self.prev_primary_bbox[3]) / 2.0
                    )
                    curr_center = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
                    dist = np.sqrt(((curr_center[0] - prev_center[0]) / w) ** 2 + ((curr_center[1] - prev_center[1]) / h) ** 2)
                    proximity_score = max(0.0, 1.0 - dist)

                # Composite score: confidence * 0.4 + area * 0.3 + proximity * 0.3
                score = (prob * 0.4) + (min(1.0, area * 4.0) * 0.3) + (proximity_score * 0.3)
                if score > best_score:
                    best_score = score
                    best_idx = i

        chosen_box = [round(float(c), 2) for c in boxes[best_idx]]
        chosen_prob = round(float(probs[best_idx]), 4) if probs is not None else 1.0
        self.prev_primary_bbox = chosen_box

        # Format landmarks
        lm = landmarks[best_idx]
        landmarks_dict = {
            "left_eye": [round(float(lm[0][0]), 2), round(float(lm[0][1]), 2)],
            "right_eye": [round(float(lm[1][0]), 2), round(float(lm[1][1]), 2)],
            "nose": [round(float(lm[2][0]), 2), round(float(lm[2][1]), 2)],
            "mouth_left": [round(float(lm[3][0]), 2), round(float(lm[3][1]), 2)],
            "mouth_right": [round(float(lm[4][0]), 2), round(float(lm[4][1]), 2)],
        }

        return chosen_box, chosen_prob, landmarks_dict

    def process_frame(
        self,
        frame_rgb: Optional[np.ndarray],
        frame_index: int,
        timestamp: float
    ) -> AlignedFaceData:
        """
        Detects, aligns, and returns standard 224x224 RGB face crop.
        Fault-tolerant: never raises an exception if detection or alignment fails.
        """
        if frame_rgb is None:
            return AlignedFaceData(
                frame_index=frame_index,
                timestamp=timestamp,
                face_detected=False,
                face_confidence=0.0,
                bbox=None,
                landmarks=None,
                aligned_face=None,
                status="unusable_frame"
            )

        try:
            pil_img = Image.fromarray(frame_rgb)
            boxes, probs, landmarks = self.mtcnn.detect(pil_img, landmarks=True)

            if boxes is None or len(boxes) == 0:
                return AlignedFaceData(
                    frame_index=frame_index,
                    timestamp=timestamp,
                    face_detected=False,
                    face_confidence=0.0,
                    bbox=None,
                    landmarks=None,
                    aligned_face=None,
                    status="no_face"
                )

            # Select primary face
            box, conf, lm_dict = self._select_primary_face(boxes, probs, landmarks, frame_rgb.shape)

            # Align using similarity transform
            left_eye = tuple(lm_dict["left_eye"])
            right_eye = tuple(lm_dict["right_eye"])
            M = self.compute_similarity_transform(left_eye, right_eye)

            aligned = cv2.warpAffine(
                frame_rgb,
                M,
                (self.target_size, self.target_size),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REFLECT_101
            )

            return AlignedFaceData(
                frame_index=frame_index,
                timestamp=timestamp,
                face_detected=True,
                face_confidence=conf,
                bbox=box,
                landmarks=lm_dict,
                aligned_face=aligned,
                status="success"
            )

        except Exception as e:
            logger.warning(f"Face detection/alignment exception on frame {frame_index}: {e}")
            return AlignedFaceData(
                frame_index=frame_index,
                timestamp=timestamp,
                face_detected=False,
                face_confidence=0.0,
                bbox=None,
                landmarks=None,
                aligned_face=None,
                status="alignment_failed"
            )
