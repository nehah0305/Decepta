"""
MTCNN Face Detector and Coarse Landmark Extraction Module.

Detects faces, bounding boxes, confidence scores, and 5 facial landmarks:
left eye, right eye, nose, left mouth corner, right mouth corner.
Includes robust primary-face selection and tracking across video frame sequences.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch


@dataclass
class FaceDetection:
    """Detection results for a single face."""
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    landmarks: Dict[str, List[float]]  # {"left_eye": [x, y], "right_eye": [x, y], "nose": [x, y], "mouth_left": [x, y], "mouth_right": [x, y]}


@dataclass
class FrameFaceResult:
    """Detection result for a single video frame."""
    frame_index: int
    frame_filename: str
    detected: bool
    primary_face: Optional[FaceDetection] = None
    all_faces: List[FaceDetection] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert detection result to serializable dictionary."""
        return {
            "frame_index": self.frame_index,
            "frame_filename": self.frame_filename,
            "detected": self.detected,
            "primary_face": asdict(self.primary_face) if self.primary_face else None,
            "all_faces": [asdict(f) for f in self.all_faces]
        }


class FaceDetector:
    """
    MTCNN-based Face and Landmark Detector with sequence-level primary-face tracking.
    """

    def __init__(
        self,
        min_face_size: int = 40,
        thresholds: Tuple[float, float, float] = (0.6, 0.7, 0.7),
        device: Optional[Union[str, torch.device]] = None
    ):
        """
        Args:
            min_face_size: Minimum face size in pixels to detect.
            thresholds: P-Net, R-Net, O-Net detection thresholds.
            device: 'cuda', 'cpu', or None for auto-detection.
        """
        from facenet_pytorch import MTCNN

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

    def detect_faces(self, image: Union[Image.Image, np.ndarray]) -> List[FaceDetection]:
        """
        Detects all faces in a PIL Image or NumPy array (RGB).

        Returns:
            List of FaceDetection objects.
        """
        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image)
        else:
            pil_img = image

        boxes, probs, landmarks = self.mtcnn.detect(pil_img, landmarks=True)

        if boxes is None or len(boxes) == 0:
            return []

        results: List[FaceDetection] = []
        for i in range(len(boxes)):
            box = [round(float(coord), 2) for coord in boxes[i]]
            conf = round(float(probs[i]), 4) if probs is not None else 1.0

            if landmarks is not None and len(landmarks) > i and landmarks[i] is not None:
                lm = landmarks[i]  # shape: (5, 2)
                landmarks_dict = {
                    "left_eye": [round(float(lm[0][0]), 2), round(float(lm[0][1]), 2)],
                    "right_eye": [round(float(lm[1][0]), 2), round(float(lm[1][1]), 2)],
                    "nose": [round(float(lm[2][0]), 2), round(float(lm[2][1]), 2)],
                    "mouth_left": [round(float(lm[3][0]), 2), round(float(lm[3][1]), 2)],
                    "mouth_right": [round(float(lm[4][0]), 2), round(float(lm[4][1]), 2)],
                }
            else:
                landmarks_dict = {
                    "left_eye": [0.0, 0.0],
                    "right_eye": [0.0, 0.0],
                    "nose": [0.0, 0.0],
                    "mouth_left": [0.0, 0.0],
                    "mouth_right": [0.0, 0.0],
                }

            results.append(
                FaceDetection(
                    bbox=box,
                    confidence=conf,
                    landmarks=landmarks_dict
                )
            )

        return results

    def select_primary_face(
        self,
        faces: List[FaceDetection],
        prev_primary: Optional[FaceDetection] = None,
        image_shape: Optional[Tuple[int, int]] = None
    ) -> Optional[FaceDetection]:
        """
        Selects the primary subject face using a composite score of confidence,
        face bounding box area, and temporal spatial continuity.

        Args:
            faces: List of detected faces in the current frame.
            prev_primary: Primary face from previous frame for temporal tracking.
            image_shape: (height, width) for normalized distance penalty.

        Returns:
            The chosen primary FaceDetection, or None if faces is empty.
        """
        if not faces:
            return None

        if len(faces) == 1:
            return faces[0]

        best_face = faces[0]
        best_score = -float("inf")

        diag = 1000.0
        if image_shape:
            h, w = image_shape
            diag = np.sqrt(h ** 2 + w ** 2)

        for face in faces:
            x1, y1, x2, y2 = face.bbox
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            area = width * height
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0

            # Base score: Confidence * sqrt(Area)
            score = face.confidence * np.sqrt(area)

            # Temporal continuity penalty if we tracked a face in the previous frame
            if prev_primary is not None:
                px1, py1, px2, py2 = prev_primary.bbox
                p_center_x = (px1 + px2) / 2.0
                p_center_y = (py1 + py2) / 2.0
                dist = np.sqrt((center_x - p_center_x) ** 2 + (center_y - p_center_y) ** 2)
                # Distance penalty normalized by diagonal
                dist_penalty = (dist / diag) * 50.0
                score -= dist_penalty

            if score > best_score:
                best_score = score
                best_face = face

        return best_face

    def process_frame(
        self,
        image: Union[Image.Image, np.ndarray, Path, str],
        frame_index: int,
        frame_filename: str,
        prev_primary: Optional[FaceDetection] = None
    ) -> FrameFaceResult:
        """
        Processes a single frame: detects faces and selects primary face.
        """
        if isinstance(image, (str, Path)):
            pil_img = Image.open(str(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image)
        else:
            pil_img = image

        w, h = pil_img.size
        faces = self.detect_faces(pil_img)

        if not faces:
            return FrameFaceResult(
                frame_index=frame_index,
                frame_filename=frame_filename,
                detected=False,
                primary_face=None,
                all_faces=[]
            )

        primary = self.select_primary_face(faces, prev_primary=prev_primary, image_shape=(h, w))

        return FrameFaceResult(
            frame_index=frame_index,
            frame_filename=frame_filename,
            detected=True,
            primary_face=primary,
            all_faces=faces
        )
