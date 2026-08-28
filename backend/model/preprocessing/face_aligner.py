"""
Face Alignment Module for Deepfake Detection Preprocessing.

Performs canonical 2D affine similarity transformation using inter-ocular eye vector
to normalize rotation, scale, and translation without altering forensic pixel artifacts.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

from .face_detector import FaceDetection


@dataclass
class AlignedFaceResult:
    """Result of facial alignment transformation."""
    success: bool
    status: str  # "success" | "failed_no_face" | "failed_landmark_error"
    aligned_face_image: Optional[np.ndarray] = None  # RGB image array
    aligned_landmarks: Optional[Dict[str, List[float]]] = None
    transform_matrix: Optional[List[List[float]]] = None
    target_size: Tuple[int, int] = (224, 224)


class FaceAligner:
    """
    Normalizes face images into canonical alignment using 5-point facial landmarks.
    """

    def __init__(
        self,
        output_size: Tuple[int, int] = (224, 224),
        desired_left_eye_ratio: Tuple[float, float] = (0.35, 0.38),
        desired_right_eye_ratio: Tuple[float, float] = (0.65, 0.38),
    ):
        """
        Args:
            output_size: Output canonical face dimension (width, height), e.g. (224, 224) or (256, 256).
            desired_left_eye_ratio: (x_ratio, y_ratio) canonical position of left eye (subject's right).
            desired_right_eye_ratio: (x_ratio, y_ratio) canonical position of right eye (subject's left).
        """
        self.output_size = output_size
        self.desired_left_eye = (
            desired_left_eye_ratio[0] * output_size[0],
            desired_left_eye_ratio[1] * output_size[1]
        )
        self.desired_right_eye = (
            desired_right_eye_ratio[0] * output_size[0],
            desired_right_eye_ratio[1] * output_size[1]
        )
        self.desired_eye_dist = self.desired_right_eye[0] - self.desired_left_eye[0]

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

        # Current eye center, angle, and distance
        eye_center = ((lx + rx) / 2.0, (ly + ry) / 2.0)
        dx = rx - lx
        dy = ry - ly
        current_dist = np.sqrt(dx ** 2 + dy ** 2)

        if current_dist < 1e-4:
            # Fallback if eyes overlap
            angle = 0.0
            scale = 1.0
        else:
            angle = np.degrees(np.arctan2(dy, dx))
            scale = self.desired_eye_dist / current_dist

        # Compute rotation matrix around current eye center
        M = cv2.getRotationMatrix2D(eye_center, angle, scale)

        # Desired eye center position in output canvas
        desired_center_x = (self.desired_left_eye[0] + self.desired_right_eye[0]) / 2.0
        desired_center_y = (self.desired_left_eye[1] + self.desired_right_eye[1]) / 2.0

        # Adjust translation so eye center maps to desired position
        t_x = desired_center_x - eye_center[0]
        t_y = desired_center_y - eye_center[1]

        M[0, 2] += t_x
        M[1, 2] += t_y

        return M

    def transform_points(self, points: Dict[str, List[float]], M: np.ndarray) -> Dict[str, List[float]]:
        """
        Applies 2x3 affine matrix to landmark points dictionary.
        """
        transformed: Dict[str, List[float]] = {}
        for name, pt in points.items():
            x, y = float(pt[0]), float(pt[1])
            homo_pt = np.array([x, y, 1.0])
            new_x = float(np.dot(M[0], homo_pt))
            new_y = float(np.dot(M[1], homo_pt))
            transformed[name] = [round(new_x, 2), round(new_y, 2)]
        return transformed

    def align(
        self,
        image: Union[np.ndarray, Image.Image],
        detection: Optional[FaceDetection]
    ) -> AlignedFaceResult:
        """
        Aligns the face image according to detected eye landmarks.

        Args:
            image: RGB image as NumPy array or PIL Image.
            detection: FaceDetection object or None if no face detected.

        Returns:
            AlignedFaceResult object.
        """
        if detection is None:
            return AlignedFaceResult(
                success=False,
                status="failed_no_face",
                target_size=self.output_size
            )

        if isinstance(image, Image.Image):
            img_np = np.array(image.convert("RGB"))
        else:
            img_np = image

        landmarks = detection.landmarks
        if "left_eye" not in landmarks or "right_eye" not in landmarks:
            return AlignedFaceResult(
                success=False,
                status="failed_landmark_error",
                target_size=self.output_size
            )

        left_eye = tuple(landmarks["left_eye"])
        right_eye = tuple(landmarks["right_eye"])

        # Compute affine similarity transform
        M = self.compute_similarity_transform(left_eye, right_eye)

        # Warp image into canonical face canvas
        # Using cv2.INTER_LANCZOS4 for high-fidelity resampling without smoothing artifacts
        aligned_img = cv2.warpAffine(
            img_np,
            M,
            self.output_size,
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REFLECT_101
        )

        # Map all 5 landmark points into canonical aligned space
        aligned_landmarks = self.transform_points(landmarks, M)

        return AlignedFaceResult(
            success=True,
            status="success",
            aligned_face_image=aligned_img,
            aligned_landmarks=aligned_landmarks,
            transform_matrix=M.tolist(),
            target_size=self.output_size
        )
