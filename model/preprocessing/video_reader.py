"""
Video Reader and Metadata Extraction Module.

Safely opens video files with OpenCV, extracts detailed video metadata
(frame count, FPS, duration, resolution, codec), and provides robust,
fault-tolerant frame extraction by chronological index or timestamp.
"""

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union
import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """Detailed metadata for an input video file."""
    video_path: str
    total_frames: int
    fps: float
    duration_seconds: float
    width: int
    height: int
    fourcc: str
    is_valid: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class VideoReader:
    """
    Robust OpenCV-based Video Reader for high-coverage deepfake analysis.
    """

    def __init__(self, video_path: Union[str, Path]):
        self.video_path = Path(video_path).resolve()
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        self.metadata = self._extract_metadata()

    def _extract_metadata(self) -> VideoMetadata:
        """Extracts video stream parameters without reading all frames into memory."""
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video stream for {self.video_path}")
            return VideoMetadata(
                video_path=str(self.video_path),
                total_frames=0,
                fps=0.0,
                duration_seconds=0.0,
                width=0,
                height=0,
                fourcc="UNKNOWN",
                is_valid=False
            )

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

        # Calculate duration safely
        if fps > 0 and total_frames > 0:
            duration = total_frames / fps
        else:
            duration = 0.0

        cap.release()

        return VideoMetadata(
            video_path=str(self.video_path),
            total_frames=total_frames,
            fps=fps if fps > 0 else 25.0,
            duration_seconds=duration,
            width=width,
            height=height,
            fourcc=fourcc,
            is_valid=total_frames > 0 and width > 0 and height > 0
        )

    def read_frame_at_index(self, frame_idx: int) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Reads a single RGB frame at a specific chronological frame index.

        Args:
            frame_idx: 0-indexed frame number.

        Returns:
            Tuple of (success, rgb_image_array, timestamp_seconds).
        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            return False, None, 0.0

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame_bgr = cap.read()
        cap.release()

        if not success or frame_bgr is None:
            return False, None, 0.0

        timestamp = frame_idx / self.metadata.fps if self.metadata.fps > 0 else 0.0
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return True, frame_rgb, timestamp

    def read_frames_by_indices(
        self,
        frame_indices: List[int]
    ) -> Generator[Tuple[int, bool, Optional[np.ndarray], float], None, None]:
        """
        Efficiently reads multiple frames in chronological order using a single VideoCapture handle.

        Args:
            frame_indices: List of frame indices to extract, must be sorted.

        Yields:
            (frame_index, success, rgb_image_array, timestamp_seconds)
        """
        sorted_indices = sorted(frame_indices)
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            for idx in sorted_indices:
                yield idx, False, None, 0.0
            return

        current_cap_pos = 0
        for target_idx in sorted_indices:
            if target_idx < 0 or target_idx >= self.metadata.total_frames:
                yield target_idx, False, None, 0.0
                continue

            # If sequential or close, read forward; otherwise seek
            if target_idx != current_cap_pos:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)

            success, frame_bgr = cap.read()
            current_cap_pos = target_idx + 1

            if not success or frame_bgr is None:
                yield target_idx, False, None, 0.0
                continue

            timestamp = target_idx / self.metadata.fps if self.metadata.fps > 0 else 0.0
            try:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                yield target_idx, True, frame_rgb, timestamp
            except Exception as e:
                logger.warning(f"Error converting frame {target_idx} to RGB: {e}")
                yield target_idx, False, None, timestamp

        cap.release()
