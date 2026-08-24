"""
Frame Quality Filtering Module.

Performs lightweight, non-aggressive quality checks on sampled video frames.
Only genuinely unusable frames (corrupted, unreadable, totally pitch black,
or entirely blown out white) are filtered out.

Subtle blur, minor compression artifacts, or slight lighting variations
are intentionally PRESERVED as they often contain forensic deepfake evidence.
"""

from dataclasses import asdict, dataclass
import logging
from typing import Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FrameQualityResult:
    """Quality assessment metadata for an individual video frame."""
    frame_index: int
    timestamp: float
    is_usable: bool
    quality_status: str  # "valid", "unreadable", "corrupted", "extreme_dark", "extreme_bright"
    mean_brightness: float
    blur_score: float  # Variance of Laplacian (higher = sharper)

    def to_dict(self) -> dict:
        return asdict(self)


class FrameQualityFilter:
    """
    Evaluates frame usability while protecting subtle forensic artifacts from aggressive filtering.
    """

    def __init__(
        self,
        min_brightness: float = 5.0,
        max_brightness: float = 250.0,
        blur_floor: float = 5.0,
        non_aggressive: bool = True
    ):
        """
        Args:
            min_brightness: Minimum mean pixel luminance [0-255]. Frames below are pitch black.
            max_brightness: Maximum mean pixel luminance [0-255]. Frames above are completely blown out.
            blur_floor: Extreme blur cutoff below which image contains no structural detail.
            non_aggressive: When True, preserves slightly blurry/dark/compressed frames.
        """
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.blur_floor = blur_floor
        self.non_aggressive = non_aggressive

    def evaluate_frame(
        self,
        frame_rgb: Optional[np.ndarray],
        frame_index: int,
        timestamp: float
    ) -> FrameQualityResult:
        """
        Assesses whether a frame is computationally usable for face detection and CNN analysis.

        Args:
            frame_rgb: RGB image numpy array or None if read failed.
            frame_index: Chronological frame position.
            timestamp: Frame timestamp in seconds.

        Returns:
            FrameQualityResult object.
        """
        if frame_rgb is None or not isinstance(frame_rgb, np.ndarray):
            return FrameQualityResult(
                frame_index=frame_index,
                timestamp=timestamp,
                is_usable=False,
                quality_status="unreadable",
                mean_brightness=0.0,
                blur_score=0.0
            )

        if frame_rgb.size == 0 or len(frame_rgb.shape) != 3 or frame_rgb.shape[2] != 3:
            return FrameQualityResult(
                frame_index=frame_index,
                timestamp=timestamp,
                is_usable=False,
                quality_status="corrupted",
                mean_brightness=0.0,
                blur_score=0.0
            )

        try:
            # Compute luminance metrics
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            mean_brightness = float(np.mean(gray))

            # Extreme darkness check (pure black frames)
            if mean_brightness < self.min_brightness:
                return FrameQualityResult(
                    frame_index=frame_index,
                    timestamp=timestamp,
                    is_usable=False,
                    quality_status="extreme_dark",
                    mean_brightness=round(mean_brightness, 2),
                    blur_score=0.0
                )

            # Extreme brightness check (pure white frames)
            if mean_brightness > self.max_brightness:
                return FrameQualityResult(
                    frame_index=frame_index,
                    timestamp=timestamp,
                    is_usable=False,
                    quality_status="extreme_bright",
                    mean_brightness=round(mean_brightness, 2),
                    blur_score=0.0
                )

            # Measure sharpness via Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            blur_score = float(laplacian.var())

            # Only reject if blur is catastrophic (e.g. solid block of color)
            # In non-aggressive mode, we accept frames with blur_score >= blur_floor
            if not self.non_aggressive and blur_score < self.blur_floor:
                return FrameQualityResult(
                    frame_index=frame_index,
                    timestamp=timestamp,
                    is_usable=False,
                    quality_status="extreme_blur",
                    mean_brightness=round(mean_brightness, 2),
                    blur_score=round(blur_score, 2)
                )

            return FrameQualityResult(
                frame_index=frame_index,
                timestamp=timestamp,
                is_usable=True,
                quality_status="valid",
                mean_brightness=round(mean_brightness, 2),
                blur_score=round(blur_score, 2)
            )

        except Exception as e:
            logger.warning(f"Error evaluating quality for frame {frame_index}: {e}")
            return FrameQualityResult(
                frame_index=frame_index,
                timestamp=timestamp,
                is_usable=False,
                quality_status="corrupted",
                mean_brightness=0.0,
                blur_score=0.0
            )
