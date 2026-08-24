"""
Visual Deepfake Detection Pipeline CLI Runner.

Usage:
    # High-Coverage Analysis on a Video (Default 70% coverage)
    python pipeline_runner.py --input path/to/video.mp4

    # Custom 80% Coverage with specific batch size
    python pipeline_runner.py --input path/to/video.mp4 --coverage 0.80 --batch-size 32

    # Run Frame Coverage Ablation Analysis (30%, 50%, 70%, 90%)
    python pipeline_runner.py --input path/to/video.mp4 --ablation-coverage

    # Run Architectural Ablation (Spatial, Frequency, Gated Fusion, Transformer)
    python pipeline_runner.py --input path/to/video.mp4 --ablation-architecture
"""

import argparse
import logging
from pathlib import Path
import sys

from config import DEFAULT_CONFIG, VisualPipelineConfig
from evaluation.evaluate import VisualDeepfakeEvaluator
from evaluation.coverage_analysis import run_architectural_ablation, run_frame_coverage_ablation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="High-Coverage Visual Deepfake Detection System (Spatial CNN + 2D FFT Frequency CNN + Gated Fusion + Temporal Transformer)"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to the input video file."
    )
    parser.add_argument(
        "--coverage", "-c",
        type=float,
        default=DEFAULT_CONFIG.FRAME_COVERAGE_RATIO,
        help=f"Target frame coverage ratio (default: {DEFAULT_CONFIG.FRAME_COVERAGE_RATIO}, e.g. 0.70 for 70% of video frames)."
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=DEFAULT_CONFIG.MIN_FRAMES,
        help=f"Minimum candidate frames to analyze (default: {DEFAULT_CONFIG.MIN_FRAMES})."
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=DEFAULT_CONFIG.FRAME_BATCH_SIZE,
        help=f"GPU/RAM processing chunk size (default: {DEFAULT_CONFIG.FRAME_BATCH_SIZE}). Controls memory, NOT number of analyzed frames."
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        default=DEFAULT_CONFIG.MODEL_MODE,
        choices=["full", "spatial_only", "frequency_only", "no_gate", "frame_average"],
        help="Model architecture mode (default: 'full')."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to model checkpoint weights (.pt)."
    )
    parser.add_argument(
        "--ablation-coverage",
        action="store_true",
        help="Run comparative ablation analysis across 30%, 50%, 70%, and 90% frame coverage ratios."
    )
    parser.add_argument(
        "--ablation-architecture",
        action="store_true",
        help="Run comparative ablation analysis across model architectures (Spatial, Frequency, No Gate, Frame Average, Full Transformer)."
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Disable writing frame metadata CSV/JSON to disk."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = Path(args.input).resolve()

    if not video_path.exists():
        logger.error(f"Input file not found: {video_path}")
        sys.exit(1)

    config = VisualPipelineConfig(
        FRAME_COVERAGE_RATIO=args.coverage,
        MIN_FRAMES=args.min_frames,
        FRAME_BATCH_SIZE=args.batch_size,
        MODEL_MODE=args.mode
    )

    if args.ablation_coverage:
        run_frame_coverage_ablation(video_path, ratios=[0.30, 0.50, 0.70, 0.90], config=config)
        return

    if args.ablation_architecture:
        run_architectural_ablation(video_path, config=config)
        return

    evaluator = VisualDeepfakeEvaluator(
        config=config,
        checkpoint_path=args.checkpoint
    )

    report, _ = evaluator.evaluate_video(
        video_path=video_path,
        coverage_ratio=args.coverage,
        save_metadata=not args.no_metadata
    )

    # Print Section 12 compliant coverage report
    report.print_summary()


if __name__ == "__main__":
    main()
