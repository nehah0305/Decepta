"""
Main CLI entry point for the Multimodal Deepfake Detection Preprocessing Pipeline.

Usage:
    python main.py --input path/to/video.mp4 --output-dir output --fps 5
"""

import argparse
import json
import sys
from pathlib import Path

from preprocessing import MultimodalPreprocessor, PreprocessingOutput


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multimodal Deepfake Detection Preprocessing Pipeline: Separate Video Frames and Audio Stream."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to the input video file (containing video and audio streams)."
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="output",
        help="Output directory (default: 'output'). Frames stored in output/frames/, audio in output/audio/audio.wav."
    )
    parser.add_argument(
        "--fps", "-f",
        type=float,
        default=5.0,
        help="Target frame extraction rate in frames per second (FPS), e.g. 5.0 or 10.0 (default: 5.0)."
    )
    parser.add_argument(
        "--sample-rate", "-sr",
        type=int,
        default=None,
        help="Audio sample rate in Hz (e.g., 16000 for speech models, 44100). Default is native sample rate."
    )
    parser.add_argument(
        "--channels", "-c",
        type=int,
        default=None,
        help="Audio channel count (e.g. 1 for mono, 2 for stereo). Default is native channels."
    )
    parser.add_argument(
        "--ffmpeg-path",
        type=str,
        default=None,
        help="Optional custom path to the FFmpeg executable."
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Disable writing metadata.json into the output directory."
    )
    return parser.parse_args()


def print_summary(result: PreprocessingOutput):
    """Prints a clean summary of the preprocessing results to stdout."""
    sep = "=" * 65
    print("\n" + sep)
    print("   MULTIMODAL DEEPFAKE DETECTION PREPROCESSING COMPLETED")
    print(sep)
    print(f"Status:                 {result.status.upper()}")
    print(f"Input Video:            {result.input_video_path}")
    print(f"Output Directory:       {result.output_directory}")
    print(f"Processing Time:        {result.processing_time_seconds:.2f}s")
    print("-" * 65)
    print(" 1. VIDEO MODALITY (FRAME EXTRACTION)")
    print(f"    - Storage Path:     {result.frames_directory}")
    print(f"    - Format:           PNG (.png) [Lossless/Forensic Preserved]")
    print(f"    - Original Video:   {result.resolution['width']}x{result.resolution['height']} @ {result.original_fps:.2f} FPS")
    print(f"    - Extracted Rate:   {result.extracted_fps:.2f} FPS")
    print(f"    - Video Duration:   {result.video_duration_seconds:.2f}s")
    print(f"    - Total Extracted:  {result.total_extracted_frames} frames")
    if result.frame_filenames:
        print(f"    - Frame Sequence:   {result.frame_filenames[0]} ... {result.frame_filenames[-1]}")
    print("-" * 65)
    print(" 2. AUDIO MODALITY (AUDIO EXTRACTION)")
    if result.has_audio:
        print(f"    - Storage Path:     {result.audio_file_path}")
        print(f"    - Format:           WAV (.wav) [PCM 16-bit]")
        print(f"    - Sample Rate:      {result.audio_sample_rate} Hz")
        print(f"    - Channels:         {result.audio_channels}")
        print(f"    - Audio Duration:   {result.audio_duration_seconds:.2f}s")
        print(f"    - Temporal Sync:    Preserved (starts at 00:00:00)")
    else:
        print("    - Audio Track:      No audio stream found in input video.")
    print("-" * 65)
    print(f" 3. METADATA FILE:      {result.metadata_file_path}")
    print(sep + "\n")


def main():
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.is_file():
        print(f"[ERROR] Input video file does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)

    try:
        preprocessor = MultimodalPreprocessor(
            extracted_fps=args.fps,
            audio_sample_rate=args.sample_rate,
            audio_channels=args.channels,
            ffmpeg_path=args.ffmpeg_path
        )

        result = preprocessor.process(
            video_path=input_path,
            output_dir=args.output_dir,
            save_metadata=not args.no_metadata
        )

        print_summary(result)

    except Exception as e:
        print(f"\n[FATAL ERROR] Preprocessing pipeline failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
