"""
CLI Entry Point for Video Face Alignment and Mouth ROI Preprocessing.

Usage:
    python preprocess_faces_cli.py --input-dir input/frames --output-dir processed_dataset/video_001 --video-id video_001 --label real
"""

import argparse
import sys
from pathlib import Path

from preprocessing import VideoFacePreprocessor, VideoPreprocessingOutput


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess sampled PNG frames: MTCNN detection -> Alignment -> Mouth ROI extraction -> Metadata."
    )
    parser.add_argument(
        "--input-dir", "-i",
        type=str,
        required=True,
        help="Path to directory containing sampled PNG frames (e.g., input/frames/)."
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="processed_dataset/video_001",
        help="Path to output directory (default: 'processed_dataset/video_001')."
    )
    parser.add_argument(
        "--video-id",
        type=str,
        default="video_001",
        help="Identifier for the video sequence (default: 'video_001')."
    )
    parser.add_argument(
        "--label",
        type=str,
        default="unknown",
        help="Ground-truth classification label, e.g. 'real', 'fake' (default: 'unknown')."
    )
    parser.add_argument(
        "--face-size",
        type=int,
        default=224,
        help="Output resolution for aligned full face (default: 224 for 224x224)."
    )
    parser.add_argument(
        "--mouth-size",
        type=int,
        default=96,
        help="Output resolution for mouth ROI (default: 96 for 96x96)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="PyTorch device ('cuda' or 'cpu'). Auto-detected if not specified."
    )
    return parser.parse_args()


def print_summary(result: VideoPreprocessingOutput):
    """Prints a structured summary of the face & mouth preprocessing results."""
    sep = "=" * 68
    print("\n" + sep)
    print("   VIDEO FACE & MOUTH ROI PREPROCESSING COMPLETED")
    print(sep)
    print(f"Video ID:               {result.video_id}")
    print(f"Label:                  {result.label}")
    print(f"Total Sampled Frames:   {result.num_frames}")
    detected_count = sum(result.face_detected)
    print(f"Faces Detected:         {detected_count} / {result.num_frames} frames")
    print("-" * 68)
    print(" 1. SAVED OUTPUT DIRECTORIES")
    print(f"    - Frames:           {result.output_directory}/frames/")
    print(f"    - Aligned Faces:    {result.output_directory}/aligned_faces/ ({result.face_size if hasattr(result, 'face_size') else '224x224'} PNG)")
    print(f"    - Mouth ROIs:       {result.output_directory}/mouth_rois/ ({result.mouth_size if hasattr(result, 'mouth_size') else '96x96'} PNG)")
    print(f"    - Landmarks Cache:  {result.landmarks_file_path}")
    print(f"    - Metadata JSON:    {result.metadata_file_path}")
    print("-" * 68)
    print(" 2. SEQUENCE SUMMARY")
    for i in range(min(5, result.num_frames)):
        status_face = "DETECTED" if result.face_detected[i] else "NO_FACE"
        conf = f"{result.face_confidence[i]:.4f}" if result.face_confidence[i] is not None else "N/A"
        align = result.alignment_status[i]
        mouth_status = result.mouth_roi_status[i]
        print(f"    Frame {result.frame_indices[i]:02d}: Face={status_face} (conf={conf}), Align={align}, Mouth={mouth_status}")
    if result.num_frames > 5:
        print(f"    ... [{result.num_frames - 5} remaining frames processed identically]")
    print(sep + "\n")


def main():
    args = parse_args()
    input_path = Path(args.input_dir)

    if not input_path.is_dir():
        print(f"[ERROR] Input frames directory does not exist: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        preprocessor = VideoFacePreprocessor(
            face_size=(args.face_size, args.face_size),
            mouth_size=(args.mouth_size, args.mouth_size),
            device=args.device
        )

        result = preprocessor.process_frames_directory(
            frames_dir=input_path,
            output_dir=args.output_dir,
            video_id=args.video_id,
            label=args.label
        )

        print_summary(result)

    except Exception as e:
        print(f"\n[FATAL ERROR] Face and Mouth preprocessing failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
