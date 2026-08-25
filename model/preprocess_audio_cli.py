"""
CLI Entry Point for Audio Preprocessing (16 kHz 4s Mono Waveform and Mel-Spectrogram).

Usage:
    python preprocess_audio_cli.py --input-audio output/audio/audio.wav --output-dir processed_dataset/video_001 --video-id video_001
"""

import argparse
import sys
from pathlib import Path

from preprocessing import (
    AudioPreprocessor,
    AudioPreprocessorConfig,
    AudioPreprocessorResult,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standardize extracted audio to 16 kHz Mono 4s (64,000 samples) and compute Mel-Spectrogram (.npy)."
    )
    parser.add_argument(
        "--input-audio", "-i",
        type=str,
        required=True,
        help="Path to the extracted audio file (e.g., output/audio/audio.wav)."
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="processed_dataset/video_001",
        help="Path to the target output directory (default: 'processed_dataset/video_001')."
    )
    parser.add_argument(
        "--video-id",
        type=str,
        default="video_001",
        help="Identifier for the sample/video (default: 'video_001')."
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target sampling rate in Hz (default: 16000)."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=4.0,
        help="Target duration in seconds (default: 4.0)."
    )
    parser.add_argument(
        "--n-mels",
        type=int,
        default=128,
        help="Number of Mel filter bands (default: 128)."
    )
    parser.add_argument(
        "--n-fft",
        type=int,
        default=1024,
        help="FFT window length for STFT (default: 1024)."
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=256,
        help="Hop length for STFT (default: 256)."
    )
    parser.add_argument(
        "--win-length",
        type=int,
        default=1024,
        help="Window length for STFT (default: 1024)."
    )
    return parser.parse_args()


def print_summary(result: AudioPreprocessorResult):
    """Prints a structured summary of the audio preprocessing results."""
    sep = "=" * 68
    print("\n" + sep)
    print("   AUDIO PREPROCESSING STAGE COMPLETED (16 kHz / 4 SEC / MEL)")
    print(sep)
    print(f"Video ID:               {result.video_id}")
    print(f"Input Audio:            {result.input_audio}")
    print(f"Original Stream:        {result.original_channels}ch @ {result.original_sample_rate} Hz ({result.original_duration:.2f}s)")
    print(f"Standardized Stream:    {result.processed_channels}ch (Mono) @ {result.processed_sample_rate} Hz ({result.processed_duration:.1f}s)")
    print(f"Exact Total Samples:    {result.num_samples} samples")
    print("-" * 68)
    print(" 1. SAVED OUTPUT ARTIFACTS")
    print(f"    - Standardized WAV: {result.waveform_path}")
    print(f"    - Mel-Spectrogram:  {result.spectrogram_path} (Shape: {result.mel_shape[0]}x{result.mel_shape[1]} .npy)")
    print(f"    - Metadata JSON:    {result.metadata_path}")
    print("-" * 68)
    print(" 2. MEL-SPECTROGRAM CONFIGURATION")
    print(f"    - n_mels:           {result.mel_parameters['n_mels']}")
    print(f"    - n_fft:            {result.mel_parameters['n_fft']}")
    print(f"    - hop_length:       {result.mel_parameters['hop_length']}")
    print(f"    - win_length:       {result.mel_parameters['win_length']}")
    print(sep + "\n")


def main():
    args = parse_args()
    input_path = Path(args.input_audio)

    if not input_path.is_file():
        print(f"[ERROR] Input audio file does not exist: {args.input_audio}", file=sys.stderr)
        sys.exit(1)

    config = AudioPreprocessorConfig(
        target_sample_rate=args.sample_rate,
        target_duration_seconds=args.duration,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length
    )

    try:
        preprocessor = AudioPreprocessor(config=config)
        result = preprocessor.process(
            input_audio_path=input_path,
            output_dir=args.output_dir,
            video_id=args.video_id
        )
        print_summary(result)
    except Exception as e:
        print(f"\n[FATAL ERROR] Audio preprocessing failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
