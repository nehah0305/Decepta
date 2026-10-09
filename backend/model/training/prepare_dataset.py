"""
Dataset Preparation Utility for Decepta.

Helps users set up new datasets by:
1. Scanning a directory of videos and auto-generating a CSV manifest
2. Validating an existing CSV manifest
3. Analyzing dataset statistics and class distribution
4. Checking for video file accessibility

Usage:
    # Scan a directory and create a manifest
    python training/prepare_dataset.py scan --input-dir /path/to/videos --output manifest.csv

    # Validate an existing manifest
    python training/prepare_dataset.py validate --manifest path/to/manifest.csv --data-root /path/to/videos

    # Analyze dataset statistics
    python training/prepare_dataset.py stats --manifest path/to/manifest.csv
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"}


def scan_directory(
    input_dir: str,
    output_csv: str,
    label_from: str = "folder",
    real_folder: str = "real",
    fake_folder: str = "fake"
) -> None:
    """
    Scan a directory of videos and generate a CSV manifest.

    Supports two labeling strategies:
    1. "folder" — Videos in a 'real/' subfolder get label=0, 'fake/' get label=1
    2. "filename" — Videos with 'real' in filename get label=0, others get label=1

    Args:
        input_dir: Root directory containing video files.
        output_csv: Output path for the generated CSV manifest.
        label_from: How to determine labels ("folder" or "filename").
        real_folder: Folder name for real videos (used when label_from="folder").
        fake_folder: Folder name for fake videos (used when label_from="folder").
    """
    input_path = Path(input_dir).resolve()
    if not input_path.exists():
        logger.error(f"Input directory not found: {input_path}")
        return

    records = []
    video_count = 0

    for file_path in sorted(input_path.rglob("*")):
        if file_path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        video_count += 1
        rel_path = file_path.relative_to(input_path)
        vid_id = file_path.stem

        # Determine label
        if label_from == "folder":
            parts = [p.lower() for p in rel_path.parts]
            if real_folder.lower() in parts:
                label = 0
            elif fake_folder.lower() in parts:
                label = 1
            else:
                logger.warning(f"Cannot determine label for: {rel_path} (not in '{real_folder}' or '{fake_folder}' folder)")
                label = -1
        elif label_from == "filename":
            if "real" in file_path.stem.lower():
                label = 0
            else:
                label = 1
        else:
            label = -1

        records.append({
            "sample_id": vid_id,
            "video_path": str(rel_path),
            "label": label,
        })

    if not records:
        logger.error(f"No video files found in {input_path}")
        return

    df = pd.DataFrame(records)
    labeled = df[df["label"] >= 0]
    unlabeled = df[df["label"] < 0]

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"Manifest generated: {output_path}")
    logger.info(f"  Total videos scanned: {video_count}")
    logger.info(f"  Labeled: {len(labeled)} (Real: {(labeled['label'] == 0).sum()}, Fake: {(labeled['label'] == 1).sum()})")
    if len(unlabeled) > 0:
        logger.warning(f"  Unlabeled (label=-1): {len(unlabeled)} — manually assign labels in the CSV")


def validate_manifest(
    manifest_path: str,
    data_root: Optional[str] = None,
    video_path_col: str = "video_path",
    label_col: str = "label"
) -> None:
    """
    Validate a CSV manifest by checking:
    1. Required columns exist
    2. Labels are valid (0 or 1)
    3. Video files exist on disk
    4. No duplicate video_ids
    """
    mpath = Path(manifest_path).resolve()
    if not mpath.exists():
        logger.error(f"Manifest not found: {mpath}")
        return

    df = pd.read_csv(mpath)
    logger.info(f"Validating: {mpath.name} ({len(df)} rows)")
    errors = 0
    warnings = 0

    # Check columns
    if video_path_col not in df.columns:
        logger.error(f"  ❌ Missing required column: '{video_path_col}'")
        logger.info(f"     Available columns: {list(df.columns)}")
        errors += 1
    if label_col not in df.columns:
        logger.error(f"  ❌ Missing required column: '{label_col}'")
        errors += 1

    if errors > 0:
        logger.error(f"Validation failed with {errors} error(s). Fix column names and retry.")
        return

    # Check label values
    unique_labels = sorted(df[label_col].unique())
    valid_labels = {0, 1}
    invalid = set(unique_labels) - valid_labels
    if invalid:
        logger.error(f"  ❌ Invalid label values found: {invalid} (expected 0=Real, 1=Fake)")
        errors += 1
    else:
        logger.info(f"  ✅ Labels valid: {unique_labels}")

    # Class distribution
    n_real = (df[label_col] == 0).sum()
    n_fake = (df[label_col] == 1).sum()
    logger.info(f"  ✅ Class distribution: Real={n_real}, Fake={n_fake}")
    if n_real == 0 or n_fake == 0:
        logger.warning(f"  ⚠️  Only one class present — model cannot learn!")
        warnings += 1

    ratio = max(n_real, n_fake) / max(min(n_real, n_fake), 1)
    if ratio > 5:
        logger.warning(f"  ⚠️  Severe class imbalance (ratio {ratio:.1f}:1). "
                       f"WeightedRandomSampler will help but consider balancing the dataset.")
        warnings += 1

    # Check duplicates
    if "sample_id" in df.columns:
        dupes = df["sample_id"].duplicated().sum()
        if dupes > 0:
            logger.warning(f"  ⚠️  {dupes} duplicate sample_id values found")
            warnings += 1
        else:
            logger.info(f"  ✅ No duplicate sample_ids")

    # Check video file existence
    if data_root:
        root = Path(data_root).resolve()
        missing = 0
        checked = 0
        for _, row in df.iterrows():
            vpath = Path(str(row[video_path_col]))
            if not vpath.is_absolute():
                vpath = root / vpath
            checked += 1
            if not vpath.exists():
                missing += 1
                if missing <= 5:
                    logger.warning(f"  ⚠️  Missing: {vpath}")

        if missing > 5:
            logger.warning(f"  ... and {missing - 5} more missing files")

        if missing == 0:
            logger.info(f"  ✅ All {checked} video files accessible")
        else:
            logger.error(f"  ❌ {missing}/{checked} video files missing")
            errors += 1
    else:
        logger.info(f"  ℹ️  Skipping file existence check (no --data-root provided)")

    # Summary
    if errors == 0:
        logger.info(f"\n✅ Validation PASSED ({warnings} warning(s))")
    else:
        logger.error(f"\n❌ Validation FAILED ({errors} error(s), {warnings} warning(s))")


def show_stats(manifest_path: str) -> None:
    """Display detailed dataset statistics."""
    mpath = Path(manifest_path).resolve()
    if not mpath.exists():
        logger.error(f"Manifest not found: {mpath}")
        return

    df = pd.read_csv(mpath)

    print(f"\n{'=' * 60}")
    print(f"Dataset Statistics: {mpath.name}")
    print(f"{'=' * 60}")
    print(f"Total samples: {len(df)}")

    if "label" in df.columns:
        print(f"\nClass Distribution:")
        n_real = (df["label"] == 0).sum()
        n_fake = (df["label"] == 1).sum()
        print(f"  Real (0): {n_real} ({n_real/len(df)*100:.1f}%)")
        print(f"  Fake (1): {n_fake} ({n_fake/len(df)*100:.1f}%)")
        print(f"  Ratio (Fake/Real): {n_fake/max(n_real, 1):.2f}")

    if "dataset" in df.columns:
        print(f"\nDatasets:")
        for ds_name, count in df["dataset"].value_counts().items():
            print(f"  {ds_name}: {count}")

    if "manipulation" in df.columns:
        print(f"\nManipulation Types:")
        for m, count in df["manipulation"].value_counts().items():
            if pd.notna(m):
                print(f"  {m}: {count}")

    # File format stats
    if "video_path" in df.columns:
        extensions = df["video_path"].apply(lambda x: Path(str(x)).suffix.lower())
        print(f"\nFile Formats:")
        for ext, count in extensions.value_counts().items():
            print(f"  {ext}: {count}")

    if "frame_count" in df.columns and df["frame_count"].notna().any():
        print(f"\nFrame Count Statistics:")
        fc = df["frame_count"].dropna()
        print(f"  Min: {fc.min():.0f}")
        print(f"  Max: {fc.max():.0f}")
        print(f"  Mean: {fc.mean():.1f}")
        print(f"  Median: {fc.median():.1f}")

    print(f"\nColumns: {list(df.columns)}")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Decepta Dataset Preparation Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Scan command
    scan_parser = subparsers.add_parser("scan", help="Scan a video directory and create a CSV manifest")
    scan_parser.add_argument("--input-dir", "-i", required=True, help="Root directory containing videos")
    scan_parser.add_argument("--output", "-o", required=True, help="Output CSV manifest path")
    scan_parser.add_argument("--label-from", choices=["folder", "filename"], default="folder",
                             help="How to determine labels (default: folder)")
    scan_parser.add_argument("--real-folder", default="real", help="Folder name for real videos (default: 'real')")
    scan_parser.add_argument("--fake-folder", default="fake", help="Folder name for fake videos (default: 'fake')")

    # Validate command
    val_parser = subparsers.add_parser("validate", help="Validate a CSV manifest")
    val_parser.add_argument("--manifest", "-m", required=True, help="Path to CSV manifest")
    val_parser.add_argument("--data-root", "-d", default=None, help="Root directory for video files")
    val_parser.add_argument("--video-path-col", default="video_path", help="Column name for video paths")
    val_parser.add_argument("--label-col", default="label", help="Column name for labels")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show dataset statistics")
    stats_parser.add_argument("--manifest", "-m", required=True, help="Path to CSV manifest")

    args = parser.parse_args()

    if args.command == "scan":
        scan_directory(args.input_dir, args.output, args.label_from, args.real_folder, args.fake_folder)
    elif args.command == "validate":
        validate_manifest(args.manifest, args.data_root, args.video_path_col, args.label_col)
    elif args.command == "stats":
        show_stats(args.manifest)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
