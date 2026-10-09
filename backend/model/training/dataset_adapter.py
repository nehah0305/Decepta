"""
Flexible Dataset Adapter for New Datasets.

Ingests CSV manifests with configurable column names and resolves video paths
against a configurable data root. Supports:
1. Standard CSV manifests with video_path and label columns
2. Auto-splitting a single manifest into train/val/test
3. Custom column name mappings
4. Multiple dataset manifests merged together
5. Dataset statistics reporting
"""

import logging
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from training.dataset import VideoSampleItem

logger = logging.getLogger(__name__)


class DatasetAdapter:
    """
    Flexible adapter that reads CSV manifests and produces VideoSampleItem lists
    ready for training.

    Expected CSV columns (names configurable via column_map):
        - video_path: Relative or absolute path to the video file
        - label: Integer label (0 = Real, 1 = Fake)
        - sample_id: (Optional) Unique video identifier

    Example CSV:
        sample_id,video_path,label
        vid_001,real/video_001.mp4,0
        vid_002,fake/video_002.mp4,1
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        column_map: Optional[Dict[str, str]] = None
    ):
        """
        Args:
            data_root: Root directory where video files are stored.
                       video_path in CSV is resolved relative to this.
            column_map: Mapping of canonical names to actual CSV column names.
                        Defaults: {"video_path": "video_path", "label": "label",
                                   "sample_id": "sample_id"}
        """
        self.data_root = Path(data_root).resolve()
        self.column_map = {
            "video_path": "video_path",
            "label": "label",
            "sample_id": "sample_id",
        }
        if column_map:
            self.column_map.update(column_map)

    def load_manifest(
        self,
        manifest_path: Union[str, Path],
        split: str = "train",
        validate_paths: bool = False
    ) -> List[VideoSampleItem]:
        """
        Load a CSV manifest and return a list of VideoSampleItem objects.

        Args:
            manifest_path: Path to the CSV file.
            split: Split label to assign ("train", "val", "test").
            validate_paths: If True, check that each video file exists and
                            skip missing files with a warning.

        Returns:
            List of VideoSampleItem objects.
        """
        manifest_path = Path(manifest_path).resolve()
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        df = pd.read_csv(manifest_path)
        logger.info(f"Loaded manifest: {manifest_path.name} ({len(df)} rows)")

        # Validate required columns exist
        vpath_col = self.column_map["video_path"]
        label_col = self.column_map["label"]
        sample_id_col = self.column_map.get("sample_id", "sample_id")

        if vpath_col not in df.columns:
            raise ValueError(
                f"Column '{vpath_col}' not found in {manifest_path.name}. "
                f"Available columns: {list(df.columns)}. "
                f"Update dataset.columns.video_path in your config."
            )
        if label_col not in df.columns:
            raise ValueError(
                f"Column '{label_col}' not found in {manifest_path.name}. "
                f"Available columns: {list(df.columns)}. "
                f"Update dataset.columns.label in your config."
            )

        has_sample_id = sample_id_col in df.columns
        items: List[VideoSampleItem] = []
        skipped = 0

        for _, row in df.iterrows():
            rel_path = str(row[vpath_col])
            label = int(row[label_col])

            # Resolve video path: try absolute first, then relative to data_root
            video_path = Path(rel_path)
            if not video_path.is_absolute():
                video_path = self.data_root / rel_path

            # Generate a sample_id
            if has_sample_id and pd.notna(row.get(sample_id_col)):
                vid_id = str(row[sample_id_col])
            else:
                vid_id = video_path.stem

            # Optionally validate path existence
            if validate_paths and not video_path.exists():
                skipped += 1
                continue

            items.append(VideoSampleItem(
                video_id=vid_id,
                video_path=str(video_path),
                label=label,
                split=split
            ))

        if skipped > 0:
            logger.warning(f"Skipped {skipped}/{len(df)} videos with missing files.")

        self._log_statistics(items, split, manifest_path.name)
        return items

    def auto_split_manifest(
        self,
        manifest_path: Union[str, Path],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        validate_paths: bool = False
    ) -> Tuple[List[VideoSampleItem], List[VideoSampleItem], List[VideoSampleItem]]:
        """
        Load a single manifest and automatically split into train/val/test
        with stratified sampling to preserve class balance.

        Args:
            manifest_path: Path to the single CSV file.
            train_ratio: Fraction for training.
            val_ratio: Fraction for validation.
            test_ratio: Fraction for testing.
            seed: Random seed for reproducibility.
            validate_paths: If True, skip videos with missing files.

        Returns:
            (train_items, val_items, test_items)
        """
        all_items = self.load_manifest(manifest_path, split="all", validate_paths=validate_paths)

        # Stratified split: separate by class, then split each class
        real_items = [i for i in all_items if i.label == 0]
        fake_items = [i for i in all_items if i.label == 1]

        rng = random.Random(seed)
        rng.shuffle(real_items)
        rng.shuffle(fake_items)

        def split_list(items: list) -> Tuple[list, list, list]:
            n = len(items)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            train = items[:n_train]
            val = items[n_train:n_train + n_val]
            test = items[n_train + n_val:]
            return train, val, test

        r_train, r_val, r_test = split_list(real_items)
        f_train, f_val, f_test = split_list(fake_items)

        # Assign correct split labels
        for item in r_train + f_train:
            item.split = "train"
        for item in r_val + f_val:
            item.split = "val"
        for item in r_test + f_test:
            item.split = "test"

        train_items = r_train + f_train
        val_items = r_val + f_val
        test_items = r_test + f_test

        rng.shuffle(train_items)
        rng.shuffle(val_items)
        rng.shuffle(test_items)

        logger.info(f"Auto-split: Train={len(train_items)}, Val={len(val_items)}, Test={len(test_items)}")
        self._log_statistics(train_items, "train", "auto-split")
        self._log_statistics(val_items, "val", "auto-split")
        self._log_statistics(test_items, "test", "auto-split")

        return train_items, val_items, test_items

    def merge_manifests(
        self,
        manifest_paths: List[Union[str, Path]],
        data_roots: Optional[List[Union[str, Path]]] = None,
        split: str = "train",
        validate_paths: bool = False
    ) -> List[VideoSampleItem]:
        """
        Merge multiple CSV manifests into a single item list.
        Useful for combining datasets (e.g., FF++ + Celeb-DF).

        Args:
            manifest_paths: List of CSV manifest file paths.
            data_roots: List of data root dirs for each manifest.
                        If None, uses self.data_root for all.
            split: Split label for all items.
            validate_paths: Skip missing video files.

        Returns:
            Combined list of VideoSampleItem objects.
        """
        all_items: List[VideoSampleItem] = []
        if data_roots is None:
            data_roots = [self.data_root] * len(manifest_paths)

        for mpath, droot in zip(manifest_paths, data_roots):
            adapter = DatasetAdapter(data_root=droot, column_map=self.column_map)
            items = adapter.load_manifest(mpath, split=split, validate_paths=validate_paths)
            all_items.extend(items)

        logger.info(f"Merged {len(manifest_paths)} manifests: {len(all_items)} total samples")
        self._log_statistics(all_items, split, "merged")
        return all_items

    @staticmethod
    def _log_statistics(items: List[VideoSampleItem], split: str, source: str) -> None:
        """Log dataset statistics."""
        if not items:
            logger.warning(f"[{split}] {source}: 0 samples loaded!")
            return

        labels = [i.label for i in items]
        n_real = labels.count(0)
        n_fake = labels.count(1)
        total = len(labels)
        ratio = n_fake / n_real if n_real > 0 else float('inf')

        logger.info(
            f"[{split.upper()}] {source}: {total} videos | "
            f"Real: {n_real} ({n_real/total*100:.1f}%) | "
            f"Fake: {n_fake} ({n_fake/total*100:.1f}%) | "
            f"Ratio (Fake/Real): {ratio:.2f}"
        )
