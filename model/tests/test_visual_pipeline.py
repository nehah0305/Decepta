"""
Comprehensive Unit Test Suite for Visual Deepfake Detection Pipeline.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import torch

# Ensure parent directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import VisualPipelineConfig
from preprocessing.frame_sampler import HighCoverageFrameSampler
from preprocessing.frame_quality import FrameQualityFilter
from models.spatial_cnn import SpatialCNN
from models.fft_module import FFT2DModule
from models.frequency_cnn import FrequencyCNN
from models.gated_fusion import GatedFusion
from models.temporal_transformer import TemporalTransformer
from models.visual_model import VisualDeepfakeDetector
from training.losses import DeepfakeDetectionLoss
from evaluation.metrics import calculate_deepfake_metrics


class TestVisualDeepfakePipeline(unittest.TestCase):
    """Unit tests for all components of the visual deepfake detection system."""

    def test_01_high_coverage_sampler(self):
        """Verify high coverage sampling on 300-frame and 600-frame videos."""
        # 300-frame video with 70% coverage -> 210 frames
        sampler_70 = HighCoverageFrameSampler(coverage_ratio=0.70, min_frames=32, chunk_size=32)
        plan_300 = sampler_70.create_sampling_plan(300)

        self.assertEqual(plan_300.total_usable_frames, 300)
        self.assertEqual(plan_300.num_candidate_frames, 210)
        self.assertEqual(len(plan_300.candidate_indices), 210)
        # Verify strictly increasing order
        self.assertEqual(plan_300.candidate_indices, sorted(plan_300.candidate_indices))
        self.assertEqual(len(set(plan_300.candidate_indices)), 210)

        # Verify chunking does NOT drop frames (210 frames with chunk_size 32 -> 7 chunks)
        total_chunked = sum(len(c) for c in plan_300.chunks)
        self.assertEqual(total_chunked, 210)
        self.assertEqual(len(plan_300.chunks), 7)

        # 600-frame video with 70% coverage -> 420 frames
        plan_600 = sampler_70.create_sampling_plan(600)
        self.assertEqual(plan_600.num_candidate_frames, 420)
        self.assertEqual(sum(len(c) for c in plan_600.chunks), 420)

    def test_02_frame_quality_filter(self):
        """Test lightweight quality checks."""
        filter_module = FrameQualityFilter(min_brightness=5.0, max_brightness=250.0)

        # Valid RGB frame
        valid_img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        res_valid = filter_module.evaluate_frame(valid_img, 0, 0.0)
        self.assertTrue(res_valid.is_usable)
        self.assertEqual(res_valid.quality_status, "valid")

        # Pitch black frame (extreme dark)
        black_img = np.zeros((480, 640, 3), dtype=np.uint8)
        res_black = filter_module.evaluate_frame(black_img, 1, 0.04)
        self.assertFalse(res_black.is_usable)
        self.assertEqual(res_black.quality_status, "extreme_dark")

        # Corrupted / empty input
        res_corrupt = filter_module.evaluate_frame(None, 2, 0.08)
        self.assertFalse(res_corrupt.is_usable)
        self.assertEqual(res_corrupt.quality_status, "unreadable")

    def test_03_spatial_cnn_architecture(self):
        """Verify custom Spatial CNN dimensions."""
        model = SpatialCNN(in_channels=3, feature_dim=256)
        dummy_input = torch.randn(4, 3, 224, 224)
        out = model(dummy_input)

        self.assertEqual(out.shape, (4, 256))
        self.assertFalse(torch.isnan(out).any())

    def test_04_fft_module(self):
        """Verify 2D FFT log-magnitude transformation."""
        fft_mod = FFT2DModule()
        dummy_face = torch.rand(4, 3, 224, 224)
        freq_map = fft_mod(dummy_face)

        self.assertEqual(freq_map.shape, (4, 1, 224, 224))
        self.assertFalse(torch.isnan(freq_map).any())
        self.assertTrue((freq_map >= 0.0).all())

    def test_05_frequency_cnn_architecture(self):
        """Verify custom Frequency CNN dimensions."""
        model = FrequencyCNN(in_channels=1, feature_dim=256)
        dummy_freq = torch.rand(4, 1, 224, 224)
        out = model(dummy_freq)

        self.assertEqual(out.shape, (4, 256))
        self.assertFalse(torch.isnan(out).any())

    def test_06_gated_fusion(self):
        """Verify Gated Fusion mathematics and gate tracking."""
        fusion = GatedFusion(spatial_dim=256, frequency_dim=256, hidden_dim=128)
        fs = torch.randn(4, 256)
        ff = torch.randn(4, 256)
        fused, gate = fusion(fs, ff)

        self.assertEqual(fused.shape, (4, 256))
        self.assertEqual(gate.shape, (4, 1))
        # Gate must be within [0, 1]
        self.assertTrue(((gate >= 0.0) & (gate <= 1.0)).all())

    def test_07_temporal_transformer(self):
        """Verify Temporal Transformer aggregation."""
        transformer = TemporalTransformer(in_dim=256, d_model=768, nhead=8, num_layers=2)
        # Sequence of 50 frame features
        seq_features = torch.randn(2, 50, 256)
        video_feat, attn = transformer(seq_features)

        self.assertEqual(video_feat.shape, (2, 768))
        self.assertEqual(attn.shape, (2, 50))
        # Attention weights across time must sum to ~1
        self.assertTrue(torch.allclose(attn.sum(dim=-1), torch.ones(2), atol=1e-4))

    def test_08_end_to_end_visual_detector(self):
        """Verify complete end-to-end model forward and backward pass."""
        detector = VisualDeepfakeDetector(
            spatial_dim=256,
            frequency_dim=256,
            fused_dim=256,
            transformer_dim=768,
            transformer_heads=8,
            transformer_layers=2,
            frame_chunk_size=16
        )

        # Single video: sequence of 35 frames (exceeding chunk size 16 to test chunking)
        single_video_frames = torch.rand(35, 3, 224, 224)
        out = detector(single_video_frames)

        self.assertEqual(out.logits.shape, torch.Size([1]))
        self.assertEqual(out.probability.shape, torch.Size([1]))
        self.assertEqual(out.video_feature.shape, torch.Size([768]))
        self.assertEqual(out.frame_fused_features.shape, (35, 256))
        self.assertEqual(out.spatial_features.shape, (35, 256))
        self.assertEqual(out.frequency_features.shape, (35, 256))
        self.assertEqual(out.gate_values.shape, (35, 1))

        # Batched video training pass with backpropagation
        batch_videos = torch.rand(2, 10, 3, 224, 224)
        targets = torch.tensor([0.0, 1.0])
        criterion = DeepfakeDetectionLoss()

        batch_out = detector(batch_videos)
        loss = criterion(batch_out.logits.view(-1), targets)
        loss.backward()

        self.assertFalse(torch.isnan(loss))
        # Ensure gradients propagated to Spatial and Frequency CNNs
        self.assertIsNotNone(detector.spatial_cnn.block1[0].weight.grad)
        self.assertIsNotNone(detector.frequency_cnn.block1[0].weight.grad)

    def test_09_ablation_modes(self):
        """Verify model runs across all ablation modes."""
        modes = ["spatial_only", "frequency_only", "no_gate", "frame_average", "full"]
        frames = torch.rand(8, 3, 224, 224)

        for mode in modes:
            model = VisualDeepfakeDetector(mode=mode, frame_chunk_size=4)
            out = model(frames)
            self.assertEqual(out.probability.numel(), 1)
            self.assertFalse(torch.isnan(out.probability))


if __name__ == "__main__":
    unittest.main()
