"""
Unit Test Suite for Audio 2D Spectrogram CNN, InfoNCE Sync Loss,
Stage 2 & Stage 3 Pretraining, and Adaptive Multimodal Fusion.
"""

import math
import os
import sys
import unittest
from pathlib import Path
import numpy as np
import torch

# Ensure parent directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DEFAULT_CONFIG, VisualPipelineConfig
from preprocessing.audio_windowing import AudioWindowExtractor
from models.audio_cnn import Audio2DCNN, Audio1DCNN
from models.audio_branch import AudioAuthenticityBranch
from models.mouth_encoder import MouthROIEncoder
from models.sync_branch import AudioVisualSyncBranch
from models.multimodal_fusion import AdaptiveModalityAttention
from models.multimodal_detector import MultimodalDeepfakeDetector
from training.losses import InfoNCESyncLoss, AudioVisualSyncLoss, MultimodalCompoundLoss
from training.dataset import VideoSampleItem
from training.train_audio import StandaloneAudioClassifier
from training.train_sync_pretrain import StandaloneSyncModel
from evaluation.sync_evaluation import evaluate_synchronization_offsets


class TestAudioSyncMultimodalPipeline(unittest.TestCase):
    """Unit tests verifying 2D Audio CNN, InfoNCE Sync, Staged Pretraining, and Adaptive Fusion."""

    def test_01_audio_window_extractor(self):
        """Verify 16 kHz windowing and Log-Mel calculation."""
        extractor = AudioWindowExtractor(
            sample_rate=16000,
            window_seconds=4.0,
            hop_seconds=2.0,
            n_mels=128
        )
        self.assertEqual(extractor.window_samples, 64000)
        self.assertEqual(extractor.hop_samples, 32000)

        # Test with 10-second synthetic sine wave (160,000 samples)
        t = np.linspace(0, 10.0, 160000, dtype=np.float32)
        sine_wave = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

        mel = extractor.compute_log_mel_spectrogram(sine_wave[:64000])
        self.assertEqual(mel.shape[0], 128)
        self.assertGreater(mel.shape[1], 200) # ~251 frames
        self.assertFalse(np.isnan(mel).any())

    def test_02_2d_spectrogram_audio_cnn(self):
        """Verify Custom 2D Spectrogram CNN with (B, 1, 128, T) input and temporal token output."""
        model = Audio2DCNN(in_channels=1, out_dim=768)

        # Case 1: Standard 4D input (B, 1, 128, T)
        mel_4d = torch.randn(2, 1, 128, 251)
        tokens_4d = model(mel_4d)
        self.assertEqual(tokens_4d.shape[0], 2)
        self.assertEqual(tokens_4d.shape[2], 768)
        self.assertGreater(tokens_4d.shape[1], 30) # ~62 tokens
        self.assertFalse(torch.isnan(tokens_4d).any())

        # Case 2: 3D input (B, 128, T) auto-reshaped
        mel_3d = torch.randn(2, 128, 251)
        tokens_3d = model(mel_3d)
        self.assertEqual(tokens_3d.shape, tokens_4d.shape)

        # Case 3: 2D input (128, T) unbatched
        mel_2d = torch.randn(128, 251)
        tokens_2d = model(mel_2d)
        self.assertEqual(tokens_2d.shape[-1], 768)

        # Case 4: Variable temporal length T = 150 vs T = 300
        tokens_short = model(torch.randn(1, 1, 128, 150))
        tokens_long = model(torch.randn(1, 1, 128, 300))
        self.assertLess(tokens_short.size(1), tokens_long.size(1))

    def test_03_audio_authenticity_branch(self):
        """Verify Audio Authenticity branch with 2D CNN, self-attention, and pooling."""
        branch = AudioAuthenticityBranch(in_mels=128, d_model=768, nhead=8, num_layers=2)

        # Multi-window single video: (2, 128, 251) -> (768,)
        multi_win_mel = torch.randn(2, 128, 251)
        out_single = branch(multi_win_mel)
        self.assertEqual(out_single.audio_feature.shape, torch.Size([768]))
        self.assertEqual(out_single.window_features.shape, (2, 768))

        # Batched video input: (2, 1, 128, 251) -> (2, 768)
        batch_mel = torch.randn(2, 1, 128, 251)
        out_batch = branch(batch_mel)
        self.assertEqual(out_batch.audio_feature.shape, (2, 768))
        self.assertFalse(torch.isnan(out_batch.audio_feature).any())

    def test_04_mouth_encoder(self):
        """Verify 112x112 Mouth ROI CNN encoder output."""
        mouth_enc = MouthROIEncoder(in_channels=3, embedding_dim=256)
        dummy_mouths = torch.rand(4, 3, 112, 112)

        embeddings = mouth_enc(dummy_mouths)
        self.assertEqual(embeddings.shape, (4, 256))
        self.assertFalse(torch.isnan(embeddings).any())

    def test_05_sync_branch_and_temporal_alignment(self):
        """Verify Sync branch alignment, cosine similarity, and 256-D feature."""
        sync = AudioVisualSyncBranch(audio_token_dim=768, mouth_dim=256, sync_dim=256)
        mouth_seq = torch.randn(2, 50, 256)
        audio_tokens = torch.randn(2, 62, 768)

        out = sync(mouth_seq, audio_tokens)
        self.assertEqual(out.sync_feature.shape, (2, 256))
        self.assertEqual(out.sync_score.shape, (2, 1))
        # Pointwise cosine similarity must be bounded in [-1.0, 1.0]
        self.assertTrue(((out.temporal_similarities >= -1.01) & (out.temporal_similarities <= 1.01)).all())

    def test_06_infonce_sync_loss(self):
        """Verify Temperature-Scaled InfoNCE loss with tau = 0.07."""
        infonce = InfoNCESyncLoss(temperature=0.07)

        # Batch representation contrast
        m_embs = torch.randn(4, 256)
        a_embs = m_embs + 0.1 * torch.randn(4, 256) # Aligned pairs have high similarity
        loss_aligned = infonce(mouth_embeddings=m_embs, audio_embeddings=a_embs)
        self.assertFalse(torch.isnan(loss_aligned))
        self.assertFalse(torch.isinf(loss_aligned))

        # Mismatched pairs should yield higher loss
        a_mismatched = torch.randn(4, 256)
        loss_mismatched = infonce(mouth_embeddings=m_embs, audio_embeddings=a_mismatched)
        self.assertGreater(loss_mismatched.item(), loss_aligned.item())

        # Explicit negative shifts
        pos_sim = torch.tensor([[0.95], [0.90]])
        neg_sim = torch.tensor([[0.20, 0.10, -0.10], [0.30, 0.05, -0.20]])
        loss_shifts = infonce(pos_similarities=pos_sim, neg_similarities=neg_sim)
        self.assertGreater(loss_shifts.item(), 0.0)

    def test_07_adaptive_modality_attention_and_masking(self):
        """Verify sample-specific modality weights and missing-modality masking."""
        fusion = AdaptiveModalityAttention(visual_dim=768, audio_dim=768, sync_dim=256, fusion_dim=768)

        f_v = torch.randn(2, 768)
        f_a = torch.randn(2, 768)
        f_s = torch.randn(2, 256)

        # Case 1: All modalities available
        mask_all = torch.tensor([[True, True, True], [True, True, True]])
        out_all = fusion(f_v, f_a, f_s, modality_mask=mask_all)
        self.assertEqual(out_all.fused_feature.shape, (2, 768))
        total_weights = out_all.alpha_v + out_all.alpha_a + out_all.alpha_s
        self.assertTrue(torch.allclose(total_weights, torch.ones(2, 1), atol=1e-4))

        # Case 2: Missing Audio and Sync (Visual only available)
        mask_v_only = torch.tensor([[True, False, False], [True, False, False]])
        out_v = fusion(f_v, f_a, f_s, modality_mask=mask_v_only)
        self.assertTrue(torch.allclose(out_v.alpha_v, torch.ones(2, 1), atol=1e-4))
        self.assertTrue(torch.allclose(out_v.alpha_a, torch.zeros(2, 1), atol=1e-4))
        self.assertTrue(torch.allclose(out_v.alpha_s, torch.zeros(2, 1), atol=1e-4))

    def test_08_end_to_end_multimodal_detector_and_loss(self):
        """Verify master model forward/backward pass with L_total = L_cls + λ_sync * L_sync (InfoNCE)."""
        detector = MultimodalDeepfakeDetector(
            visual_dim=768,
            audio_dim=768,
            sync_dim=256,
            fusion_dim=768,
            frame_chunk_size=16
        )

        face_batch = torch.rand(2, 16, 3, 224, 224)
        mouth_batch = torch.rand(2, 16, 3, 112, 112)
        mel_batch = torch.rand(2, 1, 128, 251)
        targets = torch.tensor([0.0, 1.0])

        out = detector(
            face_frames=face_batch,
            mouth_crops=mouth_batch,
            mel_windows=mel_batch
        )

        self.assertEqual(out.logits.shape, (2, 1))
        self.assertEqual(out.visual_feature.shape, (2, 768))
        self.assertEqual(out.audio_feature.shape, (2, 768))
        self.assertEqual(out.sync_feature.shape, (2, 256))

        criterion = MultimodalCompoundLoss(
            lambda_sync=0.5,
            sync_loss_type="infonce",
            sync_temperature=0.07
        )
        loss, l_cls, l_sync = criterion(out.logits.view(-1), targets, out.temporal_similarities)
        loss.backward()

        self.assertFalse(torch.isnan(loss))
        # Ensure gradients propagated to visual CNN, 2D audio CNN, and mouth encoder
        self.assertIsNotNone(detector.visual_branch.spatial_cnn.block1[0].weight.grad)
        self.assertIsNotNone(detector.audio_branch.audio_cnn.block1[0].weight.grad)
        self.assertIsNotNone(detector.mouth_encoder.block1[0].weight.grad)

    def test_09_staged_models_initialization(self):
        """Verify standalone Stage 2 and Stage 3 models execute correctly."""
        # Stage 2 Audio Classifier
        stage2_model = StandaloneAudioClassifier(in_mels=128, audio_dim=768)
        dummy_mel = torch.randn(2, 1, 128, 251)
        logits_s2, probs_s2, feat_s2 = stage2_model(dummy_mel)
        self.assertEqual(logits_s2.shape, (2, 1))
        self.assertEqual(feat_s2.shape, (2, 768))

        # Stage 3 Sync Model
        stage3_model = StandaloneSyncModel(audio_dim=768, mouth_dim=256, sync_dim=256)
        dummy_mouth = torch.randn(2, 20, 3, 112, 112)
        sync_f, sync_sc, sims, _ = stage3_model(dummy_mouth, dummy_mel)
        self.assertEqual(sync_f.shape, (2, 256))
        self.assertEqual(sync_sc.shape, (2, 1))

    def test_10_sync_temporal_offsets(self):
        """Verify synchronization evaluation under temporal shift offsets."""
        sync_branch = AudioVisualSyncBranch(audio_token_dim=768, mouth_dim=256, sync_dim=256)
        mouth_embs = torch.randn(1, 40, 256)
        audio_toks = torch.randn(1, 62, 768)

        report = evaluate_synchronization_offsets(
            sync_branch=sync_branch,
            mouth_embeddings=mouth_embs,
            audio_tokens=audio_toks,
            offsets_sec=[-1.0, -0.5, 0.0, 0.5, 1.0]
        )
        self.assertEqual(len(report.offset_results), 5)
        self.assertFalse(math.isnan(report.synchronized_similarity))


if __name__ == "__main__":
    unittest.main()
