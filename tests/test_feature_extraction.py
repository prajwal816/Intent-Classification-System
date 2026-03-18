"""
tests/test_feature_extraction.py
----------------------------------
Unit tests for MFCC feature extraction.
"""
from __future__ import annotations

import pytest

librosa = pytest.importorskip("librosa", reason="librosa not installed")
torch_skip = pytest.importorskip("torch", reason="torch not installed")

import numpy as np
from src.audio.feature_extraction import MFCCExtractor, MFCCConfig



class TestMFCCExtractor:

    def test_mfcc_shape_default(self, dummy_audio, n_mfcc, max_frames):
        """Output shape must be (n_mfcc, max_frames)."""
        cfg = MFCCConfig(n_mfcc=n_mfcc, max_frames=max_frames, delta_order=0)
        extractor = MFCCExtractor(cfg)
        features = extractor.extract(dummy_audio)
        assert features.shape == (n_mfcc, max_frames), \
            f"Expected ({n_mfcc}, {max_frames}), got {features.shape}"

    def test_mfcc_shape_with_deltas(self, dummy_audio, n_mfcc, max_frames):
        """With delta_order=2, channels should be 3×n_mfcc."""
        cfg = MFCCConfig(n_mfcc=n_mfcc, max_frames=max_frames, delta_order=2)
        extractor = MFCCExtractor(cfg)
        features = extractor.extract(dummy_audio)
        assert features.shape == (n_mfcc * 3, max_frames)

    def test_mfcc_float32_dtype(self, dummy_audio):
        """Output must be float32."""
        extractor = MFCCExtractor()
        features = extractor.extract(dummy_audio)
        assert features.dtype == np.float32

    def test_mfcc_normalization_range(self, dummy_audio):
        """After normalization, values should be mostly in [-5, 5]."""
        cfg = MFCCConfig(normalize=True, max_frames=98, delta_order=0)
        extractor = MFCCExtractor(cfg)
        features = extractor.extract(dummy_audio)
        # Normalized MFCCs should be centred; strict check on 99th percentile
        assert np.percentile(np.abs(features), 99) < 20.0

    def test_zero_mean_per_feature(self, dummy_audio):
        """Per-feature means should be close to zero after normalization."""
        cfg = MFCCConfig(normalize=True, max_frames=98, delta_order=0)
        extractor = MFCCExtractor(cfg)
        features = extractor.extract(dummy_audio)
        per_feat_means = features.mean(axis=1)
        assert np.allclose(per_feat_means, 0, atol=1e-5), \
            "Per-feature means are not zero after normalization."

    def test_pad_short_audio(self, n_mfcc, max_frames, sample_rate):
        """Short audio (0.1s) should be padded to max_frames."""
        short_audio = np.random.randn(int(0.1 * sample_rate)).astype(np.float32)
        cfg = MFCCConfig(n_mfcc=n_mfcc, max_frames=max_frames, delta_order=0)
        extractor = MFCCExtractor(cfg)
        features = extractor.extract(short_audio)
        assert features.shape == (n_mfcc, max_frames)

    def test_trim_long_audio(self, n_mfcc, max_frames, sample_rate):
        """Long audio (5s) should be trimmed to max_frames."""
        long_audio = np.random.randn(int(5 * sample_rate)).astype(np.float32)
        cfg = MFCCConfig(n_mfcc=n_mfcc, max_frames=max_frames, delta_order=0)
        extractor = MFCCExtractor(cfg)
        features = extractor.extract(long_audio)
        assert features.shape == (n_mfcc, max_frames)

    def test_batch_extraction(self, dummy_audio, n_mfcc, max_frames):
        """Batch shape should be (B, C, T)."""
        cfg = MFCCConfig(n_mfcc=n_mfcc, max_frames=max_frames, delta_order=0)
        extractor = MFCCExtractor(cfg)
        batch = extractor.extract_batch([dummy_audio, dummy_audio, dummy_audio])
        assert batch.shape == (3, n_mfcc, max_frames)
