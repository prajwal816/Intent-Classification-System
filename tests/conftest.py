"""
tests/conftest.py
------------------
Shared pytest fixtures for the test suite.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture
def sample_rate() -> int:
    return 16_000


@pytest.fixture
def dummy_audio(sample_rate) -> np.ndarray:
    """1-second 1 kHz sine wave."""
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)


@pytest.fixture
def dummy_noise(sample_rate) -> np.ndarray:
    """1-second white noise."""
    rng = np.random.default_rng(0)
    return rng.normal(0, 0.3, size=sample_rate).astype(np.float32)


@pytest.fixture
def n_mfcc() -> int:
    return 40


@pytest.fixture
def max_frames() -> int:
    return 98


@pytest.fixture
def dummy_mfcc_tensor(n_mfcc, max_frames) -> torch.Tensor:
    """Random MFCC tensor: (1, n_mfcc, max_frames)."""
    return torch.randn(1, n_mfcc, max_frames)


@pytest.fixture
def dummy_config() -> dict:
    return {
        "pipeline": {
            "sample_rate": 16000,
            "chunk_size": 8000,
            "latency_budget_ms": 100,
        },
        "wake_word": {
            "threshold": 0.75,
            "model_path": "data/models/wake_word/cnn_gru_wake_word.pt",
        },
        "stt": {
            "model_name": "tiny",
            "language": "en",
            "fp16": False,
            "beam_size": 1,
        },
        "intent": {
            "model_path": "data/models/intent/",
            "num_labels": 30,
            "max_length": 128,
            "confidence_threshold": 0.5,
        },
    }
