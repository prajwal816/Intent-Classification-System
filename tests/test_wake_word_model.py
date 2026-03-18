"""
tests/test_wake_word_model.py
------------------------------
Unit tests for the CNN-GRU wake word model.
"""
from __future__ import annotations

import torch
import pytest

from src.models.wake_word_model import WakeWordCNNGRU, ConvBlock


class TestConvBlock:

    def test_conv_block_output_shape(self):
        """ConvBlock should halve T via MaxPool."""
        block = ConvBlock(in_channels=40, out_channels=32, kernel_size=3, pool_size=2)
        x = torch.randn(4, 40, 98)     # (B, C, T)
        out = block(x)
        assert out.shape == (4, 32, 49), f"Got {out.shape}"

    def test_conv_block_grad_flows(self):
        """Gradients should flow through ConvBlock."""
        block = ConvBlock(40, 32)
        x = torch.randn(2, 40, 50, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None


class TestWakeWordCNNGRU:

    def test_forward_shape(self, dummy_mfcc_tensor):
        """Output shape must be (B, 1)."""
        model = WakeWordCNNGRU(n_mfcc=40)
        out = model(dummy_mfcc_tensor)
        assert out.shape == (1, 1), f"Expected (1, 1), got {out.shape}"

    def test_output_range(self, dummy_mfcc_tensor):
        """Sigmoid output must be in [0, 1]."""
        model = WakeWordCNNGRU(n_mfcc=40)
        out = model(dummy_mfcc_tensor)
        assert (out >= 0).all() and (out <= 1).all(), "Output outside [0, 1]"

    def test_batch_forward(self, n_mfcc, max_frames):
        """Should handle batch of size 16."""
        model = WakeWordCNNGRU(n_mfcc=n_mfcc)
        x = torch.randn(16, n_mfcc, max_frames)
        out = model(x)
        assert out.shape == (16, 1)

    def test_predict_proba_no_grad(self, dummy_mfcc_tensor):
        """predict_proba should return tensor without grad."""
        model = WakeWordCNNGRU(n_mfcc=40)
        prob = model.predict_proba(dummy_mfcc_tensor)
        assert prob.requires_grad is False

    def test_parameter_count(self):
        """Parameter count should be > 0."""
        model = WakeWordCNNGRU(n_mfcc=40)
        assert model.count_parameters() > 0

    def test_from_config_factory(self):
        """from_config should build model with correct settings."""
        cfg = {
            "n_mfcc": 40,
            "gru_hidden_size": 64,
            "gru_num_layers": 1,
            "gru_bidirectional": False,
            "conv_channels": [16, 32],
            "conv_kernel_size": 3,
            "pool_kernel_size": 2,
            "conv_dropout": 0.1,
            "gru_dropout": 0.1,
            "max_frames": 98,
        }
        model = WakeWordCNNGRU.from_config(cfg)
        x = torch.randn(2, 40, 98)
        out = model(x)
        assert out.shape == (2, 1)

    def test_gradient_flow(self):
        """Loss backward should update model parameters."""
        model = WakeWordCNNGRU(n_mfcc=40)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.randn(4, 40, 98)
        y = torch.randint(0, 2, (4, 1)).float()

        preds = model(x)
        loss = torch.nn.BCELoss()(preds, y)
        loss.backward()

        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0
