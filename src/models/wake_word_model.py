"""
src/models/wake_word_model.py
------------------------------
CNN-GRU hybrid model for binary wake-word detection.

Architecture
------------
Input  : (B, C, T)  — batch × MFCC channels × time frames
Block 1: Conv1d(C→32)  + BatchNorm + ReLU + MaxPool1d(2)
Block 2: Conv1d(32→64) + BatchNorm + ReLU + MaxPool1d(2)
GRU    : Bidirectional GRU (2 layers)
Head   : Linear → Sigmoid  (probability of wake word)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConvBlock(nn.Module):
    """Conv1d → BatchNorm1d → ReLU → MaxPool1d → Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool_size: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_size),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class WakeWordCNNGRU(nn.Module):
    """
    CNN-GRU hybrid model for wake-word detection.

    Parameters
    ----------
    n_mfcc : int
        Number of MFCC coefficients (input channels).
    gru_hidden : int
        Hidden size of each GRU layer.
    gru_layers : int
        Number of stacked GRU layers.
    bidirectional : bool
        Whether to use bidirectional GRU.
    conv_channels : list[int]
        Output channel sizes for each Conv block.
    conv_kernel : int
        Kernel size for all Conv layers.
    conv_pool : int
        Pooling size for all Conv MaxPool layers.
    conv_dropout : float
        Dropout rate after each Conv block.
    gru_dropout : float
        Dropout rate between GRU layers.
    """

    def __init__(
        self,
        n_mfcc: int = 40,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        bidirectional: bool = True,
        conv_channels: list[int] = None,
        conv_kernel: int = 3,
        conv_pool: int = 2,
        conv_dropout: float = 0.25,
        gru_dropout: float = 0.30,
    ):
        super().__init__()
        if conv_channels is None:
            conv_channels = [32, 64]

        # ── CNN Feature Extractor ──────────────────────────────────────────────
        cnn_layers = []
        in_ch = n_mfcc
        for out_ch in conv_channels:
            cnn_layers.append(
                ConvBlock(in_ch, out_ch, conv_kernel, conv_pool, conv_dropout)
            )
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)
        self.cnn_out_channels = in_ch  # = conv_channels[-1]

        # ── Bidirectional GRU ─────────────────────────────────────────────────
        self.gru = nn.GRU(
            input_size=self.cnn_out_channels,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=gru_dropout if gru_layers > 1 else 0.0,
        )
        self.bidirectional = bidirectional
        gru_out_size = gru_hidden * (2 if bidirectional else 1)

        # ── Classifier Head ───────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(gru_out_size),
            nn.Dropout(p=0.30),
            nn.Linear(gru_out_size, 1),
            nn.Sigmoid(),
        )

        self._init_weights()
        logger.info(
            "WakeWordCNNGRU initialised: n_mfcc=%d, conv_channels=%s, "
            "gru_hidden=%d, gru_layers=%d, bidirectional=%s",
            n_mfcc, conv_channels, gru_hidden, gru_layers, bidirectional,
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, C, T)``  — batch × MFCC channels × time frames.

        Returns
        -------
        torch.Tensor
            Shape ``(B, 1)``  — wake-word probability per sample.
        """
        # CNN: (B, C, T) → (B, conv_channels[-1], T')
        cnn_out = self.cnn(x)

        # Transpose to (B, T', features) for GRU
        gru_in = cnn_out.permute(0, 2, 1)

        # GRU: (B, T', features) → (B, T', gru_out_size)
        gru_out, _ = self.gru(gru_in)

        # Take the last time step
        last = gru_out[:, -1, :]      # (B, gru_out_size)

        return self.head(last)         # (B, 1)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return prob scores without gradient computation."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    @classmethod
    def from_config(cls, model_cfg: dict) -> "WakeWordCNNGRU":
        """Instantiate from a config dict (loaded from model_config.yaml)."""
        return cls(
            n_mfcc=model_cfg.get("n_mfcc", 40),
            gru_hidden=model_cfg.get("gru_hidden_size", 128),
            gru_layers=model_cfg.get("gru_num_layers", 2),
            bidirectional=model_cfg.get("gru_bidirectional", True),
            conv_channels=model_cfg.get("conv_channels", [32, 64]),
            conv_kernel=model_cfg.get("conv_kernel_size", 3),
            conv_pool=model_cfg.get("pool_kernel_size", 2),
            conv_dropout=model_cfg.get("conv_dropout", 0.25),
            gru_dropout=model_cfg.get("gru_dropout", 0.30),
        )
