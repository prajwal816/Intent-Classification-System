"""
src/audio/feature_extraction.py
---------------------------------
MFCC feature extraction using librosa.
Produces (n_mfcc x T) arrays ready for CNN-GRU ingestion.
"""
from __future__ import annotations

import numpy as np
import librosa
from dataclasses import dataclass
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MFCCConfig:
    sample_rate: int = 16_000
    n_mfcc: int = 40
    n_fft: int = 512
    hop_length: int = 160          # 10 ms @ 16 kHz
    win_length: int = 400          # 25 ms @ 16 kHz
    n_mels: int = 80
    fmin: float = 20.0
    fmax: Optional[float] = 8_000.0
    normalize: bool = True
    delta_order: int = 2           # 0 = static only, 1 = +delta, 2 = +delta-delta
    max_frames: Optional[int] = 98 # pads/trims to this many frames


class MFCCExtractor:
    """
    Extract MFCC features (optionally + Δ and ΔΔ) from a raw audio waveform.

    Parameters
    ----------
    config : MFCCConfig
        Feature extraction configuration.

    Examples
    --------
    >>> extractor = MFCCExtractor()
    >>> audio = np.random.randn(16000).astype(np.float32)
    >>> features = extractor.extract(audio)
    >>> features.shape   # (n_mfcc, T) or (3*n_mfcc, T) with deltas
    (40, 98)
    """

    def __init__(self, config: Optional[MFCCConfig] = None):
        self.cfg = config or MFCCConfig()

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute MFCC features from raw audio samples.

        Parameters
        ----------
        audio : np.ndarray
            1-D float32 array of audio samples.

        Returns
        -------
        np.ndarray
            Shape ``(C, T)`` where C = n_mfcc * (1 + delta_order).
        """
        audio = self._ensure_mono(audio)
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.cfg.sample_rate,
            n_mfcc=self.cfg.n_mfcc,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length,
            n_mels=self.cfg.n_mels,
            fmin=self.cfg.fmin,
            fmax=self.cfg.fmax,
        )  # shape: (n_mfcc, T)

        features = [mfcc]
        if self.cfg.delta_order >= 1:
            features.append(librosa.feature.delta(mfcc, order=1))
        if self.cfg.delta_order >= 2:
            features.append(librosa.feature.delta(mfcc, order=2))

        output = np.concatenate(features, axis=0)  # (C, T)

        if self.cfg.max_frames is not None:
            output = self._pad_or_trim(output, self.cfg.max_frames)

        if self.cfg.normalize:
            output = self._normalize(output)

        logger.debug(
            "MFCC extracted: shape=%s, sr=%d, n_mfcc=%d",
            output.shape,
            self.cfg.sample_rate,
            self.cfg.n_mfcc,
        )
        return output.astype(np.float32)

    def extract_batch(self, audios: list[np.ndarray]) -> np.ndarray:
        """Extract features for a list of audio arrays. Returns (B, C, T)."""
        return np.stack([self.extract(a) for a in audios], axis=0)

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _ensure_mono(audio: np.ndarray) -> np.ndarray:
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        elif audio.ndim != 1:
            raise ValueError(f"Unexpected audio shape: {audio.shape}")
        return audio.astype(np.float32)

    @staticmethod
    def _pad_or_trim(features: np.ndarray, max_frames: int) -> np.ndarray:
        """Pad with zeros or trim along time axis to *max_frames*."""
        T = features.shape[1]
        if T < max_frames:
            pad_width = max_frames - T
            features = np.pad(features, ((0, 0), (0, pad_width)), mode="constant")
        elif T > max_frames:
            features = features[:, :max_frames]
        return features

    @staticmethod
    def _normalize(features: np.ndarray) -> np.ndarray:
        """Per-feature (row-wise) zero-mean unit-variance normalization."""
        mean = features.mean(axis=1, keepdims=True)
        std = features.std(axis=1, keepdims=True) + 1e-9
        return (features - mean) / std
