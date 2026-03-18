"""
src/audio/preprocessor.py
--------------------------
Audio preprocessing: pre-emphasis, energy-based VAD, and normalization.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PreprocessorConfig:
    sample_rate: int = 16_000
    target_duration_s: Optional[float] = 1.0   # None = no fixed length
    pre_emphasis_coef: float = 0.97
    vad_energy_threshold: float = 0.01         # fraction of max energy
    normalize_amplitude: bool = True
    clip_amplitude: float = 0.99               # hard-clip to avoid clipping artefacts


class AudioPreprocessor:
    """
    Apply pre-processing to raw audio waveforms before feature extraction.

    Pipeline
    --------
    1. Convert to float32 and normalise amplitude to [-1, 1].
    2. Apply high-pass pre-emphasis filter.
    3. Pad or trim to ``target_duration_s``.
    4. (Optional) simple energy-based voice activity detection.

    Parameters
    ----------
    config : PreprocessorConfig
    """

    def __init__(self, config: Optional[PreprocessorConfig] = None):
        self.cfg = config or PreprocessorConfig()

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Full preprocessing pipeline.

        Parameters
        ----------
        audio : np.ndarray
            1-D integer or float audio array.

        Returns
        -------
        np.ndarray
            Float32 preprocessed audio of shape ``(N,)``.
        """
        audio = self._to_float32(audio)
        if self.cfg.normalize_amplitude:
            audio = self._normalize(audio)
        audio = self._pre_emphasis(audio, self.cfg.pre_emphasis_coef)
        if self.cfg.target_duration_s is not None:
            target_len = int(self.cfg.target_duration_s * self.cfg.sample_rate)
            audio = self._pad_or_trim(audio, target_len)
        audio = np.clip(audio, -self.cfg.clip_amplitude, self.cfg.clip_amplitude)
        logger.debug("Preprocessed audio: length=%d, sr=%d", len(audio), self.cfg.sample_rate)
        return audio

    def is_speech(self, audio: np.ndarray, frame_ms: int = 20) -> bool:
        """
        Simple energy-based voice activity detection.

        Returns ``True`` if any frame exceeds the energy threshold.
        """
        frame_len = int(frame_ms * self.cfg.sample_rate / 1_000)
        if len(audio) < frame_len:
            return False
        frames = np.array_split(audio, max(1, len(audio) // frame_len))
        energies = [np.mean(f ** 2) for f in frames]
        max_energy = max(energies)
        has_speech = any(e > self.cfg.vad_energy_threshold * max_energy for e in energies)
        logger.debug("VAD: max_energy=%.4f, speech_detected=%s", max_energy, has_speech)
        return has_speech

    def resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
    ) -> np.ndarray:
        """
        Resample *audio* from *orig_sr* to ``self.cfg.sample_rate``.
        Requires scipy.
        """
        if orig_sr == self.cfg.sample_rate:
            return audio
        try:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(self.cfg.sample_rate, orig_sr)
            up = self.cfg.sample_rate // g
            down = orig_sr // g
            resampled = resample_poly(audio, up, down)
        except ImportError:
            import librosa
            resampled = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.cfg.sample_rate)
        return resampled.astype(np.float32)

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _to_float32(audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio)
        if np.issubdtype(audio.dtype, np.integer):
            audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
        return audio.astype(np.float32)

    @staticmethod
    def _normalize(audio: np.ndarray) -> np.ndarray:
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak
        return audio

    @staticmethod
    def _pre_emphasis(audio: np.ndarray, coef: float) -> np.ndarray:
        """y[n] = x[n] - coef * x[n-1]"""
        return np.append(audio[0], audio[1:] - coef * audio[:-1])

    @staticmethod
    def _pad_or_trim(audio: np.ndarray, target_len: int) -> np.ndarray:
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)), mode="constant")
        elif len(audio) > target_len:
            audio = audio[:target_len]
        return audio
