"""
src/inference/wake_word_detector.py
------------------------------------
Runs the CNN-GRU model on audio chunks and returns detection results with confidence.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from src.audio.feature_extraction import MFCCExtractor, MFCCConfig
from src.audio.preprocessor import AudioPreprocessor
from src.models.wake_word_model import WakeWordCNNGRU
from src.utils.logger import get_logger

logger = get_logger(__name__)


class WakeWordDetector:
    """
    Loads a trained CNN-GRU model and exposes ``detect(audio_chunk)`` for real-time use.

    Parameters
    ----------
    model_path : str | Path
        Path to saved ``.pt`` checkpoint file.
    threshold : float
        Confidence threshold above which a detection is declared.
    n_mfcc : int
        MFCC coefficient count (must match trained model).
    max_frames : int
        Time frame count (must match trained model).
    sample_rate : int
        Audio sample rate in Hz.
    device : str | None
        ``"cuda"``, ``"cpu"``, or auto-detect.
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        threshold: float = 0.75,
        n_mfcc: int = 40,
        max_frames: int = 98,
        sample_rate: int = 16_000,
        device: Optional[str] = None,
    ):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        mfcc_config = MFCCConfig(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            max_frames=max_frames,
            delta_order=0,
        )
        self.extractor = MFCCExtractor(mfcc_config)
        self.preprocessor = AudioPreprocessor()

        self.model = WakeWordCNNGRU(n_mfcc=n_mfcc)
        self.model.to(self.device)

        if model_path and Path(model_path).exists():
            state_dict = torch.load(str(model_path), map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info("WakeWordDetector loaded checkpoint from '%s'", model_path)
        else:
            logger.warning(
                "No checkpoint found at '%s'. Using random-initialised model.", model_path
            )

        self.model.eval()

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, audio_chunk: np.ndarray) -> Tuple[bool, float]:
        """
        Detect wake word in a single audio chunk.

        Parameters
        ----------
        audio_chunk : np.ndarray
            Raw 1-D float32 audio samples.

        Returns
        -------
        (detected, confidence)
            - detected   : bool  — True if confidence ≥ threshold
            - confidence : float — model score in [0, 1]
        """
        audio = self.preprocessor.process(audio_chunk)
        mfcc = self.extractor.extract(audio)                      # (C, T)
        tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, C, T)

        with torch.no_grad():
            prob = self.model(tensor).item()

        detected = prob >= self.threshold
        logger.debug("WakeWord: prob=%.4f, detected=%s", prob, detected)
        return detected, float(prob)

    def detect_stream(
        self,
        audio_stream,   # Iterable[np.ndarray]
    ):
        """
        Yield detection results over a stream of audio chunks.

        Parameters
        ----------
        audio_stream : Iterable[np.ndarray]

        Yields
        ------
        dict with keys: chunk_idx, detected, confidence
        """
        for idx, chunk in enumerate(audio_stream):
            detected, confidence = self.detect(chunk)
            result = {
                "chunk_idx": idx,
                "detected": detected,
                "confidence": round(confidence, 4),
            }
            yield result
            if detected:
                logger.info(
                    "Wake word DETECTED at chunk %d (confidence=%.4f)", idx, confidence
                )

    @classmethod
    def from_config(
        cls,
        pipeline_cfg,
        model_cfg,
    ) -> "WakeWordDetector":
        """Factory method from loaded config objects."""
        return cls(
            model_path=pipeline_cfg.wake_word.model_path,
            threshold=float(pipeline_cfg.wake_word.threshold),
            n_mfcc=int(model_cfg.wake_word_model.n_mfcc),
            max_frames=int(model_cfg.wake_word_model.max_frames),
            sample_rate=int(pipeline_cfg.pipeline.sample_rate),
        )
