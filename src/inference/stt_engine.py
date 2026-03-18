"""
src/inference/stt_engine.py
----------------------------
Lightweight Speech-to-Text wrapper around OpenAI Whisper (tiny model).
Adds streaming simulation, latency logging, and a fallback stub for testing.
"""
from __future__ import annotations

import time
from typing import Optional, Union

import numpy as np

from src.utils.logger import get_logger
from src.utils.metrics import LatencyReport, LatencyTracker

logger = get_logger(__name__)


class STTEngine:
    """
    Speech-to-Text engine using OpenAI Whisper (tiny by default).

    Falls back to a keyword-based stub if Whisper is unavailable —
    useful for unit tests without GPU/heavy dependencies.

    Parameters
    ----------
    model_name : str
        Whisper model size: ``"tiny"`` | ``"base"`` | ``"small"``.
    language : str
        Target language code (e.g., ``"en"``).
    fp16 : bool
        Use half precision (GPU only).
    beam_size : int
        Beam search width (1 = greedy, fastest).
    simulate_latency_ms : float
        Add synthetic delay to simulate edge-device RTF (0 = off).
    """

    def __init__(
        self,
        model_name: str = "tiny",
        language: str = "en",
        fp16: bool = False,
        beam_size: int = 1,
        simulate_latency_ms: float = 0.0,
    ):
        self.model_name = model_name
        self.language = language
        self.fp16 = fp16
        self.beam_size = beam_size
        self.simulate_latency_ms = simulate_latency_ms
        self._model = None
        self._stub_mode = False

        self._load_model()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        try:
            import whisper  # type: ignore
            logger.info("Loading Whisper model: '%s'", self.model_name)
            self._model = whisper.load_model(self.model_name)
            logger.info("Whisper '%s' loaded successfully.", self.model_name)
        except ImportError:
            logger.warning(
                "openai-whisper not installed. STTEngine running in stub mode. "
                "Install with: pip install openai-whisper"
            )
            self._stub_mode = True

    # ── Public API ────────────────────────────────────────────────────────────

    def transcribe(
        self,
        audio: Union[np.ndarray, str],
        report: Optional[LatencyReport] = None,
    ) -> dict:
        """
        Transcribe audio to text.

        Parameters
        ----------
        audio : np.ndarray | str
            Float32 audio array at 16 kHz mono, or a path to a WAV file.
        report : LatencyReport | None
            If provided, STT latency is recorded.

        Returns
        -------
        dict with keys:
            - ``text``           : str — transcription
            - ``language``       : str
            - ``elapsed_ms``     : float — STT wall-clock time
            - ``segments``       : list (Whisper segments, empty in stub mode)
        """
        with LatencyTracker("stt", report) as tracker:
            if self._stub_mode:
                result = self._stub_transcribe(audio)
            else:
                result = self._whisper_transcribe(audio)

            if self.simulate_latency_ms > 0:
                time.sleep(self.simulate_latency_ms / 1_000)

        result["elapsed_ms"] = round(tracker.elapsed_ms, 3)
        logger.info(
            "STT: '%s' (%.1f ms)", result["text"][:80], result["elapsed_ms"]
        )
        return result

    def transcribe_stream(
        self,
        audio_stream,  # Iterable[np.ndarray]
        chunk_sr: int = 16_000,
    ):
        """
        Yield transcriptions for each audio chunk in a stream.
        Chunks are concatenated before transcription (sliding window mode).

        Yields
        ------
        dict — same as ``transcribe()`` output
        """
        buffer = np.array([], dtype=np.float32)
        for chunk in audio_stream:
            buffer = np.concatenate([buffer, chunk])
            # Transcribe every ~2 seconds of audio
            if len(buffer) >= chunk_sr * 2:
                result = self.transcribe(buffer[:chunk_sr * 2])
                yield result
                buffer = buffer[chunk_sr * 2:]  # slide

        if len(buffer) > 0:
            result = self.transcribe(buffer)
            yield result

    # ── Internal ──────────────────────────────────────────────────────────────

    def _whisper_transcribe(self, audio: Union[np.ndarray, str]) -> dict:
        import whisper  # type: ignore

        if isinstance(audio, np.ndarray):
            # Convert to float32 if needed
            audio = audio.astype(np.float32)
            # Whisper pad_or_trim to 30 s
            audio = whisper.pad_or_trim(audio)

        result = self._model.transcribe(
            audio,
            language=self.language,
            fp16=self.fp16,
            beam_size=self.beam_size,
        )
        return {
            "text": result.get("text", "").strip(),
            "language": result.get("language", self.language),
            "segments": result.get("segments", []),
            "elapsed_ms": 0.0,   # will be filled by caller
        }

    @staticmethod
    def _stub_transcribe(audio: Union[np.ndarray, str]) -> dict:
        """Return a deterministic stub transcription based on audio energy."""
        stubs = [
            "set an alarm for seven AM",
            "what is the weather like today",
            "play some jazz music",
            "send a message to Alice",
            "navigate to the airport",
            "turn on the living room lights",
            "tell me a joke",
            "convert 100 miles to kilometres",
        ]
        if isinstance(audio, np.ndarray):
            # Use energy as a hash to pick a stable stub
            idx = int(np.abs(audio).sum() * 1000) % len(stubs)
        else:
            idx = hash(str(audio)) % len(stubs)
        return {
            "text": stubs[idx],
            "language": "en",
            "segments": [],
            "elapsed_ms": 0.0,
        }

    @classmethod
    def from_config(cls, pipeline_cfg) -> "STTEngine":
        """Factory from pipeline config."""
        stt = pipeline_cfg.stt
        return cls(
            model_name=str(stt.model_name),
            language=str(stt.language),
            fp16=bool(stt.fp16),
            beam_size=int(stt.beam_size),
        )
