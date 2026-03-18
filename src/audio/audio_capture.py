"""
src/audio/audio_capture.py
---------------------------
Real-time audio streaming simulation via chunk-based iteration.
Supports WAV file playback, numpy array injection, and (optional) live mic.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Generator, Iterator, Optional, Union

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AudioStreamer:
    """
    Simulate real-time audio streaming by yielding fixed-size chunks.

    Can source audio from:
    - A NumPy array (for testing / simulation)
    - A WAV file path
    - Live microphone (requires pyaudio; optional)

    Parameters
    ----------
    sample_rate : int
        Expected sample rate in Hz.
    chunk_duration_ms : int
        Duration of each emitted chunk in milliseconds.
    realtime : bool
        If True, sleep between chunks to simulate real-time pacing.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        chunk_duration_ms: int = 500,
        realtime: bool = False,
    ):
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1_000)
        self.realtime = realtime
        self._chunk_duration_s = chunk_duration_ms / 1_000
        logger.info(
            "AudioStreamer ready: sr=%d, chunk=%d samples (%.0f ms), realtime=%s",
            sample_rate,
            self.chunk_size,
            chunk_duration_ms,
            realtime,
        )

    # ── Generators ────────────────────────────────────────────────────────────

    def stream_array(
        self,
        audio: np.ndarray,
    ) -> Generator[np.ndarray, None, None]:
        """
        Yield fixed-size chunks from a NumPy audio array.

        Parameters
        ----------
        audio : np.ndarray
            1-D float32 array of audio samples.

        Yields
        ------
        np.ndarray
            Audio chunk of shape ``(chunk_size,)``. Last chunk is zero-padded.
        """
        audio = np.asarray(audio, dtype=np.float32)
        total_chunks = max(1, int(np.ceil(len(audio) / self.chunk_size)))
        logger.info("Streaming %d chunk(s) from array (total samples=%d)", total_chunks, len(audio))

        for i in range(total_chunks):
            start = i * self.chunk_size
            end = start + self.chunk_size
            chunk = audio[start:end]
            # zero-pad the last chunk if needed
            if len(chunk) < self.chunk_size:
                chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)), mode="constant")
            if self.realtime:
                time.sleep(self._chunk_duration_s)
            yield chunk

    def stream_file(
        self,
        wav_path: Union[str, Path],
    ) -> Generator[np.ndarray, None, None]:
        """
        Load a WAV file and stream it chunk-by-chunk.

        Parameters
        ----------
        wav_path : str | Path
            Path to a ``.wav`` file.

        Yields
        ------
        np.ndarray
        """
        try:
            import soundfile as sf
        except ImportError:
            import librosa
            audio, sr = librosa.load(str(wav_path), sr=self.sample_rate, mono=True)
            yield from self.stream_array(audio)
            return

        with sf.SoundFile(str(wav_path)) as f:
            if f.samplerate != self.sample_rate:
                logger.warning(
                    "WAV sample rate %d ≠ expected %d — chunks may be misaligned.",
                    f.samplerate,
                    self.sample_rate,
                )
            logger.info("Streaming WAV: %s (frames=%d)", wav_path, f.frames)
            while True:
                chunk = f.read(self.chunk_size, dtype="float32", always_2d=False)
                if len(chunk) == 0:
                    break
                if len(chunk) < self.chunk_size:
                    chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)), mode="constant")
                if self.realtime:
                    time.sleep(self._chunk_duration_s)
                yield chunk

    def stream_mic(self) -> Generator[np.ndarray, None, None]:  # pragma: no cover
        """
        Stream audio from the default microphone via PyAudio.
        Requires ``pip install pyaudio``.
        This is a live-only generator that yields indefinitely.
        """
        try:
            import pyaudio
        except ImportError as exc:
            raise ImportError("PyAudio is required for mic streaming: pip install pyaudio") from exc

        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )
        logger.info("Mic streaming started (Ctrl+C to stop).")
        try:
            while True:
                raw = stream.read(self.chunk_size, exception_on_overflow=False)
                chunk = np.frombuffer(raw, dtype=np.float32)
                yield chunk
        except KeyboardInterrupt:
            logger.info("Mic streaming stopped.")
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def generate_silence(duration_s: float, sample_rate: int = 16_000) -> np.ndarray:
        """Return a zero-filled silence array of *duration_s* seconds."""
        return np.zeros(int(duration_s * sample_rate), dtype=np.float32)

    @staticmethod
    def generate_tone(
        frequency_hz: float,
        duration_s: float,
        amplitude: float = 0.5,
        sample_rate: int = 16_000,
    ) -> np.ndarray:
        """Generate a simple sine-wave tone (useful for unit tests)."""
        t = np.linspace(0, duration_s, int(duration_s * sample_rate), endpoint=False)
        return (amplitude * np.sin(2 * np.pi * frequency_hz * t)).astype(np.float32)
