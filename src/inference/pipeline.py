"""
src/inference/pipeline.py
--------------------------
End-to-end voice pipeline: Audio → Wake Word → STT → Intent Classification.
Tracks per-stage and total latency, and provides a --demo CLI mode.

Usage
-----
    python src/inference/pipeline.py --demo
    python src/inference/pipeline.py --audio path/to/file.wav
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np

from src.audio.audio_capture import AudioStreamer
from src.inference.wake_word_detector import WakeWordDetector
from src.inference.stt_engine import STTEngine
from src.inference.intent_classifier import IntentClassifier
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.metrics import LatencyReport, LatencyTracker

logger = get_logger(__name__)


class VoicePipeline:
    """
    Orchestrates the three-stage voice processing pipeline.

    Stages
    ------
    1. Wake Word Detection  (CNN-GRU)
    2. Speech-to-Text       (Whisper Tiny)
    3. Intent Classification (BERT)

    Parameters
    ----------
    wake_word_detector : WakeWordDetector
    stt_engine         : STTEngine
    intent_classifier  : IntentClassifier
    latency_budget_ms  : float
        Log a warning if total latency exceeds this.
    """

    def __init__(
        self,
        wake_word_detector: WakeWordDetector,
        stt_engine: STTEngine,
        intent_classifier: IntentClassifier,
        latency_budget_ms: float = 100.0,
    ):
        self.detector = wake_word_detector
        self.stt = stt_engine
        self.classifier = intent_classifier
        self.latency_budget_ms = latency_budget_ms

    # ── Core pipeline ─────────────────────────────────────────────────────────

    def process_chunk(
        self,
        audio_chunk: np.ndarray,
        top_k: int = 3,
    ) -> dict:
        """
        Run the full pipeline on a single audio chunk.

        Parameters
        ----------
        audio_chunk : np.ndarray
            1-D float32 audio array.
        top_k : int
            Number of intent candidates to return.

        Returns
        -------
        dict with keys:
            wake_detected, wake_confidence,
            transcription, language, stt_elapsed_ms,
            intent, intent_confidence, top_k_intents,
            low_confidence, intent_elapsed_ms,
            total_ms, within_budget
        """
        report = LatencyReport()
        result: dict = {}

        # ── Stage 1: Wake Word ────────────────────────────────────────────────
        with LatencyTracker("wake_word", report) as ww_t:
            detected, ww_confidence = self.detector.detect(audio_chunk)

        result["wake_detected"] = detected
        result["wake_confidence"] = round(ww_confidence, 4)
        result["wake_elapsed_ms"] = round(ww_t.elapsed_ms, 2)

        if not detected:
            result["total_ms"] = round(report.total_ms, 2)
            result["within_budget"] = result["total_ms"] <= self.latency_budget_ms
            return result

        # ── Stage 2: STT ──────────────────────────────────────────────────────
        stt_result = self.stt.transcribe(audio_chunk, report=report)

        result["transcription"] = stt_result.get("text", "")
        result["language"] = stt_result.get("language", "en")
        result["stt_elapsed_ms"] = stt_result.get("elapsed_ms", 0.0)

        # ── Stage 3: Intent ───────────────────────────────────────────────────
        text = result["transcription"]
        if text:
            intent_result = self.classifier.classify(text, top_k=top_k, report=report)
            result["intent"] = intent_result.get("intent", "unknown")
            result["intent_confidence"] = intent_result.get("confidence", 0.0)
            result["top_k_intents"] = intent_result.get("top_k", [])
            result["low_confidence"] = intent_result.get("low_confidence", False)
            result["intent_elapsed_ms"] = intent_result.get("elapsed_ms", 0.0)
        else:
            result["intent"] = "unknown"
            result["intent_confidence"] = 0.0
            result["top_k_intents"] = []
            result["low_confidence"] = True

        result["total_ms"] = round(report.total_ms, 2)
        result["within_budget"] = result["total_ms"] <= self.latency_budget_ms
        result["latency_report"] = report.to_dict()

        if not result["within_budget"]:
            logger.warning(
                "Latency budget exceeded: %.1f ms > %.1f ms",
                result["total_ms"],
                self.latency_budget_ms,
            )

        return result

    def process_stream(self, audio_stream, top_k: int = 3):
        """
        Process a stream of audio chunks, yielding results only when the
        wake word is detected.

        Yields
        ------
        dict — same as ``process_chunk()`` output.
        """
        for idx, chunk in enumerate(audio_stream):
            result = self.process_chunk(chunk, top_k=top_k)
            result["chunk_idx"] = idx
            if result.get("wake_detected"):
                yield result

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_configs(
        cls,
        pipeline_config_path: str = "configs/pipeline_config.yaml",
        model_config_path: str = "configs/model_config.yaml",
    ) -> "VoicePipeline":
        pipeline_cfg = load_config(pipeline_config_path)
        model_cfg = load_config(model_config_path)

        detector = WakeWordDetector.from_config(pipeline_cfg, model_cfg)
        stt = STTEngine.from_config(pipeline_cfg)
        classifier = IntentClassifier.from_config(pipeline_cfg)

        return cls(
            wake_word_detector=detector,
            stt_engine=stt,
            intent_classifier=classifier,
            latency_budget_ms=float(pipeline_cfg.pipeline.latency_budget_ms),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Demo / CLI
# ─────────────────────────────────────────────────────────────────────────────

def _print_result(result: dict) -> None:
    from rich.console import Console
    from rich.table import Table
    console = Console()

    wake = "✅ DETECTED" if result.get("wake_detected") else "❌ Not detected"
    console.print(f"\n[bold cyan]Wake Word:[/bold cyan] {wake} "
                  f"(confidence={result.get('wake_confidence', 0):.4f}, "
                  f"latency={result.get('wake_elapsed_ms', 0):.1f} ms)")

    if result.get("wake_detected"):
        console.print(f"[bold cyan]Transcription:[/bold cyan] \"{result.get('transcription', '')}\" "
                      f"({result.get('stt_elapsed_ms', 0):.1f} ms)")
        console.print(f"[bold cyan]Intent:[/bold cyan] {result.get('intent', 'unknown')} "
                      f"(confidence={result.get('intent_confidence', 0):.4f}, "
                      f"latency={result.get('intent_elapsed_ms', 0):.1f} ms)")

        table = Table(title="Top-K Intents", show_header=True)
        table.add_column("Rank", style="dim", width=6)
        table.add_column("Intent")
        table.add_column("Confidence")
        for i, item in enumerate(result.get("top_k_intents", []), 1):
            table.add_row(str(i), item["intent"], f"{item['confidence']:.4f}")
        console.print(table)

    budget_icon = "✅" if result.get("within_budget") else "⚠️ "
    console.print(
        f"[bold green]Total latency:[/bold green] "
        f"{result.get('total_ms', 0):.1f} ms [{budget_icon} budget]"
    )


def run_demo(
    pipeline_config: str = "configs/pipeline_config.yaml",
    model_config: str = "configs/model_config.yaml",
    audio_path: Optional[str] = None,
) -> None:
    """Run end-to-end demo with simulated or file audio."""
    logger.info("Initialising VoicePipeline...")
    pipeline = VoicePipeline.from_configs(pipeline_config, model_config)

    streamer = AudioStreamer(sample_rate=16_000, chunk_duration_ms=1000)

    if audio_path and Path(audio_path).exists():
        logger.info("Streaming from file: %s", audio_path)
        chunks = list(streamer.stream_file(audio_path))
    else:
        logger.info("No audio file specified — using synthetic demo signals.")
        # Simulate 3 seconds: 1 s silence + 1 s tone + 1 s noise
        silence = AudioStreamer.generate_silence(1.0)
        tone    = AudioStreamer.generate_tone(1000.0, 1.0)   # wake-word sim
        noise   = np.random.randn(16_000).astype(np.float32) * 0.3
        demo_audio = np.concatenate([silence, tone, noise])
        chunks = list(streamer.stream_array(demo_audio))

    try:
        from rich.console import Console
        Console().rule("[bold]Voice Pipeline Demo")
    except ImportError:
        print("=" * 50 + "\nVoice Pipeline Demo\n" + "=" * 50)

    for idx, chunk in enumerate(chunks):
        result = pipeline.process_chunk(chunk)
        result["chunk_idx"] = idx
        try:
            _print_result(result)
        except ImportError:
            print(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice Pipeline End-to-End Demo")
    parser.add_argument("--pipeline-config", default="configs/pipeline_config.yaml")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--audio", default=None, help="Path to a WAV file")
    parser.add_argument("--demo", action="store_true", help="Run synthetic demo")
    args = parser.parse_args()

    run_demo(
        pipeline_config=args.pipeline_config,
        model_config=args.model_config,
        audio_path=args.audio,
    )


if __name__ == "__main__":
    main()
