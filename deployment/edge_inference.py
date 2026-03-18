"""
deployment/edge_inference.py
-----------------------------
Simulates ARM CPU inference using the TFLite wake-word model via
TensorFlow Lite Interpreter (equivalent to what runs on ARM edge devices).

Usage
-----
    python deployment/edge_inference.py
    python deployment/edge_inference.py --tflite-path deployment/wake_word.tflite \\
                                        --n-samples 20
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np

from src.audio.feature_extraction import MFCCExtractor, MFCCConfig
from src.audio.preprocessor import AudioPreprocessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EdgeWakeWordInference:
    """
    Runs wake-word inference using the TFLite flatbuffer model.

    This class simulates how the model would be deployed on an ARM Cortex-A/M
    processor using TFLite's C++ delegate pipeline.

    Parameters
    ----------
    tflite_path : str | Path
        Path to the ``.tflite`` model file.
    n_mfcc : int
    max_frames : int
    sample_rate : int
    num_threads : int
        Number of threads to pass to TFLite interpreter (simulates ARM cores).
    """

    def __init__(
        self,
        tflite_path: str | Path,
        n_mfcc: int = 40,
        max_frames: int = 98,
        sample_rate: int = 16_000,
        num_threads: int = 4,
    ):
        self.tflite_path = str(tflite_path)
        self.n_mfcc = n_mfcc
        self.max_frames = max_frames
        self.sample_rate = sample_rate
        self.num_threads = num_threads

        mfcc_cfg = MFCCConfig(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            max_frames=max_frames,
            delta_order=0,
        )
        self.extractor = MFCCExtractor(mfcc_cfg)
        self.preprocessor = AudioPreprocessor()
        self.interpreter = self._load_interpreter()

    # ── Interpreter ───────────────────────────────────────────────────────────

    def _load_interpreter(self):
        try:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(
                model_path=self.tflite_path,
                num_threads=self.num_threads,
            )
            interpreter.allocate_tensors()
            self._input_details = interpreter.get_input_details()
            self._output_details = interpreter.get_output_details()
            logger.info(
                "TFLite interpreter loaded: %s | threads=%d",
                self.tflite_path, self.num_threads,
            )
            logger.debug("Input details: %s", self._input_details)
            logger.debug("Output details: %s", self._output_details)
            return interpreter
        except ImportError:
            logger.error("tensorflow not installed. Install: pip install tensorflow")
            raise
        except Exception as exc:
            logger.error("Failed to load TFLite model: %s", exc)
            raise

    # ── Inference ─────────────────────────────────────────────────────────────

    def infer(self, audio_chunk: np.ndarray) -> dict:
        """
        Run a single inference pass.

        Parameters
        ----------
        audio_chunk : np.ndarray
            Raw 1-D float32 audio.

        Returns
        -------
        dict with keys: prob, detected, elapsed_ms
        """
        audio = self.preprocessor.process(audio_chunk)
        mfcc = self.extractor.extract(audio)              # (C, T)
        input_tensor = mfcc[np.newaxis, :, :].astype(np.float32)  # (1, C, T)

        t0 = time.perf_counter()
        self.interpreter.set_tensor(self._input_details[0]["index"], input_tensor)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self._output_details[0]["index"])
        elapsed_ms = (time.perf_counter() - t0) * 1_000

        prob = float(output.flat[0])
        return {
            "prob": round(prob, 4),
            "detected": prob >= 0.75,
            "elapsed_ms": round(elapsed_ms, 3),
        }

    def infer_batch(self, audio_list: list[np.ndarray]) -> list[dict]:
        """Run inference on a list of audio chunks (one-by-one)."""
        return [self.infer(a) for a in audio_list]


# ─────────────────────────────────────────────────────────────────────────────
# CLI Demo
# ─────────────────────────────────────────────────────────────────────────────

def run_edge_demo(
    tflite_path: str = "deployment/wake_word.tflite",
    n_samples: int = 20,
    sample_rate: int = 16_000,
) -> None:
    if not Path(tflite_path).exists():
        print(f"[ERROR] TFLite model not found at '{tflite_path}'.")
        print("Run 'python deployment/convert_to_tflite.py' first.")
        return

    engine = EdgeWakeWordInference(tflite_path=tflite_path, sample_rate=sample_rate)
    rng = np.random.default_rng(42)

    print(f"\n{'='*55}")
    print(f"  Edge Inference Demo  —  {n_samples} samples")
    print(f"{'='*55}")
    print(f"  {'#':<5} {'Prob':>8} {'Detected':>10} {'Latency (ms)':>14}")
    print(f"  {'-'*45}")

    latencies = []
    for i in range(n_samples):
        # Alternate between tones (positive) and noise (negative)
        if i % 3 == 0:
            t = np.linspace(0, 1.0, sample_rate, endpoint=False)
            audio = (0.5 * np.sin(2 * np.pi * rng.uniform(950, 1050) * t)).astype(np.float32)
        else:
            audio = rng.normal(0, 0.3, size=sample_rate).astype(np.float32)

        result = engine.infer(audio)
        latencies.append(result["elapsed_ms"])
        det_icon = "✅" if result["detected"] else "—"
        print(
            f"  {i+1:<5} {result['prob']:>8.4f} {det_icon:>10} {result['elapsed_ms']:>12.2f} ms"
        )

    latencies_arr = np.array(latencies)
    print(f"\n{'='*55}")
    print(f"  Mean latency  : {latencies_arr.mean():.2f} ms")
    print(f"  P50 latency   : {np.percentile(latencies_arr, 50):.2f} ms")
    print(f"  P90 latency   : {np.percentile(latencies_arr, 90):.2f} ms")
    print(f"  P99 latency   : {np.percentile(latencies_arr, 99):.2f} ms")
    print(f"{'='*55}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Edge TFLite Wake Word Inference")
    parser.add_argument("--tflite-path", default="deployment/wake_word.tflite")
    parser.add_argument("--n-samples", type=int, default=20)
    args = parser.parse_args()
    run_edge_demo(tflite_path=args.tflite_path, n_samples=args.n_samples)


if __name__ == "__main__":
    main()
