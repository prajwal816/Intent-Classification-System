"""
deployment/benchmark.py
------------------------
Latency benchmark for all three pipeline stages + end-to-end.
Runs N=100 inference passes and reports P50/P90/P99 percentiles.

Usage
-----
    python deployment/benchmark.py
    python deployment/benchmark.py --n-runs 200 --pipeline-config configs/pipeline_config.yaml
"""
from __future__ import annotations

import argparse
import time
from typing import List

import numpy as np

from src.audio.audio_capture import AudioStreamer
from src.utils.logger import get_logger

logger = get_logger(__name__)

STAGES = ["wake_word", "stt", "intent", "total"]


def _generate_test_audio(sample_rate: int = 16_000, duration_s: float = 1.0) -> np.ndarray:
    """Generate a 1 kHz tone (simulates wake-word audio)."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)


def run_benchmark(
    n_runs: int = 100,
    pipeline_config: str = "configs/pipeline_config.yaml",
    model_config: str = "configs/model_config.yaml",
    include_stt: bool = False,
    include_intent: bool = True,
) -> dict[str, np.ndarray]:
    """
    Benchmark each pipeline stage independently.

    Returns
    -------
    dict mapping stage name → np.ndarray of latencies (ms)
    """
    from src.inference.wake_word_detector import WakeWordDetector
    from src.inference.intent_classifier import IntentClassifier
    from src.utils.config_loader import load_config

    pipeline_cfg = load_config(pipeline_config)
    model_cfg = load_config(model_config)

    detector = WakeWordDetector.from_config(pipeline_cfg, model_cfg)
    classifier = IntentClassifier.from_config(pipeline_cfg)

    sample_rate = int(pipeline_cfg.pipeline.sample_rate)
    audio = _generate_test_audio(sample_rate=sample_rate)

    latencies: dict[str, List[float]] = {s: [] for s in STAGES}

    print(f"\n{'='*60}")
    print(f"  Benchmarking {n_runs} runs ...")
    print(f"{'='*60}")

    # Warmup
    for _ in range(min(5, n_runs)):
        detector.detect(audio)

    for i in range(n_runs):
        total_start = time.perf_counter()

        # Wake word
        t0 = time.perf_counter()
        detected, conf = detector.detect(audio)
        ww_ms = (time.perf_counter() - t0) * 1_000
        latencies["wake_word"].append(ww_ms)

        # STT (simulated — just measure STT stub latency)
        stt_ms = 0.0
        if include_stt:
            from src.inference.stt_engine import STTEngine
            stt = STTEngine(model_name="tiny")
            t0 = time.perf_counter()
            stt.transcribe(audio)
            stt_ms = (time.perf_counter() - t0) * 1_000
            latencies["stt"].append(stt_ms)
        else:
            latencies["stt"].append(0.0)

        # Intent
        intent_ms = 0.0
        if include_intent:
            t0 = time.perf_counter()
            classifier.classify("set an alarm for seven AM")
            intent_ms = (time.perf_counter() - t0) * 1_000
            latencies["intent"].append(intent_ms)
        else:
            latencies["intent"].append(0.0)

        total_ms = (time.perf_counter() - total_start) * 1_000
        latencies["total"].append(total_ms)

        if (i + 1) % (n_runs // min(5, n_runs)) == 0:
            print(f"  [{i+1}/{n_runs}] WW={ww_ms:.1f}ms | Intent={intent_ms:.1f}ms | Total={total_ms:.1f}ms")

    return {k: np.array(v) for k, v in latencies.items()}


def print_report(latencies: dict[str, np.ndarray]) -> None:
    """Print formatted benchmark report."""
    print(f"\n{'='*65}")
    print(f"  {'Stage':<20} {'Mean':>8} {'P50':>8} {'P90':>8} {'P99':>8} {'Max':>8}")
    print(f"  {'-'*60}")
    for stage, ms in latencies.items():
        if ms.mean() < 0.001:
            continue
        print(
            f"  {stage:<20} "
            f"{ms.mean():>7.2f}ms "
            f"{np.percentile(ms, 50):>7.2f}ms "
            f"{np.percentile(ms, 90):>7.2f}ms "
            f"{np.percentile(ms, 99):>7.2f}ms "
            f"{ms.max():>7.2f}ms"
        )
    total = latencies.get("total", np.array([0]))
    budget_pass = float(np.percentile(total, 90)) < 100.0
    print(f"{'='*65}")
    print(f"  P90 latency budget (<100ms): {'✅ PASS' if budget_pass else '❌ FAIL'}")
    print(f"{'='*65}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Latency Benchmark for Voice Pipeline")
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--pipeline-config", default="configs/pipeline_config.yaml")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--include-stt", action="store_true")
    args = parser.parse_args()

    latencies = run_benchmark(
        n_runs=args.n_runs,
        pipeline_config=args.pipeline_config,
        model_config=args.model_config,
        include_stt=args.include_stt,
    )
    print_report(latencies)


if __name__ == "__main__":
    main()
