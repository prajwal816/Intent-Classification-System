"""
data/data_generator.py
-----------------------
Standalone script that generates synthetic training data and saves it to disk.

Generates
---------
  data/raw/wake_word/
      positive_<N>.wav   — 1 kHz tone samples
      negative_<N>.wav   — noise samples

  data/raw/intents/
      train.tsv  — tab-separated text\tlabel
      val.tsv
      test.tsv

Usage
-----
    python data/data_generator.py
    python data/data_generator.py --n-wake 500 --n-intent 100
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)

SAMPLE_RATE = 16_000


# ─────────────────────────────────────────────────────────────────────────────
# Wake Word WAV Generator
# ─────────────────────────────────────────────────────────────────────────────

def _save_wav(path: str, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    try:
        import soundfile as sf
        sf.write(path, audio, sr, subtype="PCM_16")
    except ImportError:
        # Fallback: minimal WAV writer (raw PCM, 16-bit, mono)
        import struct, wave
        audio_i16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        with wave.open(path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(audio_i16.tobytes())


def generate_wake_word_wavs(
    out_dir: str = "data/raw/wake_word",
    n_positive: int = 200,
    n_negative: int = 200,
    sr: int = SAMPLE_RATE,
    seed: int = 42,
) -> None:
    rng = np.random.default_rng(seed)
    pos_dir = Path(out_dir) / "positive"
    neg_dir = Path(out_dir) / "negative"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    t = np.linspace(0, 1.0, sr, endpoint=False)

    logger.info("Generating %d positive wake-word WAV files ...", n_positive)
    for i in range(n_positive):
        freq = rng.uniform(900, 1100)
        audio = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        audio += rng.normal(0, 0.02, size=sr).astype(np.float32)
        _save_wav(str(pos_dir / f"positive_{i:04d}.wav"), audio, sr)

    logger.info("Generating %d negative WAV files ...", n_negative)
    for i in range(n_negative):
        if i % 3 == 0:
            audio = rng.normal(0, 0.3, size=sr).astype(np.float32)
        elif i % 3 == 1:
            freq = rng.uniform(60, 300)
            audio = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        else:
            # Random mixture
            audio = rng.normal(0, 0.2, size=sr).astype(np.float32)
            for _ in range(rng.integers(1, 4)):
                f = rng.uniform(100, 3000)
                audio += 0.1 * np.sin(2 * np.pi * f * t).astype(np.float32)
        audio = np.clip(audio, -1, 1)
        _save_wav(str(neg_dir / f"negative_{i:04d}.wav"), audio, sr)

    logger.info("Wake word WAVs saved to: %s", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Intent TSV Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_intent_tsvs(
    out_dir: str = "data/raw/intents",
    n_per_class: int = 100,
    val_split: float = 0.15,
    test_split: float = 0.10,
    seed: int = 42,
) -> None:
    # Import dataset module
    from src.training.dataset import IntentDataset
    from datasets import DatasetDict

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    dataset = IntentDataset(n_per_class=n_per_class, seed=seed)
    splits = dataset.generate_split(val_split=val_split, test_split=test_split)

    for split_name, ds in splits.items():
        tsv_path = Path(out_dir) / f"{split_name}.tsv"
        with open(tsv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["text", "label", "intent_name"])
            label_names = dataset.id2label
            for row in ds:
                writer.writerow([row["text"], row["label"], label_names[row["label"]]])
        logger.info("Saved %d rows → %s", len(ds), tsv_path)

    logger.info("Intent TSVs saved to: %s", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--wake-out", default="data/raw/wake_word")
    parser.add_argument("--intent-out", default="data/raw/intents")
    parser.add_argument("--n-wake", type=int, default=500,
                        help="Number of positive and negative wake-word WAVs each")
    parser.add_argument("--n-intent", type=int, default=200,
                        help="Number of intent samples per class")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("\n🎙️  Generating Wake Word WAVs ...")
    generate_wake_word_wavs(
        out_dir=args.wake_out,
        n_positive=args.n_wake,
        n_negative=args.n_wake,
        seed=args.seed,
    )

    print("\n📝  Generating Intent TSVs ...")
    generate_intent_tsvs(
        out_dir=args.intent_out,
        n_per_class=args.n_intent,
        seed=args.seed,
    )

    print("\n✅  Data generation complete.")
    print(f"   Wake word WAVs : {args.wake_out}/")
    print(f"   Intent TSVs   : {args.intent_out}/")


if __name__ == "__main__":
    main()
