"""
deployment/convert_to_tflite.py
---------------------------------
Convert the trained PyTorch CNN-GRU wake-word model to TensorFlow Lite
via the ONNX → TF → TFLite export pipeline with dynamic-range quantisation.

Pipeline
--------
  PyTorch (.pt) → ONNX (.onnx) → TF SavedModel → TFLite (.tflite)

Usage
-----
    python deployment/convert_to_tflite.py
    python deployment/convert_to_tflite.py --model-path data/models/wake_word/cnn_gru_wake_word.pt \\
                                            --output-path deployment/wake_word.tflite
"""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from src.models.wake_word_model import WakeWordCNNGRU
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: PyTorch → ONNX
# ─────────────────────────────────────────────────────────────────────────────

def export_to_onnx(
    model: WakeWordCNNGRU,
    onnx_path: str,
    n_mfcc: int = 40,
    max_frames: int = 98,
    batch_size: int = 1,
) -> None:
    """Export PyTorch model to ONNX format."""
    model.eval()
    dummy_input = torch.randn(batch_size, n_mfcc, max_frames)
    logger.info("Exporting model to ONNX: %s", onnx_path)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["mfcc_input"],
        output_names=["wake_prob"],
        dynamic_axes={
            "mfcc_input": {0: "batch_size"},
            "wake_prob": {0: "batch_size"},
        },
    )
    logger.info("ONNX export complete: %s", onnx_path)

    # Validate ONNX model
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model check passed.")
    except ImportError:
        logger.warning("onnx not installed — skipping validation.")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: ONNX → TF SavedModel
# ─────────────────────────────────────────────────────────────────────────────

def onnx_to_tf_savedmodel(onnx_path: str, tf_saved_model_dir: str) -> None:
    """Convert ONNX model to TensorFlow SavedModel."""
    try:
        import tf2onnx  # noqa: F401
        import subprocess, sys
        logger.info("Converting ONNX → TF SavedModel: %s", tf_saved_model_dir)
        cmd = [
            sys.executable, "-m", "tf2onnx.convert",
            "--onnx", onnx_path,
            "--output", tf_saved_model_dir,
            "--opset", "12",
            "--target", "tensorflow",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("tf2onnx conversion error:\n%s", result.stderr)
            raise RuntimeError("ONNX → TF conversion failed.")
        logger.info("TF SavedModel written to: %s", tf_saved_model_dir)
    except ImportError:
        logger.error("tf2onnx not installed. Install with: pip install tf2onnx")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: TF SavedModel → TFLite
# ─────────────────────────────────────────────────────────────────────────────

def savedmodel_to_tflite(
    tf_saved_model_dir: str,
    tflite_path: str,
    quantize: bool = True,
) -> int:
    """
    Convert TF SavedModel to TFLite flatbuffer with optional dynamic-range
    quantisation.

    Returns the size of the output .tflite file in bytes.
    """
    try:
        import tensorflow as tf
        logger.info("Converting TF SavedModel → TFLite ...")
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_saved_model_dir)

        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            logger.info("Dynamic-range quantisation enabled.")

        tflite_model = converter.convert()

        Path(tflite_path).parent.mkdir(parents=True, exist_ok=True)
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)

        size_kb = len(tflite_model) / 1024
        logger.info("TFLite model saved: %s (%.1f KB)", tflite_path, size_kb)
        return len(tflite_model)
    except ImportError:
        logger.error("tensorflow not installed. Install with: pip install tensorflow")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def convert(
    model_path: str = "data/models/wake_word/cnn_gru_wake_word.pt",
    output_path: str = "deployment/wake_word.tflite",
    model_config_path: str = "configs/model_config.yaml",
    quantize: bool = True,
) -> None:
    model_cfg = load_config(model_config_path).wake_word_model
    n_mfcc = int(model_cfg.n_mfcc)
    max_frames = int(model_cfg.max_frames)

    # Load model
    logger.info("Loading wake-word model from: %s", model_path)
    model = WakeWordCNNGRU.from_config(dict(model_cfg))
    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint)
        logger.info("Checkpoint loaded successfully.")
    else:
        logger.warning("No checkpoint at '%s' — using random weights.", model_path)

    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_path = os.path.join(tmp_dir, "wake_word.onnx")
        tf_dir = os.path.join(tmp_dir, "savedmodel")

        export_to_onnx(model, onnx_path, n_mfcc=n_mfcc, max_frames=max_frames)
        onnx_to_tf_savedmodel(onnx_path, tf_dir)
        size = savedmodel_to_tflite(tf_dir, output_path, quantize=quantize)

    print(f"\n✅ TFLite model saved: {output_path} ({size / 1024:.1f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Wake Word Model to TFLite")
    parser.add_argument("--model-path", default="data/models/wake_word/cnn_gru_wake_word.pt")
    parser.add_argument("--output-path", default="deployment/wake_word.tflite")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--no-quantize", action="store_true")
    args = parser.parse_args()
    convert(
        model_path=args.model_path,
        output_path=args.output_path,
        model_config_path=args.model_config,
        quantize=not args.no_quantize,
    )


if __name__ == "__main__":
    main()
