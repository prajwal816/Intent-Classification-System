"""
src/inference/__init__.py — lazy imports to avoid hard dependency errors at test-collection time.
"""

def __getattr__(name):
    if name == "WakeWordDetector":
        from .wake_word_detector import WakeWordDetector
        return WakeWordDetector
    if name == "STTEngine":
        from .stt_engine import STTEngine
        return STTEngine
    if name == "IntentClassifier":
        from .intent_classifier import IntentClassifier
        return IntentClassifier
    if name == "VoicePipeline":
        from .pipeline import VoicePipeline
        return VoicePipeline
    raise AttributeError(f"module 'src.inference' has no attribute {name!r}")

__all__ = ["WakeWordDetector", "STTEngine", "IntentClassifier", "VoicePipeline"]
