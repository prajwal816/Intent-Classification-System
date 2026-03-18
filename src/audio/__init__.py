"""
src/audio/__init__.py — lazy imports to avoid hard dependency errors at test-collection time.
"""

def __getattr__(name):
    if name == "MFCCExtractor":
        from .feature_extraction import MFCCExtractor
        return MFCCExtractor
    if name == "AudioPreprocessor":
        from .preprocessor import AudioPreprocessor
        return AudioPreprocessor
    if name == "AudioStreamer":
        from .audio_capture import AudioStreamer
        return AudioStreamer
    raise AttributeError(f"module 'src.audio' has no attribute {name!r}")

__all__ = ["MFCCExtractor", "AudioPreprocessor", "AudioStreamer"]
