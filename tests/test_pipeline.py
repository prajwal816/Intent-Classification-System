"""
tests/test_pipeline.py
------------------------
Integration tests for the end-to-end VoicePipeline (all stages in stub/mock mode).
"""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.inference.pipeline import VoicePipeline
from src.inference.wake_word_detector import WakeWordDetector
from src.inference.stt_engine import STTEngine
from src.inference.intent_classifier import IntentClassifier
from src.utils.metrics import LatencyReport, LatencyTracker


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def stub_detector_detected():
    """WakeWordDetector that always detects the wake word."""
    d = MagicMock(spec=WakeWordDetector)
    d.detect.return_value = (True, 0.95)
    return d


@pytest.fixture
def stub_detector_not_detected():
    """WakeWordDetector that never detects the wake word."""
    d = MagicMock(spec=WakeWordDetector)
    d.detect.return_value = (False, 0.10)
    return d


@pytest.fixture
def stub_stt():
    stt = MagicMock(spec=STTEngine)
    stt.transcribe.return_value = {
        "text": "set an alarm for seven AM",
        "language": "en",
        "elapsed_ms": 12.5,
        "segments": [],
    }
    return stt


@pytest.fixture
def stub_classifier():
    clf = MagicMock(spec=IntentClassifier)
    clf.classify.return_value = {
        "intent": "SetAlarm",
        "confidence": 0.92,
        "top_k": [{"intent": "SetAlarm", "confidence": 0.92}],
        "low_confidence": False,
        "elapsed_ms": 5.0,
    }
    return clf


@pytest.fixture
def pipeline_detected(stub_detector_detected, stub_stt, stub_classifier):
    return VoicePipeline(
        wake_word_detector=stub_detector_detected,
        stt_engine=stub_stt,
        intent_classifier=stub_classifier,
        latency_budget_ms=100,
    )


@pytest.fixture
def pipeline_not_detected(stub_detector_not_detected, stub_stt, stub_classifier):
    return VoicePipeline(
        wake_word_detector=stub_detector_not_detected,
        stt_engine=stub_stt,
        intent_classifier=stub_classifier,
        latency_budget_ms=100,
    )


@pytest.fixture
def audio_chunk():
    return np.random.randn(16000).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestVoicePipelineDetected:

    def test_result_has_intent_key(self, pipeline_detected, audio_chunk):
        result = pipeline_detected.process_chunk(audio_chunk)
        assert "intent" in result

    def test_wake_detected_true(self, pipeline_detected, audio_chunk):
        result = pipeline_detected.process_chunk(audio_chunk)
        assert result["wake_detected"] is True

    def test_wake_confidence_value(self, pipeline_detected, audio_chunk):
        result = pipeline_detected.process_chunk(audio_chunk)
        assert result["wake_confidence"] == pytest.approx(0.95, abs=1e-3)

    def test_transcription_present(self, pipeline_detected, audio_chunk):
        result = pipeline_detected.process_chunk(audio_chunk)
        assert "transcription" in result
        assert isinstance(result["transcription"], str)
        assert len(result["transcription"]) > 0

    def test_intent_is_set_alarm(self, pipeline_detected, audio_chunk):
        result = pipeline_detected.process_chunk(audio_chunk)
        assert result["intent"] == "SetAlarm"

    def test_total_ms_non_negative(self, pipeline_detected, audio_chunk):
        result = pipeline_detected.process_chunk(audio_chunk)
        assert result["total_ms"] >= 0

    def test_within_budget_key_present(self, pipeline_detected, audio_chunk):
        result = pipeline_detected.process_chunk(audio_chunk)
        assert "within_budget" in result

    def test_latency_report_in_result(self, pipeline_detected, audio_chunk):
        result = pipeline_detected.process_chunk(audio_chunk)
        assert "latency_report" in result
        assert "stages" in result["latency_report"]
        assert "total_ms" in result["latency_report"]


class TestVoicePipelineNotDetected:

    def test_wake_detected_false(self, pipeline_not_detected, audio_chunk):
        result = pipeline_not_detected.process_chunk(audio_chunk)
        assert result["wake_detected"] is False

    def test_no_intent_when_not_detected(self, pipeline_not_detected, audio_chunk):
        result = pipeline_not_detected.process_chunk(audio_chunk)
        assert "intent" not in result

    def test_stt_not_called_when_not_detected(self, pipeline_not_detected, audio_chunk, stub_stt):
        pipeline_not_detected.process_chunk(audio_chunk)
        stub_stt.transcribe.assert_not_called()


class TestStreamProcessing:

    def test_stream_yields_only_on_detection(self, pipeline_detected):
        chunks = [np.random.randn(16000).astype(np.float32) for _ in range(3)]
        results = list(pipeline_detected.process_stream(chunks))
        assert len(results) == 3  # all chunks trigger detection

    def test_stream_yields_nothing_when_no_detection(self, pipeline_not_detected):
        chunks = [np.random.randn(16000).astype(np.float32) for _ in range(5)]
        results = list(pipeline_not_detected.process_stream(chunks))
        assert len(results) == 0  # no chunks trigger detection


class TestLatencyTracker:

    def test_elapsed_ms_non_negative(self):
        import time
        report = LatencyReport()
        with LatencyTracker("test_stage", report) as t:
            time.sleep(0.001)
        assert t.elapsed_ms >= 0.0

    def test_report_records_stage(self):
        report = LatencyReport()
        with LatencyTracker("stt", report):
            pass
        assert any(r.stage == "stt" for r in report.records)

    def test_report_total_ms(self):
        report = LatencyReport()
        with LatencyTracker("stage_a", report):
            pass
        with LatencyTracker("stage_b", report):
            pass
        assert report.total_ms >= 0.0
        assert len(report.records) == 2

    def test_report_to_dict(self):
        report = LatencyReport()
        with LatencyTracker("wake_word", report):
            pass
        d = report.to_dict()
        assert "stages" in d
        assert "total_ms" in d
        assert "wake_word" in d["stages"]
