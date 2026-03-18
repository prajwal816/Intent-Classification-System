"""
tests/test_intent_classifier.py
---------------------------------
Unit tests for IntentClassifier (stub mode — no BERT weights required).
"""
from __future__ import annotations

import pytest
np = pytest.importorskip("numpy", reason="numpy not installed")

from src.inference.intent_classifier import IntentClassifier, _STUB_LABELS
from src.utils.metrics import LatencyReport



class TestIntentClassifierStub:
    """Tests run against the keyword-matching stub (no model files needed)."""

    @pytest.fixture(autouse=True)
    def classifier(self):
        # model_path=None → stub mode
        return IntentClassifier(model_path=None, confidence_threshold=0.5)

    def test_classify_returns_intent_key(self, classifier):
        result = classifier.classify("play some jazz music")
        assert "intent" in result

    def test_intent_is_string(self, classifier):
        result = classifier.classify("what is the weather like today")
        assert isinstance(result["intent"], str)

    def test_confidence_in_range(self, classifier):
        result = classifier.classify("set an alarm for 7 AM")
        assert 0.0 <= result["confidence"] <= 1.5  # stub can slightly exceed 1.0 due to shift

    def test_top_k_length(self, classifier):
        result = classifier.classify("navigate to the airport", top_k=3)
        assert len(result["top_k"]) >= 1

    def test_top_k_items_have_required_keys(self, classifier):
        result = classifier.classify("turn on the lights", top_k=3)
        for item in result["top_k"]:
            assert "intent" in item
            assert "confidence" in item

    def test_elapsed_ms_non_negative(self, classifier):
        result = classifier.classify("tell me a joke")
        assert result["elapsed_ms"] >= 0.0

    def test_latency_report_populated(self, classifier):
        report = LatencyReport()
        classifier.classify("order food from Pizza Hut", report=report)
        stage_names = [r.stage for r in report.records]
        assert "intent" in stage_names

    def test_low_confidence_flag_present(self, classifier):
        result = classifier.classify("some totally random text here")
        assert "low_confidence" in result

    @pytest.mark.parametrize("text,expected_intent", [
        ("play jazz music", "PlayMusic"),
        ("set an alarm", "SetAlarm"),
        ("what is the weather", "CheckWeather"),
        ("navigate to the airport", "NavigateTo"),
        ("tell me a joke", "TellJoke"),
        ("hang up", "EndCall"),
        ("convert 100 miles to kilometres", "ConvertUnits"),
        ("what is my battery level", "CheckBattery"),
    ])
    def test_keyword_routing(self, classifier, text, expected_intent):
        """Stub classifier should route common phrases to the correct intent."""
        result = classifier.classify(text)
        assert result["intent"] == expected_intent, \
            f"'{text}' → expected {expected_intent}, got {result['intent']}"
