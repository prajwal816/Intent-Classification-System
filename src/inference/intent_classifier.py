"""
src/inference/intent_classifier.py
------------------------------------
High-level intent classifier backed by IntentBERTClassifier with config integration
and latency tracking.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from src.utils.logger import get_logger
from src.utils.metrics import LatencyReport, LatencyTracker

logger = get_logger(__name__)

_STUB_LABELS = [
    "PlayMusic", "PauseMusic", "StopMusic", "SetAlarm", "CancelAlarm",
    "SetTimer", "CheckWeather", "GetNews", "SendMessage", "ReadMessage",
    "MakeCall", "EndCall", "SetReminder", "NavigateTo", "FindRestaurant",
    "OrderFood", "TurnOnLight", "TurnOffLight", "SetVolume", "AdjustThermostat",
    "CheckCalendar", "AddCalendarEvent", "SearchWeb", "OpenApp", "CloseApp",
    "TellJoke", "GetDefinition", "TranslateSentence", "ConvertUnits", "CheckBattery",
]


class IntentClassifier:
    """
    Intent classification with automatic model loading and latency tracking.

    Falls back to a keyword-matching stub if BERT is unavailable or the model
    directory does not exist (useful during integration testing).

    Parameters
    ----------
    model_path : str | Path
        Directory containing a saved ``BertForSequenceClassification`` model.
    num_labels : int
    max_length : int
    confidence_threshold : float
        Minimum confidence to consider a classification confident.
    device : str | None
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        num_labels: int = 30,
        max_length: int = 128,
        confidence_threshold: float = 0.50,
        device: Optional[str] = None,
    ):
        self.confidence_threshold = confidence_threshold
        self._stub_mode = False
        self._bert: Optional[object] = None
        self.label_map: dict[int, str] = {i: l for i, l in enumerate(_STUB_LABELS)}

        if model_path and Path(str(model_path)).exists():
            label_map_path = Path(str(model_path)) / "label_map.json"
            if label_map_path.exists():
                with open(label_map_path) as f:
                    raw = json.load(f)
                self.label_map = {int(k): v for k, v in raw.items()}

            try:
                from src.models.intent_model import IntentBERTClassifier
                self._bert = IntentBERTClassifier(
                    model_path=model_path,
                    num_labels=num_labels,
                    max_length=max_length,
                    device=device,
                    label_map=self.label_map,
                )
                logger.info("IntentClassifier loaded BERT from '%s'", model_path)
            except Exception as exc:
                logger.warning(
                    "Could not load BERT model (%s). Falling back to stub.", exc
                )
                self._stub_mode = True
        else:
            logger.warning(
                "model_path '%s' not found. IntentClassifier running in stub mode.", model_path
            )
            self._stub_mode = True

    # ── Public API ────────────────────────────────────────────────────────────

    def classify(
        self,
        text: str,
        top_k: int = 3,
        report: Optional[LatencyReport] = None,
    ) -> dict:
        """
        Classify an utterance.

        Parameters
        ----------
        text : str
        top_k : int
        report : LatencyReport | None

        Returns
        -------
        dict with keys: intent, confidence, top_k, low_confidence, elapsed_ms
        """
        with LatencyTracker("intent", report) as tracker:
            if self._stub_mode:
                result = self._stub_classify(text, top_k)
            else:
                result = self._bert.classify(text, top_k=top_k)

        result["elapsed_ms"] = round(tracker.elapsed_ms, 3)
        result["low_confidence"] = result["confidence"] < self.confidence_threshold
        logger.info(
            "Intent: '%s' → %s (%.2f%%, %.1f ms)",
            text[:60],
            result["intent"],
            result["confidence"] * 100,
            result["elapsed_ms"],
        )
        return result

    # ── Private ───────────────────────────────────────────────────────────────

    def _stub_classify(self, text: str, top_k: int = 3) -> dict:
        """Keyword-matching stub — deterministic, fast, no model required."""
        text_lower = text.lower()

        rules = [
            (["play", "music", "song", "listen"], "PlayMusic"),
            (["pause", "hold"], "PauseMusic"),
            (["stop music", "turn off music"], "StopMusic"),
            (["alarm"], "SetAlarm"),
            (["cancel alarm", "dismiss alarm"], "CancelAlarm"),
            (["timer"], "SetTimer"),
            (["weather", "temperature", "rain"], "CheckWeather"),
            (["news", "headlines"], "GetNews"),
            (["send message", "text"], "SendMessage"),
            (["read message", "check message"], "ReadMessage"),
            (["call", "dial", "ring"], "MakeCall"),
            (["hang up", "end call"], "EndCall"),
            (["remind", "reminder"], "SetReminder"),
            (["navigate", "directions"], "NavigateTo"),
            (["restaurant", "eat", "food nearby"], "FindRestaurant"),
            (["order food", "order from"], "OrderFood"),
            (["turn on light", "lights on"], "TurnOnLight"),
            (["turn off light", "lights off"], "TurnOffLight"),
            (["volume"], "SetVolume"),
            (["thermostat", "temperature"], "AdjustThermostat"),
            (["calendar", "schedule", "meeting"], "CheckCalendar"),
            (["add event", "book"], "AddCalendarEvent"),
            (["search", "google", "look up"], "SearchWeb"),
            (["open", "launch", "start app"], "OpenApp"),
            (["close", "quit", "exit"], "CloseApp"),
            (["joke", "funny"], "TellJoke"),
            (["define", "meaning", "definition"], "GetDefinition"),
            (["translate"], "TranslateSentence"),
            (["convert"], "ConvertUnits"),
            (["battery"], "CheckBattery"),
        ]

        matched_intent = "CheckWeather"  # fallback
        for keywords, intent in rules:
            if any(k in text_lower for k in keywords):
                matched_intent = intent
                break

        # Build a fake distribution
        rng = np.random.default_rng(hash(text) % (2**31))
        probs = rng.dirichlet(np.ones(len(_STUB_LABELS)) * 0.5)
        top_idx = int(np.argmax([1.0 if l == matched_intent else p for l, p in zip(_STUB_LABELS, probs)]))

        # Force matched intent to top
        top_indices = [_STUB_LABELS.index(matched_intent)] + sorted(
            range(len(_STUB_LABELS)), key=lambda i: -probs[i]
        )[:top_k]
        top_indices = list(dict.fromkeys(top_indices))[:top_k]

        return {
            "intent": matched_intent,
            "confidence": round(float(probs[top_indices[0]] + 0.5), 4),
            "top_k": [
                {
                    "intent": _STUB_LABELS[i],
                    "confidence": round(float(probs[i]), 4),
                }
                for i in top_indices
            ],
            "raw_logits": [],
            "elapsed_ms": 0.0,
        }

    @classmethod
    def from_config(cls, pipeline_cfg) -> "IntentClassifier":
        intent = pipeline_cfg.intent
        return cls(
            model_path=intent.model_path,
            num_labels=int(intent.num_labels),
            max_length=int(intent.max_length),
            confidence_threshold=float(intent.confidence_threshold),
        )
