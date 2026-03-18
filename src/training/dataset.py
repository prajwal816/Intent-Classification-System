"""
src/training/dataset.py
------------------------
Synthetic dataset generation for wake-word detection and intent classification.

WakeWordDataset: generates MFCC tensors with positive (wake tone) and negative
                 (random noise) classes.

IntentDataset  : generates template-based text utterances across 30 intent classes
                 suitable for HuggingFace BERT fine-tuning.
"""
from __future__ import annotations

import random
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 30 Intent labels
# ─────────────────────────────────────────────────────────────────────────────

INTENT_LABELS: List[str] = [
    "PlayMusic",
    "PauseMusic",
    "StopMusic",
    "SetAlarm",
    "CancelAlarm",
    "SetTimer",
    "CheckWeather",
    "GetNews",
    "SendMessage",
    "ReadMessage",
    "MakeCall",
    "EndCall",
    "SetReminder",
    "NavigateTo",
    "FindRestaurant",
    "OrderFood",
    "TurnOnLight",
    "TurnOffLight",
    "SetVolume",
    "AdjustThermostat",
    "CheckCalendar",
    "AddCalendarEvent",
    "SearchWeb",
    "OpenApp",
    "CloseApp",
    "TellJoke",
    "GetDefinition",
    "TranslateSentence",
    "ConvertUnits",
    "CheckBattery",
]

INTENT_TEMPLATES: dict[str, List[str]] = {
    "PlayMusic": [
        "play {genre} music",
        "start playing {artist}",
        "I want to listen to {genre}",
        "put on some {genre}",
        "play some music for me",
    ],
    "PauseMusic": [
        "pause the music",
        "stop the song for a moment",
        "pause playback please",
        "hold the music",
    ],
    "StopMusic": [
        "stop the music",
        "turn off the music",
        "stop playing music",
        "end music",
    ],
    "SetAlarm": [
        "set an alarm for {time}",
        "wake me up at {time}",
        "alarm at {time} tomorrow",
        "set alarm {time}",
    ],
    "CancelAlarm": [
        "cancel my alarm",
        "dismiss the alarm",
        "remove alarm for tomorrow",
        "delete alarm",
    ],
    "SetTimer": [
        "set a timer for {duration}",
        "start a {duration} timer",
        "countdown {duration}",
        "timer for {duration} please",
    ],
    "CheckWeather": [
        "what is the weather like",
        "will it rain today",
        "what is the temperature outside",
        "weather forecast for {city}",
    ],
    "GetNews": [
        "give me the latest news",
        "what happened today",
        "read me the headlines",
        "any news updates",
    ],
    "SendMessage": [
        "send a message to {name}",
        "text {name} saying {text}",
        "message {name}",
        "send {name} a note",
    ],
    "ReadMessage": [
        "read my messages",
        "any new messages",
        "check my texts",
        "read the latest message from {name}",
    ],
    "MakeCall": [
        "call {name}",
        "dial {name}",
        "make a call to {name}",
        "ring {name}",
    ],
    "EndCall": [
        "hang up",
        "end the call",
        "disconnect the call",
        "stop the call",
    ],
    "SetReminder": [
        "remind me to {task} at {time}",
        "set a reminder for {task}",
        "don't let me forget to {task}",
        "reminder: {task}",
    ],
    "NavigateTo": [
        "navigate to {place}",
        "directions to {place}",
        "take me to {place}",
        "how do I get to {place}",
    ],
    "FindRestaurant": [
        "find restaurants near me",
        "where can I eat {cuisine}",
        "best {cuisine} restaurants nearby",
        "look up {cuisine} places",
    ],
    "OrderFood": [
        "order food from {restaurant}",
        "I want to order {food}",
        "get me {food} from {restaurant}",
        "place a food order",
    ],
    "TurnOnLight": [
        "turn on the lights",
        "lights on",
        "switch on the lights in {room}",
        "brighten the {room}",
    ],
    "TurnOffLight": [
        "turn off the lights",
        "lights off",
        "switch off the {room} lights",
        "dim the lights",
    ],
    "SetVolume": [
        "set volume to {level}",
        "increase the volume",
        "lower the volume",
        "volume at {level} percent",
    ],
    "AdjustThermostat": [
        "set thermostat to {temp} degrees",
        "make it warmer",
        "cool down the room",
        "temperature to {temp}",
    ],
    "CheckCalendar": [
        "what is on my calendar today",
        "any meetings tomorrow",
        "show my schedule",
        "what do I have at {time}",
    ],
    "AddCalendarEvent": [
        "add {event} to my calendar",
        "schedule {event} at {time}",
        "book {event} for {date}",
        "put {event} on my calendar",
    ],
    "SearchWeb": [
        "search for {query}",
        "look up {query} online",
        "find information about {query}",
        "google {query}",
    ],
    "OpenApp": [
        "open {app}",
        "launch {app}",
        "start {app}",
        "open the {app} app",
    ],
    "CloseApp": [
        "close {app}",
        "quit {app}",
        "exit {app}",
        "shut down {app}",
    ],
    "TellJoke": [
        "tell me a joke",
        "make me laugh",
        "say something funny",
        "I need a joke",
    ],
    "GetDefinition": [
        "what does {word} mean",
        "define {word}",
        "meaning of {word}",
        "what is the definition of {word}",
    ],
    "TranslateSentence": [
        "translate {phrase} to {language}",
        "how do you say {phrase} in {language}",
        "{phrase} in {language}",
    ],
    "ConvertUnits": [
        "convert {amount} {unit} to {target_unit}",
        "how many {target_unit} in {amount} {unit}",
        "{amount} {unit} to {target_unit}",
    ],
    "CheckBattery": [
        "what is my battery level",
        "how much battery do I have",
        "check battery percentage",
        "battery status",
    ],
}

_FILLERS: dict[str, List[str]] = {
    "genre": ["jazz", "pop", "rock", "classical", "hip hop", "lo-fi"],
    "artist": ["Taylor Swift", "The Beatles", "Drake", "Mozart"],
    "time": ["7 AM", "8:30", "noon", "6 PM", "midnight"],
    "duration": ["5 minutes", "10 minutes", "30 seconds", "1 hour"],
    "city": ["London", "Paris", "New York", "Tokyo"],
    "name": ["Alice", "Bob", "John", "Sarah"],
    "text": ["I am running late", "see you soon", "call me back"],
    "task": ["buy groceries", "attend the meeting", "send the report"],
    "place": ["the airport", "home", "downtown", "the gym"],
    "cuisine": ["Italian", "Mexican", "Chinese", "Indian"],
    "restaurant": ["Pizza Hut", "Subway", "McDonald's"],
    "food": ["a burger", "sushi", "pizza"],
    "room": ["living room", "bedroom", "kitchen"],
    "level": ["50", "70", "20", "80"],
    "temp": ["72", "68", "75"],
    "event": ["team meeting", "dentist appointment", "birthday party"],
    "date": ["Friday", "next Monday", "tomorrow"],
    "query": ["machine learning", "Python tutorials", "weather in Japan"],
    "app": ["Spotify", "YouTube", "Maps", "Gmail"],
    "word": ["ephemeral", "serendipity", "ubiquitous"],
    "phrase": ["good morning", "thank you", "where is the station"],
    "language": ["Spanish", "French", "Japanese", "German"],
    "amount": ["5", "100", "2.5"],
    "unit": ["miles", "kilograms", "Fahrenheit"],
    "target_unit": ["kilometres", "pounds", "Celsius"],
}


def _fill_template(template: str) -> str:
    """Replace {placeholder} tokens with random fillers."""
    import re
    placeholders = re.findall(r"\{(\w+)\}", template)
    text = template
    for ph in placeholders:
        choices = _FILLERS.get(ph, [ph])
        text = text.replace("{" + ph + "}", random.choice(choices), 1)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Wake Word Dataset
# ─────────────────────────────────────────────────────────────────────────────

class WakeWordDataset(Dataset):
    """
    Synthetic MFCC dataset for wake-word binary classification.

    Positive samples (label=1): MFCC of a 1 kHz tone (simulates keyword).
    Negative samples (label=0): MFCC of random Gaussian noise.

    Parameters
    ----------
    n_positive : int
        Number of wake-word (positive) samples.
    n_negative : int
        Number of background (negative) samples.
    n_mfcc : int
        Number of MFCC coefficients.
    max_frames : int
        Time dimension after padding/trimming.
    sample_rate : int
        Audio sample rate for MFCC extraction.
    seed : int
        Random seed for reproducibility.
    augment : bool
        If True, apply random noise + time-shift augmentation to positive samples.
    """

    def __init__(
        self,
        n_positive: int = 2000,
        n_negative: int = 2000,
        n_mfcc: int = 40,
        max_frames: int = 98,
        sample_rate: int = 16_000,
        seed: int = 42,
        augment: bool = False,
    ):
        from src.audio.feature_extraction import MFCCExtractor, MFCCConfig
        from src.audio.preprocessor import AudioPreprocessor

        rng = np.random.default_rng(seed)
        config = MFCCConfig(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            max_frames=max_frames,
            delta_order=0,
        )
        extractor = MFCCExtractor(config)
        preprocessor = AudioPreprocessor()
        audio_len = sample_rate  # 1 second

        self.features: List[torch.Tensor] = []
        self.labels: List[int] = []

        logger.info("Generating WakeWordDataset: %d pos + %d neg ...", n_positive, n_negative)

        # Positive samples: tone at 1 kHz with slight frequency variation
        for _ in range(n_positive):
            freq = rng.uniform(900, 1100)
            t = np.linspace(0, 1.0, audio_len, endpoint=False)
            audio = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
            if augment:
                audio += rng.normal(0, 0.03, size=audio_len).astype(np.float32)
            audio = preprocessor.process(audio)
            mfcc = extractor.extract(audio)
            self.features.append(torch.tensor(mfcc, dtype=torch.float32))
            self.labels.append(1)

        # Negative samples: white noise / pink-ish noise
        for i in range(n_negative):
            if i % 2 == 0:
                audio = rng.normal(0, 0.3, size=audio_len).astype(np.float32)
            else:
                # Low-frequency rumble
                freq = rng.uniform(60, 300)
                t = np.linspace(0, 1.0, audio_len, endpoint=False)
                audio = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
                audio += rng.normal(0, 0.1, size=audio_len).astype(np.float32)
            audio = preprocessor.process(audio)
            mfcc = extractor.extract(audio)
            self.features.append(torch.tensor(mfcc, dtype=torch.float32))
            self.labels.append(0)

        logger.info("WakeWordDataset ready: %d samples, shape=%s", len(self), self.features[0].shape)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], torch.tensor(self.labels[idx], dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Intent Dataset
# ─────────────────────────────────────────────────────────────────────────────

class IntentDataset:
    """
    Generates a synthetic text dataset for intent classification.

    Returns a HuggingFace ``datasets.Dataset`` suitable for Trainer.

    Parameters
    ----------
    n_per_class : int
        Number of text samples per intent class.
    seed : int
    """

    def __init__(self, n_per_class: int = 200, seed: int = 42):
        random.seed(seed)
        self.label2id = {label: i for i, label in enumerate(INTENT_LABELS)}
        self.id2label = {i: label for i, label in enumerate(INTENT_LABELS)}
        self.n_per_class = n_per_class
        logger.info(
            "IntentDataset: %d classes × %d samples = %d total",
            len(INTENT_LABELS),
            n_per_class,
            len(INTENT_LABELS) * n_per_class,
        )

    def generate(self) -> "datasets.Dataset":
        """Return a HuggingFace Dataset with columns: ``text``, ``label``."""
        try:
            from datasets import Dataset as HFDataset
        except ImportError as e:
            raise ImportError("Install `datasets`: pip install datasets") from e

        texts, labels = [], []
        for intent in INTENT_LABELS:
            templates = INTENT_TEMPLATES.get(intent, [f"please {intent.lower()}"])
            for _ in range(self.n_per_class):
                tmpl = random.choice(templates)
                texts.append(_fill_template(tmpl))
                labels.append(self.label2id[intent])

        # Shuffle
        pairs = list(zip(texts, labels))
        random.shuffle(pairs)
        texts, labels = zip(*pairs) if pairs else ([], [])

        return HFDataset.from_dict({"text": list(texts), "label": list(labels)})

    def generate_split(
        self,
        val_split: float = 0.15,
        test_split: float = 0.10,
    ) -> dict:
        """Return train/val/test DatasetDict splits."""
        from datasets import DatasetDict
        full = self.generate()
        n = len(full)
        n_test = int(n * test_split)
        n_val = int(n * val_split)
        n_train = n - n_val - n_test
        splits = full.train_test_split(test_size=n_test, seed=42)
        train_val = splits["train"].train_test_split(test_size=n_val, seed=42)
        return DatasetDict({
            "train": train_val["train"],
            "validation": train_val["test"],
            "test": splits["test"],
        })
