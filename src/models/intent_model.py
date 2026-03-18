"""
src/models/intent_model.py
---------------------------
Fine-tuned BERT wrapper for multi-class intent classification.
Wraps HuggingFace BertForSequenceClassification with convenience methods
for prediction and probability extraction.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F

from src.utils.logger import get_logger

logger = get_logger(__name__)


class IntentBERTClassifier:
    """
    Wraps ``transformers.BertForSequenceClassification`` for intent classification.

    Parameters
    ----------
    model_path : str | Path
        Path to a saved HuggingFace model directory **or** a model hub identifier
        such as ``"bert-base-uncased"`` (used during initial fine-tuning).
    num_labels : int
        Number of intent classes.
    max_length : int
        Maximum token length for BERT tokenisation.
    device : str | None
        ``"cuda"``, ``"cpu"``, or ``None`` (auto-detect).
    label_map : dict[int, str] | None
        Mapping from integer label index to human-readable intent name.
    """

    def __init__(
        self,
        model_path: str | Path = "bert-base-uncased",
        num_labels: int = 30,
        max_length: int = 128,
        device: Optional[str] = None,
        label_map: Optional[dict[int, str]] = None,
    ):
        from transformers import BertForSequenceClassification, BertTokenizerFast

        self.num_labels = num_labels
        self.max_length = max_length
        self.device = (
            device
            if device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.label_map = label_map or {i: f"intent_{i}" for i in range(num_labels)}

        logger.info(
            "Loading IntentBERTClassifier from '%s' (num_labels=%d, device=%s)",
            model_path,
            num_labels,
            self.device,
        )
        self.tokenizer = BertTokenizerFast.from_pretrained(str(model_path))
        self.model = BertForSequenceClassification.from_pretrained(
            str(model_path),
            num_labels=num_labels,
        )
        self.model.to(self.device)
        self.model.eval()
        logger.info("IntentBERTClassifier loaded successfully.")

    # ── Inference ─────────────────────────────────────────────────────────────

    def classify(
        self,
        text: str,
        top_k: int = 3,
    ) -> dict:
        """
        Classify a single utterance.

        Parameters
        ----------
        text : str
            Raw transcription text.
        top_k : int
            Return the top-k intent predictions.

        Returns
        -------
        dict with keys:
            - ``intent``       : str  — predicted intent label
            - ``confidence``   : float — softmax probability of top intent
            - ``top_k``        : list[dict] — top-k intent names + probabilities
            - ``raw_logits``   : list[float]
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits  # (1, num_labels)

        probs = F.softmax(logits, dim=-1).squeeze(0)  # (num_labels,)
        top_idx = torch.topk(probs, k=min(top_k, self.num_labels)).indices.tolist()

        result = {
            "intent": self.label_map[top_idx[0]],
            "confidence": float(probs[top_idx[0]]),
            "top_k": [
                {
                    "intent": self.label_map[i],
                    "confidence": float(probs[i]),
                }
                for i in top_idx
            ],
            "raw_logits": logits.squeeze(0).tolist(),
        }
        logger.debug("Classified '%s' → %s (%.2f%%)", text, result["intent"], result["confidence"] * 100)
        return result

    def classify_batch(
        self,
        texts: List[str],
    ) -> List[dict]:
        """Classify a list of utterances."""
        return [self.classify(t) for t in texts]

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, output_dir: str | Path) -> None:
        """Save the model and tokenizer to *output_dir*."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))
        logger.info("IntentBERTClassifier saved to '%s'", output_dir)

    @classmethod
    def load(
        cls,
        model_dir: str | Path,
        label_map: Optional[dict[int, str]] = None,
        device: Optional[str] = None,
    ) -> "IntentBERTClassifier":
        """Load a previously saved classifier."""
        from transformers import BertForSequenceClassification
        config = BertForSequenceClassification.from_pretrained(str(model_dir)).config
        return cls(
            model_path=model_dir,
            num_labels=config.num_labels,
            max_length=128,
            device=device,
            label_map=label_map,
        )
