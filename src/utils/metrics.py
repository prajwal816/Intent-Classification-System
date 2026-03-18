"""
src/utils/metrics.py
--------------------
Evaluation metrics for wake word detection and intent classification,
plus a context-manager LatencyTracker for timing pipeline stages.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, List, Optional

import numpy as np

# ── Optional sklearn ──────────────────────────────────────────────────────────
try:
    from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKLEARN_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Wake-word metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_tpr_far(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute True Positive Rate (TPR / Recall) and False Acceptance Rate (FAR).

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0 or 1).
    y_pred : np.ndarray
        Predicted probability scores in [0, 1].
    threshold : float
        Decision threshold.

    Returns
    -------
    dict with keys: tpr, far, precision, f1
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_bin = (np.asarray(y_pred) >= threshold).astype(int)

    tp = int(np.sum((y_pred_bin == 1) & (y_true == 1)))
    fn = int(np.sum((y_pred_bin == 0) & (y_true == 1)))
    fp = int(np.sum((y_pred_bin == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred_bin == 0) & (y_true == 0)))

    tpr = tp / (tp + fn + 1e-9)
    far = fp / (fp + tn + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    f1 = 2 * tpr * precision / (tpr + precision + 1e-9)

    return {
        "tpr": round(tpr, 4),
        "far": round(far, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def compute_roc_auc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> dict[str, float | np.ndarray]:
    """
    Compute ROC-AUC and return the full ROC curve.

    Returns
    -------
    dict with keys: auc, fpr, tpr, thresholds
    """
    if not _SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for compute_roc_auc()")

    auc = float(roc_auc_score(y_true, y_scores))
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    return {"auc": round(auc, 4), "fpr": fpr, "tpr": tpr, "thresholds": thresholds}


# ─────────────────────────────────────────────────────────────────────────────
# Classification metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Return classification accuracy."""
    if _SKLEARN_AVAILABLE:
        return float(accuracy_score(y_true, y_pred))
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


# ─────────────────────────────────────────────────────────────────────────────
# Latency tracking
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LatencyRecord:
    stage: str
    elapsed_ms: float


@dataclass
class LatencyReport:
    records: List[LatencyRecord] = field(default_factory=list)

    def add(self, stage: str, elapsed_ms: float) -> None:
        self.records.append(LatencyRecord(stage=stage, elapsed_ms=elapsed_ms))

    @property
    def total_ms(self) -> float:
        return sum(r.elapsed_ms for r in self.records)

    def summary(self) -> str:
        lines = ["── Latency Report ─────────────────"]
        for r in self.records:
            lines.append(f"  {r.stage:<30} {r.elapsed_ms:>8.2f} ms")
        lines.append(f"  {'TOTAL':<30} {self.total_ms:>8.2f} ms")
        lines.append("───────────────────────────────────")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "stages": {r.stage: round(r.elapsed_ms, 3) for r in self.records},
            "total_ms": round(self.total_ms, 3),
        }


class LatencyTracker:
    """
    Context manager for timing code blocks.

    Usage
    -----
    >>> report = LatencyReport()
    >>> with LatencyTracker("wake_word", report) as t:
    ...     run_model()
    >>> print(report.summary())
    """

    def __init__(self, stage: str, report: Optional[LatencyReport] = None):
        self.stage = stage
        self.report = report if report is not None else LatencyReport()
        self._start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "LatencyTracker":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1_000
        self.report.add(self.stage, self.elapsed_ms)


@contextmanager
def timed_stage(
    stage: str,
    report: Optional[LatencyReport] = None,
) -> Generator[LatencyTracker, None, None]:
    """Functional-style alias for :class:`LatencyTracker`."""
    tracker = LatencyTracker(stage, report)
    with tracker:
        yield tracker
