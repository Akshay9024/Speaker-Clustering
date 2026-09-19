from __future__ import annotations

from collections import Counter

import numpy as np

from .config import DerVariant
from .rttm import Turn


def to_annotation(turns: list[Turn], uri: str):
    from pyannote.core import Annotation, Segment

    annotation = Annotation(uri=uri)
    for i, t in enumerate(turns):
        if t.duration > 0:
            annotation[Segment(t.start, t.end), i] = t.speaker
    return annotation


def der(reference: list[Turn], hypothesis: list[Turn], uri: str, variant: DerVariant) -> dict:
    from pyannote.metrics.diarization import DiarizationErrorRate

    metric = DiarizationErrorRate(collar=variant.collar, skip_overlap=variant.skip_overlap)
    detail = metric(to_annotation(reference, uri), to_annotation(hypothesis, uri), detailed=True)
    total = float(detail["total"])
    return {
        f"der_{variant.name}": float(detail["diarization error rate"]),
        f"miss_{variant.name}": float(detail["missed detection"]) / total if total else 0.0,
        f"fa_{variant.name}": float(detail["false alarm"]) / total if total else 0.0,
        f"conf_{variant.name}": float(detail["confusion"]) / total if total else 0.0,
        f"total_{variant.name}": total,
    }


def count_metrics(k_true: np.ndarray, k_pred: np.ndarray) -> dict:
    k_true = np.asarray(k_true, dtype=np.int64)
    k_pred = np.asarray(k_pred, dtype=np.int64)
    if len(k_true) == 0:
        return {"n": 0}
    error = np.abs(k_pred - k_true)
    return {
        "n": int(len(k_true)),
        "exact_match": float((error == 0).mean()),
        "within_1": float((error <= 1).mean()),
        "mae": float(error.mean()),
        "bias": float((k_pred - k_true).mean()),
    }


def confusion(k_true: np.ndarray, k_pred: np.ndarray) -> tuple[list[int], list[int], np.ndarray]:
    pairs = Counter(zip(map(int, k_true), map(int, k_pred)))
    rows = sorted({t for t, _ in pairs})
    cols = sorted({p for _, p in pairs})
    matrix = np.zeros((len(rows), len(cols)), dtype=np.int64)
    for (t, p), count in pairs.items():
        matrix[rows.index(t), cols.index(p)] = count
    return rows, cols, matrix


def weighted_der(frame, variant_name: str) -> float:
    """Corpus-level DER: per-recording rates weighted by scored reference duration."""
    weights = frame[f"total_{variant_name}"]
    total = weights.sum()
    if total <= 0:
        return float("nan")
    return float((frame[f"der_{variant_name}"] * weights).sum() / total)
