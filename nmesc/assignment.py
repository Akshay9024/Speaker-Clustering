from __future__ import annotations

import numpy as np

from .config import AssignmentConfig
from .rttm import Turn
from .segmentation import Segment


def label_name(label: int) -> str:
    return f"spk{int(label):02d}"


def to_turns(
    recording_id: str,
    segments: list[Segment],
    labels: np.ndarray,
    intervals: list[tuple[float, float]],
    cfg: AssignmentConfig,
) -> list[Turn]:
    """Frames inside each oracle speech interval take the label of the nearest segment centre.

    Each interval is subdivided independently so emitted turn boundaries land exactly on the
    oracle speech boundaries rather than on a global frame grid.
    """
    if not segments or len(labels) == 0:
        return []

    centers = np.fromiter((s.center for s in segments), dtype=np.float64, count=len(segments))
    starts = np.fromiter((s.start for s in segments), dtype=np.float64, count=len(segments))
    ends = np.fromiter((s.end for s in segments), dtype=np.float64, count=len(segments))
    labels = np.asarray(labels, dtype=np.int64)

    turns: list[Turn] = []
    for interval_start, interval_end in intervals:
        span = interval_end - interval_start
        if span <= 0:
            continue
        n_frames = max(1, int(round(span / cfg.frame_sec)))
        edges = interval_start + np.arange(n_frames + 1) * (span / n_frames)
        midpoints = 0.5 * (edges[:-1] + edges[1:])

        right = np.searchsorted(centers, midpoints)
        left = np.clip(right - 1, 0, len(centers) - 1)
        right = np.clip(right, 0, len(centers) - 1)
        nearest = np.where(
            np.abs(midpoints - centers[left]) <= np.abs(centers[right] - midpoints), left, right
        )

        frame_labels = labels[nearest]
        if not cfg.fill_oracle_speech:
            covered = (midpoints >= starts[nearest]) & (midpoints < ends[nearest])
            frame_labels = np.where(covered, frame_labels, -1)

        turns.extend(_runs(recording_id, frame_labels, edges))
    return turns


def _runs(recording_id: str, frame_labels: np.ndarray, edges: np.ndarray) -> list[Turn]:
    change = np.empty(len(frame_labels), dtype=bool)
    change[0] = True
    change[1:] = frame_labels[1:] != frame_labels[:-1]
    boundaries = np.flatnonzero(change)

    turns = []
    for i, begin in enumerate(boundaries):
        label = int(frame_labels[begin])
        if label < 0:
            continue
        stop = boundaries[i + 1] if i + 1 < len(boundaries) else len(frame_labels)
        start, end = float(edges[begin]), float(edges[stop])
        if end > start:
            turns.append(Turn(recording_id, start, end - start, label_name(label)))
    return turns
