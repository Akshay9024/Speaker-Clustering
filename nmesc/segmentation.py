from __future__ import annotations

import math
from dataclasses import dataclass

from .config import SegmentationConfig

_TOL = 1e-9


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    interval_index: int

    @property
    def center(self) -> float:
        return 0.5 * (self.start + self.end)


def segment_intervals(
    intervals: list[tuple[float, float]], cfg: SegmentationConfig
) -> list[Segment]:
    """Sliding windows laid inside each speech interval; never crossing silence."""
    window, hop = cfg.window_sec, cfg.hop_sec
    segments: list[Segment] = []

    for idx, (start, end) in enumerate(intervals):
        span = end - start
        if span + _TOL < cfg.min_interval_sec:
            continue
        n = int(math.floor((span - window) / hop + _TOL)) + 1
        if n < 1:
            continue
        if n == 1 and cfg.center_single_window:
            offset = 0.5 * (span - window)
            segments.append(Segment(start + offset, start + offset + window, idx))
            continue
        for i in range(n):
            s = start + i * hop
            segments.append(Segment(s, s + window, idx))
    return segments
