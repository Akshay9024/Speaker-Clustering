from __future__ import annotations

from .rttm import Turn


def speech_intervals(
    turns: list[Turn],
    duration: float | None = None,
    merge_gap_sec: float = 0.0,
) -> list[tuple[float, float]]:
    """Union of all speaker turns: overlap collapses into one speech interval."""
    spans = []
    for t in turns:
        start, end = t.start, t.end
        if duration is not None:
            start, end = max(0.0, start), min(end, duration)
        if end > start:
            spans.append((start, end))
    if not spans:
        return []

    spans.sort()
    merged = [list(spans[0])]
    for start, end in spans[1:]:
        if start - merged[-1][1] <= merge_gap_sec:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def total_duration(intervals: list[tuple[float, float]]) -> float:
    return sum(e - s for s, e in intervals)
