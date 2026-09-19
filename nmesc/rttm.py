from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Turn:
    recording_id: str
    start: float
    duration: float
    speaker: str

    @property
    def end(self) -> float:
        return self.start + self.duration


def read_rttm(path: str | Path) -> list[Turn]:
    turns: list[Turn] = []
    for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
        fields = line.split()
        if not fields or fields[0] != "SPEAKER":
            continue
        if len(fields) < 8:
            raise ValueError(f"{path}:{lineno}: malformed SPEAKER line")
        turns.append(Turn(fields[1], float(fields[3]), float(fields[4]), fields[7]))
    return turns


def write_rttm(path: str | Path, turns: list[Turn]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"SPEAKER {t.recording_id} 1 {t.start:.3f} {t.duration:.3f} <NA> <NA> {t.speaker} <NA> <NA>"
        for t in turns
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def speaker_count(turns: list[Turn]) -> int:
    return len({t.speaker for t in turns})
