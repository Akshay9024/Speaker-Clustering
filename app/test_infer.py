"""Model-free checks for the inference wrapper.

These run without loading ECAPA/pyannote (and so without a GPU): construction is
lazy, and the duration cap is enforced *before* the VAD runs. The real end-to-end
smoke test is `python -m app.infer <wav>` on the GPU box.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from app.infer import AudioTooLong, SpeakerCounter


def _silent_wav(path: Path, seconds: float = 1.0, sr: int = 16000) -> None:
    sf.write(path, np.zeros(int(seconds * sr), dtype="float32"), sr)


def test_duration_cap_rejects_long_audio() -> None:
    counter = SpeakerCounter(max_duration_sec=0.001)  # no warmup -> no models
    with tempfile.TemporaryDirectory() as d:
        wav = Path(d) / "a.wav"
        _silent_wav(wav, seconds=1.0)
        try:
            counter.count(wav)
        except AudioTooLong:
            pass
        else:
            raise AssertionError("expected AudioTooLong for a 1s clip under a 0.001s cap")


def test_wav_passthrough_no_transcode() -> None:
    counter = SpeakerCounter(max_duration_sec=3600)
    with tempfile.TemporaryDirectory() as d:
        wav = Path(d) / "a.wav"
        _silent_wav(wav)
        assert counter._ensure_decodable(wav) == wav  # .wav is read directly, no ffmpeg


if __name__ == "__main__":
    test_duration_cap_rejects_long_audio()
    test_wav_passthrough_no_transcode()
    print("ok")
