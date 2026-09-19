from __future__ import annotations

from pathlib import Path

import numpy as np


def probe_duration(path: str | Path) -> float:
    import soundfile as sf

    info = sf.info(str(path))
    return info.frames / info.samplerate


def load_mono(path: str | Path, target_sr: int) -> np.ndarray:
    import soundfile as sf

    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != target_sr:
        import torch
        import torchaudio.functional as AF

        wav = AF.resample(torch.from_numpy(wav), sr, target_sr).numpy()
    return np.ascontiguousarray(wav, dtype=np.float32)
