"""Warm-loaded speaker-count pipeline for the web app.

Wraps the existing offline flow (real VAD -> segment -> ECAPA -> NME-SC) that
``nmesc.meet._external`` runs, but loads the VAD and embedder **once** so they are
reused across requests instead of being rebuilt per call. The app never uses oracle
VAD: an uploaded file has no reference, so speech regions come from
``pyannote/segmentation-3.0`` via :class:`nmesc.vad.SegmentationVad`.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from nmesc import audio, nme_sc
from nmesc.config import load_config
from nmesc.embedding import EcapaEmbedder, resolve_device
from nmesc.segmentation import segment_intervals
from nmesc.vad import SegmentationVad

# soundfile (libsndfile) reads these directly; anything else goes through ffmpeg.
_SOUNDFILE_EXT = {".wav", ".flac", ".ogg", ".oga", ".aiff", ".aif", ".aifc", ".w64"}


class AudioTooLong(ValueError):
    """Raised when an upload exceeds the configured duration cap."""


class SpeakerCounter:
    """Reusable pipeline: load models once, then :meth:`count` per upload."""

    def __init__(
        self,
        config_path: str | Path = "configs/voxconverse.yaml",
        token: str | None = None,
        device: str | None = None,
        max_duration_sec: float = 3600.0,
    ) -> None:
        self.cfg = load_config(config_path)
        # resolve_device("cuda") -> cuda if a GPU is present, else mps, else cpu.
        self.device = device or resolve_device("cuda")
        self.max_duration_sec = max_duration_sec
        self.vad = SegmentationVad(model=self.cfg.vad.model, device=self.device, token=token)
        self.embedder = EcapaEmbedder(
            replace(self.cfg.embedding, device=self.device), Path(self.cfg.root)
        )

    def warmup(self) -> None:
        """Force the lazy model loads so the first request isn't slow."""
        _ = self.vad.pipeline
        _ = self.embedder.model

    def count(self, wav_path: str | Path) -> dict:
        """Estimate the number of speakers in one audio file.

        Returns a dict with ``num_speakers`` plus a few diagnostics. ``num_speakers``
        is 0 when the VAD finds no speech.
        """
        source = Path(wav_path)
        decoded = self._ensure_decodable(source)
        try:
            duration = audio.probe_duration(decoded)
            if self.max_duration_sec and duration > self.max_duration_sec:
                raise AudioTooLong(
                    f"audio is {duration:.0f}s; limit is {self.max_duration_sec:.0f}s"
                )

            intervals = self.vad.intervals(decoded)
            segments = segment_intervals([(s, e) for s, e in intervals], self.cfg.segmentation)
            speech_sec = float(sum(e - s for s, e in intervals))
            if not segments:
                return {
                    "num_speakers": 0,
                    "num_segments": 0,
                    "speech_sec": speech_sec,
                    "duration_sec": float(duration),
                    "device": self.device,
                }

            waveform = audio.load_mono(decoded, self.cfg.embedding.sample_rate)
            embeddings = self.embedder.encode(
                waveform, segments, self.cfg.segmentation.window_sec
            )
            result = nme_sc.estimate(embeddings, replace(self.cfg.clustering, device=self.device))
            return {
                "num_speakers": int(result.k),
                "num_segments": int(result.n_segments),
                "speech_sec": speech_sec,
                "duration_sec": float(duration),
                "device": self.device,
            }
        finally:
            if decoded != source:
                decoded.unlink(missing_ok=True)

    def _ensure_decodable(self, source: Path) -> Path:
        """Return a path soundfile can read, transcoding via ffmpeg if needed."""
        if source.suffix.lower() in _SOUNDFILE_EXT:
            return source
        if shutil.which("ffmpeg") is None:
            raise ValueError(
                f"cannot read {source.suffix or 'this'} audio without ffmpeg; "
                "upload a .wav/.flac/.ogg file or install ffmpeg on the server"
            )
        tmp = Path(tempfile.mkdtemp(prefix="spk_")) / "decoded.wav"
        # 16 kHz mono wav; load_mono would resample anyway, but this keeps it small.
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", str(source), "-ac", "1", "-ar",
             str(self.cfg.embedding.sample_rate), str(tmp)],
            check=True,
        )
        return tmp


if __name__ == "__main__":  # pragma: no cover - manual smoke test on the GPU box
    import sys

    counter = SpeakerCounter(token=None)
    counter.warmup()
    print(counter.count(sys.argv[1]))
