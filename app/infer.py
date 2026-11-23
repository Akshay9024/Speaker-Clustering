"""Warm-loaded audio decoding, VAD, ECAPA and NME-SC for the web application."""

from __future__ import annotations

import logging
import math
import shutil
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import TypedDict

import soundfile as sf

from nmesc import audio, nme_sc
from nmesc.config import load_config
from nmesc.embedding import EcapaEmbedder, resolve_device
from nmesc.segmentation import segment_intervals
from nmesc.vad import SegmentationVad

_LOG = logging.getLogger(__name__)
_NATIVE_FORMATS = {"WAV", "WAVEX", "RF64", "FLAC"}


class CountResult(TypedDict):
    num_speakers: int
    num_segments: int
    speech_sec: float
    duration_sec: float
    device: str


class AudioTooLong(ValueError):
    """The audio exceeds the configured duration limit."""


class InvalidAudio(ValueError):
    """The file does not contain a readable, nonempty audio stream."""


def ffmpeg_executable() -> str | None:
    """Find system FFmpeg or the executable bundled by the pinned wheel."""
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except RuntimeError:
        return None


class SpeakerCounter:
    """Load the models once and reuse them for individual audio recordings."""

    def __init__(
        self,
        config_path: str | Path = "configs/voxconverse.yaml",
        token: str | None = None,
        device: str | None = None,
        max_duration_sec: float = 3600.0,
        decode_timeout_sec: float = 120.0,
    ) -> None:
        if not math.isfinite(max_duration_sec) or max_duration_sec < 0:
            raise ValueError("max_duration_sec must be finite and nonnegative")
        if not math.isfinite(decode_timeout_sec) or decode_timeout_sec <= 0:
            raise ValueError("decode_timeout_sec must be finite and positive")
        self.cfg = load_config(config_path)
        self.device = resolve_device(device or "cuda")
        self.max_duration_sec = max_duration_sec
        self.decode_timeout_sec = decode_timeout_sec
        self.vad = SegmentationVad(
            model=self.cfg.vad.model,
            device=self.device,
            token=token,
            min_duration_on=self.cfg.vad.min_duration_on,
            min_duration_off=self.cfg.vad.min_duration_off,
        )
        self.embedder = EcapaEmbedder(
            replace(self.cfg.embedding, device=self.device), Path(self.cfg.root)
        )

    def warmup(self) -> None:
        """Load the lazy models before accepting requests."""
        _ = self.vad.pipeline
        _ = self.embedder.model

    def _check_duration(self, duration: float) -> None:
        if not math.isfinite(duration) or duration <= 0:
            raise InvalidAudio("This file contains no readable audio. Choose another recording.")
        if self.max_duration_sec and duration > self.max_duration_sec:
            raise AudioTooLong(
                f"Audio exceeds the {self.max_duration_sec:g}-second limit. Choose a shorter clip."
            )

    def count(self, wav_path: str | Path) -> CountResult:
        """Return the speaker estimate and measured durations for one audio file."""
        source = Path(wav_path)
        decoded = self._ensure_decodable(source)
        try:
            duration = audio.probe_duration(decoded)
            self._check_duration(duration)
            intervals = self.vad.intervals(decoded)
            segments = segment_intervals(intervals, self.cfg.segmentation)
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
            embeddings = self.embedder.encode(waveform, segments, self.cfg.segmentation.window_sec)
            # Short uploads may have fewer windows than the configured pruning floor.
            clustering = replace(
                self.cfg.clustering,
                device=self.device,
                p_min=min(self.cfg.clustering.p_min, max(1, len(embeddings) - 1)),
            )
            result = nme_sc.estimate(embeddings, clustering)
            return {
                "num_speakers": int(result.k),
                "num_segments": int(result.n_segments),
                "speech_sec": speech_sec,
                "duration_sec": float(duration),
                "device": self.device,
            }
        finally:
            if decoded != source:
                shutil.rmtree(decoded.parent)

    def _ensure_decodable(self, source: Path) -> Path:
        """Inspect the content and decode compressed audio or video into mono WAV."""
        if not source.is_file() or not source.stat().st_size:
            raise InvalidAudio("The audio file is empty or unavailable. Choose another recording.")
        try:
            info = sf.info(source)
        except (sf.LibsndfileError, RuntimeError):
            info = None
        if info is not None:
            self._check_duration(info.duration)
            if info.format in _NATIVE_FORMATS:
                return source

        executable = ffmpeg_executable()
        if executable is None:
            raise InvalidAudio(
                "This format needs the server's audio decoder. Use WAV/FLAC or install "
                "the dependencies from requirements.txt."
            )
        directory = Path(tempfile.mkdtemp(prefix="spk_decode_"))
        decoded = directory / "decoded.wav"
        command = [
            executable, "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
            "-protocol_whitelist", "file,pipe", "-i", str(source),
            "-map", "0:a:0", "-vn", "-sn", "-dn", "-ac", "1",
            "-ar", str(self.cfg.embedding.sample_rate), "-c:a", "pcm_s16le",
        ]
        if self.max_duration_sec:
            # One second beyond the cap rejects long inputs instead of silently truncating.
            command += ["-t", str(self.max_duration_sec + 1.0)]
        command.append(str(decoded))
        try:
            try:
                subprocess.run(
                    command, check=True, capture_output=True, text=True,
                    timeout=self.decode_timeout_sec,
                )
            except subprocess.TimeoutExpired as exc:
                raise InvalidAudio("Audio conversion timed out. Try a shorter recording.") from exc
            except subprocess.CalledProcessError as exc:
                _LOG.info("Audio decoder rejected an upload: %s", exc.stderr)
                raise InvalidAudio(
                    "This file has no decodable audio stream or is damaged. "
                    "Try WAV, MP3, M4A, FLAC, OGG or WebM."
                ) from exc
            self._check_duration(audio.probe_duration(decoded))
            return decoded
        except BaseException:
            shutil.rmtree(directory)
            raise


if __name__ == "__main__":  # pragma: no cover - GPU smoke test
    import sys

    counter = SpeakerCounter()
    counter.warmup()
    print(counter.count(sys.argv[1]))
