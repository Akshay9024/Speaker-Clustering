"""Speech-region VAD for audio without a reference RTTM, via pyannote/segmentation-3.0.

Replaces the oracle RTTM mask in oracle_vad for external audio. It only supplies
speech intervals; the speaker count is still estimated by NME-SC downstream.
"""

from __future__ import annotations

import os
from pathlib import Path


class SegmentationVad:
    def __init__(
        self,
        model: str = "pyannote/segmentation-3.0",
        device: str = "cpu",
        min_duration_on: float = 0.0,
        min_duration_off: float = 0.0,
        token: str | None = None,
    ):
        self.model = model
        self.device = device
        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off
        self.token = token or os.environ.get("HF_TOKEN") or True
        self._pipeline = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            import torch
            from pyannote.audio import Model
            from pyannote.audio.pipelines import VoiceActivityDetection

            model = Model.from_pretrained(self.model, use_auth_token=self.token)
            pipeline = VoiceActivityDetection(segmentation=model)
            pipeline.instantiate(
                {"min_duration_on": self.min_duration_on, "min_duration_off": self.min_duration_off}
            )
            self._pipeline = pipeline.to(torch.device(self.device))
        return self._pipeline

    def intervals(self, wav_path: str | Path) -> list[tuple[float, float]]:
        """Merged, sorted speech intervals in seconds."""
        annotation = self.pipeline(str(wav_path))
        return [(float(s.start), float(s.end)) for s in annotation.get_timeline().support()]
