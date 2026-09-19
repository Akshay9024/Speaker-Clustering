from __future__ import annotations

import numpy as np
import torch

from .config import EmbeddingConfig
from .segmentation import Segment


def resolve_device(requested: str) -> str:
    if requested != "cuda":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class EcapaEmbedder:
    """SpeechBrain ECAPA-TDNN, one L2-normalized 192-d embedding per segment."""

    def __init__(self, cfg: EmbeddingConfig, root):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self._savedir = str(root / cfg.savedir)
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from speechbrain.inference.speaker import EncoderClassifier

            self._model = EncoderClassifier.from_hparams(
                source=self.cfg.model,
                savedir=self._savedir,
                run_opts={"device": self.device},
            )
            self._model.eval()
        return self._model

    @torch.inference_mode()
    def encode(self, waveform: np.ndarray, segments: list[Segment], window_sec: float) -> np.ndarray:
        sr = self.cfg.sample_rate
        n_samples = int(round(window_sec * sr))
        if not segments:
            return np.zeros((0, 192), dtype=np.float32)

        wav = torch.from_numpy(waveform)
        starts = np.fromiter((s.start for s in segments), dtype=np.float64, count=len(segments))
        offsets = np.rint(starts * sr).astype(np.int64)
        offsets = np.clip(offsets, 0, max(0, wav.numel() - n_samples))

        out = np.empty((len(segments), 192), dtype=np.float32)
        batch = self.cfg.batch_size
        lengths = torch.ones(batch, device=self.device)

        for lo in range(0, len(segments), batch):
            chunk = offsets[lo : lo + batch]
            frames = torch.stack([wav[o : o + n_samples] for o in chunk]).to(self.device)
            emb = self.model.encode_batch(frames, lengths[: len(chunk)]).squeeze(1)
            emb = torch.nn.functional.normalize(emb, p=2.0, dim=-1)
            out[lo : lo + len(chunk)] = emb.float().cpu().numpy()
        return out
