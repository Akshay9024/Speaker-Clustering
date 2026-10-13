from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetConfig:
    name: str
    rttm_dir: str
    audio_dir: str
    audio_template: str = "{rec}.wav"


@dataclass
class SegmentationConfig:
    window_sec: float = 1.5
    hop_sec: float = 0.75
    min_interval_sec: float = 1.5
    merge_gap_sec: float = 0.0
    center_single_window: bool = True


@dataclass
class EmbeddingConfig:
    model: str = "speechbrain/spkrec-ecapa-voxceleb"
    savedir: str = "checkpoints/ecapa"
    sample_rate: int = 16000
    batch_size: int = 64
    device: str = "cuda"


@dataclass
class SingleSpeakerConfig:
    enabled: bool = False
    mean_cosine_threshold: float = 0.50


@dataclass
class ClusteringConfig:
    # "unnormalized" is Park et al. eq. (7); "sym_norm" is the I - D^-1/2 A D^-1/2 variant.
    laplacian: str = "unnormalized"
    p_min: int = 1
    p_max_ratio: float = 0.25
    p_max_abs: int | None = None
    p_stride: int = 1
    k_max: int = 10
    zero_diagonal: bool = True
    eps: float = 1e-10
    device: str = "cuda"
    dtype: str = "float32"
    row_normalize_spectral_embedding: bool = False
    min_segments: int = 2
    kmeans_n_init: int = 10
    seed: int = 0
    single_speaker: SingleSpeakerConfig = field(default_factory=SingleSpeakerConfig)


@dataclass
class VadConfig:
    source: str = "oracle"  # oracle | pyannote
    model: str = "pyannote/segmentation-3.0"
    min_duration_on: float = 0.0
    min_duration_off: float = 0.0
    device: str = "cpu"


@dataclass
class AssignmentConfig:
    enabled: bool = True
    frame_sec: float = 0.01
    fill_oracle_speech: bool = True


@dataclass
class DerVariant:
    name: str
    collar: float = 0.0
    skip_overlap: bool = False


@dataclass
class EvaluationConfig:
    der: bool = True
    rel_error_tol: float = 0.15
    der_variants: list[DerVariant] = field(
        default_factory=lambda: [
            DerVariant("collar250_nooverlap", 0.25, True),
            DerVariant("strict", 0.0, False),
        ]
    )


@dataclass
class PathsConfig:
    work_dir: str = "work"
    results_dir: str = "results"


@dataclass
class Config:
    dataset: DatasetConfig
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    assignment: AssignmentConfig = field(default_factory=AssignmentConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    root: str = "."

    def path(self, *parts: str) -> Path:
        return Path(self.root).joinpath(*parts)

    @property
    def work(self) -> Path:
        return self.path(self.paths.work_dir, self.dataset.name)

    @property
    def results(self) -> Path:
        return self.path(self.paths.results_dir, self.dataset.name)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    for dotted, val in (overrides or {}).items():
        node = raw
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = val

    cfg = Config(
        dataset=DatasetConfig(**raw["dataset"]),
        segmentation=SegmentationConfig(**raw.get("segmentation", {})),
        embedding=EmbeddingConfig(**raw.get("embedding", {})),
        clustering=_clustering(raw.get("clustering", {})),
        vad=VadConfig(**raw.get("vad", {})),
        assignment=AssignmentConfig(**raw.get("assignment", {})),
        evaluation=_evaluation(raw.get("evaluation", {})),
        paths=PathsConfig(**raw.get("paths", {})),
        root=raw.get("root", str(path.resolve().parent.parent)),
    )
    _validate(cfg)
    return cfg


def _clustering(raw: dict) -> ClusteringConfig:
    raw = dict(raw)
    single = raw.pop("single_speaker", {})
    return ClusteringConfig(single_speaker=SingleSpeakerConfig(**single), **raw)


def _evaluation(raw: dict) -> EvaluationConfig:
    raw = dict(raw)
    variants = raw.pop("der_variants", None)
    cfg = EvaluationConfig(**raw)
    if variants is not None:
        cfg.der_variants = [DerVariant(**v) for v in variants]
    return cfg


def _validate(cfg: Config) -> None:
    if cfg.clustering.laplacian not in ("unnormalized", "sym_norm"):
        raise ValueError(f"clustering.laplacian must be unnormalized|sym_norm, got {cfg.clustering.laplacian!r}")
    if cfg.clustering.dtype not in ("float32", "float64"):
        raise ValueError(f"clustering.dtype must be float32|float64, got {cfg.clustering.dtype!r}")
    if cfg.segmentation.hop_sec <= 0 or cfg.segmentation.window_sec <= 0:
        raise ValueError("segmentation window_sec and hop_sec must be positive")
    if cfg.segmentation.min_interval_sec < cfg.segmentation.window_sec:
        raise ValueError("segmentation.min_interval_sec must be >= window_sec (segments must not cross silence)")
    if cfg.clustering.k_max < 1:
        raise ValueError("clustering.k_max must be >= 1")
    if cfg.clustering.p_min < 1:
        raise ValueError("clustering.p_min must be >= 1")
    if cfg.vad.source not in ("oracle", "pyannote"):
        raise ValueError(f"vad.source must be oracle|pyannote, got {cfg.vad.source!r}")


def as_dict(cfg: Config) -> dict:
    return dataclasses.asdict(cfg)
