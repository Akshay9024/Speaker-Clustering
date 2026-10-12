"""Meeting deliverable: per-sample eigen/clustering plots and relative-error accuracy.

Scope stops at the clustering stage; nothing here depends on diarization/DER. Samples
are either cached AMI recordings or external audio with VAD from pyannote/segmentation-3.0
and a manually established true speaker count.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from . import nme_sc, plots
from .config import load_config
from .segmentation import segment_intervals


def _cached_ami(cfg, rec: str) -> tuple[dict, int, float]:
    npz_path = cfg.work / "clustering" / f"{rec}.npz"
    seg_path = cfg.work / "segments" / f"{rec}.json"
    if not npz_path.exists() or not seg_path.exists():
        raise FileNotFoundError(f"missing cached artifacts for {rec}: {npz_path} / {seg_path}")
    payload = json.loads(seg_path.read_text())
    with np.load(npz_path, allow_pickle=False) as d:
        data = {k: d[k] for k in d.files}
    data["intervals"] = payload["intervals"]
    return data, int(payload["k_true"]), float(payload["duration"])


def _external(cfg, wav: Path, device: str, vad) -> tuple[dict, float]:
    from .audio import load_mono, probe_duration
    from .embedding import EcapaEmbedder

    duration = probe_duration(wav)
    intervals = vad.intervals(wav)
    segments = segment_intervals([(s, e) for s, e in intervals], cfg.segmentation)
    if segments:
        embedder = EcapaEmbedder(replace(cfg.embedding, device=device), Path(cfg.root))
        waveform = load_mono(wav, cfg.embedding.sample_rate)
        embeddings = embedder.encode(waveform, segments, cfg.segmentation.window_sec)
    else:
        embeddings = np.zeros((0, 192), dtype=np.float32)

    result = nme_sc.estimate(embeddings, replace(cfg.clustering, device=device))
    data = {
        "k_pred": np.int64(result.k),
        "p_star": np.int64(result.p),
        "n_segments": np.int64(result.n_segments),
        "p_values": result.p_values,
        "r_values": result.r_values,
        "eigenvalues": result.eigenvalues,
        "eigengaps": result.eigengaps,
        "labels": result.labels,
        "intervals": intervals,
    }
    return data, duration


def _sample_summary(name: str, source: str, k_true: int, data: dict, tol: float) -> str:
    k_pred = int(data["k_pred"])
    rel = abs(k_pred - k_true) / k_true
    rows = [
        ("sample", name),
        ("source", source),
        ("true K", str(k_true)),
        ("predicted K̂", str(k_pred)),
        ("p*", str(int(data["p_star"]))),
        ("segments", str(int(data["n_segments"]))),
        ("relative error", f"{rel:.4f}"),
        (f"correct (τ={tol})", "yes" if rel <= tol else "no"),
    ]
    lines = [f"# {name}", "", "| Field | Value |", "| --- | --- |"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    return "\n".join(lines) + "\n"


def run(meet_config: str | Path, token: str | None = None) -> None:
    meet_path = Path(meet_config)
    spec = yaml.safe_load(meet_path.read_text())
    root = meet_path.resolve().parent
    out_root = root / spec.get("out_dir", "results_meet")
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = load_config(root / spec["ami_config"])
    device = spec.get("device", "cpu")
    tol = cfg.evaluation.rel_error_tol
    k_max = cfg.clustering.k_max

    vad = None
    table: list[tuple[str, str, int, int, float]] = []

    for sample in spec["samples"]:
        name = sample["name"]
        source = sample["source"]
        out_dir = out_root / name

        if source == "ami":
            data, k_true, duration = _cached_ami(cfg, sample["recording_id"])
        elif source == "external":
            wav = (root / sample["wav"]).resolve()
            k_true = sample.get("true_k")
            if k_true is None or not wav.exists():
                print(f"skip {name}: need existing wav and true_k (wav={wav}, true_k={k_true})")
                continue
            if vad is None:
                v = spec.get("vad", {})
                from .vad import SegmentationVad

                vad = SegmentationVad(device=device, token=token, **v)
            data, duration = _external(cfg, wav, device, vad)
            k_true = int(k_true)
        else:
            raise ValueError(f"unknown sample source {source!r} for {name}")

        plots.render_all(data, k_true, k_max, duration, out_dir)
        (out_dir / "summary.md").write_text(_sample_summary(name, source, k_true, data, tol))
        k_pred = int(data["k_pred"])
        rel = abs(k_pred - k_true) / k_true
        table.append((name, source, k_true, k_pred, rel))
        print(f"{name}: K̂={k_pred} K={k_true} rel_err={rel:.3f} → {out_dir}")

    _write_overview(out_root, table, tol)


def _write_overview(out_root: Path, table, tol: float) -> None:
    lines = [
        "# Meeting deliverable — speaker-count via NME-SC (clustering stage only)",
        "",
        f"Relative error = |K̂ − K| / K; correct when ≤ τ = {tol}.",
        "",
        "| Sample | Source | True K | K̂ | Rel. error | Correct |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, source, k_true, k_pred, rel in table:
        lines.append(
            f"| {name} | {source} | {k_true} | {k_pred} | {rel:.3f} | "
            f"{'yes' if rel <= tol else 'no'} |"
        )
    if table:
        acc = sum(1 for *_, rel in table if rel <= tol) / len(table)
        lines += ["", f"Relative accuracy over {len(table)} samples (τ={tol}): **{acc:.3f}**", ""]
    (out_root / "summary.md").write_text("\n".join(lines) + "\n")
