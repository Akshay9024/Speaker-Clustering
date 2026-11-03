from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import Config
from .metrics import confusion, count_metrics, relative_count_metrics, weighted_der


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def build(cfg: Config, frame) -> str:
    counts = count_metrics(frame["k_true"].to_numpy(), frame["k_pred"].to_numpy())
    tol = cfg.evaluation.rel_error_tol
    rel = relative_count_metrics(frame["k_true"].to_numpy(), frame["k_pred"].to_numpy(), tol)
    clustering = cfg.clustering

    parts = [
        f"# NME-SC speaker-count baseline — {cfg.dataset.name}",
        "",
        "## Configuration",
        "",
        _table(
            ["Setting", "Value"],
            [
                ["Embedding", cfg.embedding.model],
                ["Window / hop", f"{cfg.segmentation.window_sec} s / {cfg.segmentation.hop_sec} s"],
                [
                    "VAD",
                    "oracle (reference RTTM)"
                    if cfg.vad.source == "oracle"
                    else f"pyannote ({cfg.vad.model})",
                ],
                ["Laplacian", clustering.laplacian],
                [
                    "p sweep",
                    f"[{clustering.p_min}, floor({clustering.p_max_ratio}·N)]"
                    + (f" capped at {clustering.p_max_abs}" if clustering.p_max_abs else "")
                    + (f", stride {clustering.p_stride}" if clustering.p_stride > 1 else ""),
                ],
                ["k_max", str(clustering.k_max)],
                [
                    "Single-speaker shortcut",
                    f"enabled (τ={clustering.single_speaker.mean_cosine_threshold})"
                    if clustering.single_speaker.enabled
                    else "disabled",
                ],
                ["Eigen dtype / device", f"{clustering.dtype} / {clustering.device}"],
            ],
        ),
        "",
        "## Speaker-count accuracy",
        "",
        _table(
            ["Metric", "Value"],
            [
                ["Recordings", str(counts["n"])],
                ["Exact-match accuracy", f"{counts['exact_match']:.4f}"],
                ["±1 accuracy", f"{counts['within_1']:.4f}"],
                ["MAE", f"{counts['mae']:.4f}"],
                ["Mean signed error (K̂ − K)", f"{counts['bias']:+.4f}"],
            ],
        ),
        "",
        "## Relative speaker-count accuracy",
        "",
        f"Error normalised by the true count: rel_err = |K̂ − K| / K; correct when rel_err ≤ τ = {tol}.",
        "",
        _table(
            ["Metric", "Value"],
            [
                [f"Relative accuracy (τ={tol})", f"{rel['rel_accuracy']:.4f}"],
                ["Mean relative error", f"{rel['rel_mae']:.4f}"],
                ["Median relative error", f"{rel['rel_median']:.4f}"],
            ],
        ),
    ]

    der_columns = [c for c in frame.columns if c.startswith("der_")]
    if der_columns:
        rows = []
        for column in der_columns:
            name = column[len("der_") :]
            rows.append(
                [
                    name,
                    f"{weighted_der(frame, name) * 100:.2f}",
                    f"{frame[column].mean() * 100:.2f}",
                    f"{frame[f'miss_{name}'].mean() * 100:.2f}",
                    f"{frame[f'fa_{name}'].mean() * 100:.2f}",
                    f"{frame[f'conf_{name}'].mean() * 100:.2f}",
                ]
            )
        parts += [
            "",
            "## Diarization error rate (%)",
            "",
            "Duration-weighted is the corpus-level DER; the remaining columns are unweighted "
            "per-recording means.",
            "",
            _table(
                ["Variant", "Duration-weighted", "Mean", "Miss", "False alarm", "Confusion"], rows
            ),
        ]

    rows_k, cols_k, matrix = confusion(frame["k_true"].to_numpy(), frame["k_pred"].to_numpy())
    parts += [
        "",
        "## Per-true-K confusion (rows = true K, cols = predicted K̂)",
        "",
        _table(
            ["true K \\ K̂"] + [str(c) for c in cols_k],
            [
                [str(rows_k[i])] + [str(v) if v else "·" for v in matrix[i]]
                for i in range(len(rows_k))
            ],
        ),
    ]

    per_k = []
    for k in rows_k:
        subset = frame[frame["k_true"] == k]
        stats = count_metrics(subset["k_true"].to_numpy(), subset["k_pred"].to_numpy())
        per_k.append(
            [
                str(k),
                str(stats["n"]),
                f"{stats['exact_match']:.3f}",
                f"{stats['within_1']:.3f}",
                f"{stats['mae']:.3f}",
                f"{subset['k_pred'].mean():.2f}",
            ]
        )
    parts += [
        "",
        "## Breakdown by true K",
        "",
        _table(["true K", "n", "Exact", "±1", "MAE", "mean K̂"], per_k),
        "",
        "## Notes",
        "",
        f"- Segments: {int(frame['n_segments'].sum()):,} across {len(frame)} recordings "
        f"(min {int(frame['n_segments'].min())}, median {int(frame['n_segments'].median())}, "
        f"max {int(frame['n_segments'].max())}).",
        f"- p* range observed: {int(frame['p_star'].min())}–{int(frame['p_star'].max())}.",
        f"- Clustering wall time: {frame['cluster_sec'].sum() / 60:.1f} min total.",
    ]
    if "n_components" in frame.columns:
        graphed = frame[frame["n_components"] > 0]
        frag = graphed[graphed["n_components"] > 1]
        parts.append(
            f"- Binarised-affinity connected components at p*: "
            f"min {int(graphed['n_components'].min())}, median {int(graphed['n_components'].median())}, "
            f"max {int(graphed['n_components'].max())}; {len(frag)}/{len(graphed)} recordings fragmented (>1)."
            if len(graphed)
            else "- Binarised-affinity connected components at p*: no graphed recordings."
        )
    if "saturated" in frame.columns:
        sat = frame[frame["saturated"]]["recording_id"].tolist()
        parts.append(
            f"- K_max saturation (K̂ == min(k_max, N−1)): {len(sat)}/{len(frame)} recordings"
            + (f" [{', '.join(sat)}]" if sat else "")
            + "."
        )
    parts += [
        "- Raw per-recording values are in `predictions.csv`; re-run `report` to rebuild this file.",
        "",
    ]
    return "\n".join(parts)


def write(cfg: Config, frame) -> Path:
    cfg.results.mkdir(parents=True, exist_ok=True)
    path = cfg.results / "summary.md"
    path.write_text(build(cfg, frame))
    return path
