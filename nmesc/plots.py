"""Deterministic figures for the speaker-count analysis (clustering stage only)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def eigen_figure(
    eigenvalues: np.ndarray,
    eigengaps: np.ndarray,
    k_pred: int,
    k_true: int,
    k_max: int,
    out: Path,
) -> None:
    """Smallest eigenvalues at p* and the eigengaps that pick K̂ = argmax gap."""
    top = int(min(len(eigengaps), k_max + 3))
    idx = np.arange(1, top + 1)
    gaps = eigengaps[:top]

    fig, (ax_lambda, ax_gap) = plt.subplots(1, 2, figsize=(11, 4))

    ax_lambda.plot(np.arange(1, top + 1), eigenvalues[:top], "o-", color="#1f77b4")
    ax_lambda.axvline(k_pred, color="#d62728", ls="--", label=f"K̂ = {k_pred}")
    ax_lambda.axvline(k_true, color="#2ca02c", ls=":", label=f"K = {k_true}")
    ax_lambda.set_xlabel("eigenvalue index i")
    ax_lambda.set_ylabel("$\\lambda_i$")
    ax_lambda.set_title("Smallest eigenvalues at p*")
    ax_lambda.legend()

    colors = ["#d62728" if i == k_pred else "#7f7f7f" for i in idx]
    ax_gap.bar(idx, gaps, color=colors)
    if k_max < top:
        ax_gap.axvspan(k_max + 0.5, top + 0.5, color="#000000", alpha=0.05, label="beyond k_max")
    ax_gap.axvline(k_true, color="#2ca02c", ls=":", label=f"K = {k_true}")
    ax_gap.set_xlabel("candidate speaker count k")
    ax_gap.set_ylabel("eigengap $g_k = \\lambda_{k+1} - \\lambda_k$")
    ax_gap.set_title(f"Max eigengap at k = {k_pred} → K̂ = {k_pred}")
    ax_gap.legend()

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def psweep_figure(p_values: np.ndarray, r_values: np.ndarray, p_star: int, out: Path) -> None:
    """The NME ratio r(p) over the p sweep; p* = argmin r(p)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(p_values, r_values, "-", color="#1f77b4")
    ax.axvline(p_star, color="#d62728", ls="--", label=f"p* = {p_star}")
    ax.set_xlabel("p (nearest neighbours per row)")
    ax.set_ylabel("r(p) = p / max-eigengap ratio (log scale)")
    ax.set_title("NME ratio sweep — p* minimises r(p)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def clustering_figure(labels: np.ndarray, k_pred: int, k_true: int, out: Path) -> None:
    """Spectral-clustering output: segments assigned to each cluster, K̂ against true K."""
    uniq, counts = np.unique(np.asarray(labels, dtype=np.int64), return_counts=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([f"spk{int(u):02d}" for u in uniq], counts, color="#1f77b4")
    ax.set_xlabel("predicted cluster")
    ax.set_ylabel("segments")
    ax.set_title(f"Spectral-clustering output: K̂ = {k_pred}  vs  true K = {k_true}")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def utterance_figure(
    intervals: list[tuple[float, float]], duration: float, out: Path
) -> None:
    """Input utterances fed to the embedder: the VAD speech intervals on a timeline."""
    speech = sum(e - s for s, e in intervals)
    fig, ax = plt.subplots(figsize=(11, 2.2))
    for s, e in intervals:
        ax.axvspan(s, e, ymin=0.2, ymax=0.8, color="#1f77b4")
    ax.set_xlim(0, duration)
    ax.set_yticks([])
    ax.set_xlabel("time (s)")
    ax.set_title(
        f"Input utterances: {len(intervals)} speech regions, "
        f"{speech:.1f} s of {duration:.1f} s ({100 * speech / duration:.0f}%)"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def render_all(data: dict, k_true: int, k_max: int, duration: float, out_dir: Path) -> list[Path]:
    """Write the four meeting figures for one recording into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    k_pred = int(data["k_pred"])
    paths = []

    if len(data["eigengaps"]):
        eigen = out_dir / "eigen.png"
        eigen_figure(data["eigenvalues"], data["eigengaps"], k_pred, k_true, k_max, eigen)
        paths.append(eigen)

    if len(data["p_values"]):
        psweep = out_dir / "psweep.png"
        psweep_figure(data["p_values"], data["r_values"], int(data["p_star"]), psweep)
        paths.append(psweep)

    clustering = out_dir / "clustering.png"
    clustering_figure(data["labels"], k_pred, k_true, clustering)
    paths.append(clustering)

    utter = out_dir / "utterances.png"
    utterance_figure([(s, e) for s, e in data["intervals"]], duration, utter)
    paths.append(utter)
    return paths
