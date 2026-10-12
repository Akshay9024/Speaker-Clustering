#!/usr/bin/env python3
"""NME-SC speaker-count baseline: stage runner and inspector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nmesc import pipeline, report  # noqa: E402
from nmesc.config import load_config  # noqa: E402
from nmesc.embedding import EcapaEmbedder, resolve_device  # noqa: E402


def _parse_override(text: str):
    key, _, raw = text.partition("=")
    if not _:
        raise argparse.ArgumentTypeError(f"--set expects key=value, got {text!r}")
    try:
        return key, json.loads(raw)
    except json.JSONDecodeError:
        return key, raw


def _select(cfg, args) -> list[str]:
    recs = pipeline.recording_ids(cfg)
    if args.only:
        missing = set(args.only) - set(recs)
        if missing:
            raise SystemExit(f"unknown recording ids: {sorted(missing)}")
        recs = [r for r in recs if r in set(args.only)]
    if args.limit:
        recs = recs[: args.limit]
    return recs


def cmd_run(args) -> None:
    from tqdm import tqdm

    cfg = load_config(args.config, dict(args.set or []))
    stages = args.stages.split(",")
    unknown = set(stages) - set(pipeline.STAGES)
    if unknown:
        raise SystemExit(f"unknown stages {sorted(unknown)}; choose from {pipeline.STAGES}")

    recs = _select(cfg, args)
    print(f"dataset={cfg.dataset.name} recordings={len(recs)} stages={stages}", flush=True)

    if "segment" in stages:
        total = 0
        for rec in tqdm(recs, desc="segment"):
            total += len(pipeline.stage_segment(cfg, rec, force=args.force)["segments"])
        print(f"  segments: {total:,}", flush=True)

    if "embed" in stages:
        embedder = EcapaEmbedder(cfg.embedding, Path(cfg.root))
        print(f"  embedding device: {embedder.device}", flush=True)
        for rec in tqdm(recs, desc="embed"):
            pipeline.stage_embed(cfg, rec, embedder, force=args.force)

    if "cluster" in stages:
        print(f"  clustering device: {resolve_device(cfg.clustering.device)}", flush=True)
        outer = tqdm(recs, desc="cluster")
        for rec in outer:
            bar = {"inner": None}

            def progress(done, total, bar=bar, rec=rec):
                if bar["inner"] is None:
                    bar["inner"] = tqdm(total=total, desc=f"  {rec} p-sweep", leave=False)
                bar["inner"].update(done - bar["inner"].n)

            result = pipeline.stage_cluster(cfg, rec, force=args.force, progress=progress)
            if bar["inner"] is not None:
                bar["inner"].close()
            outer.set_postfix(k=int(result["k_pred"]), p=int(result["p_star"]))

    if "assign" in stages and cfg.assignment.enabled:
        for rec in tqdm(recs, desc="assign"):
            pipeline.stage_assign(cfg, rec, force=args.force)

    if "evaluate" in stages:
        frame = pipeline.stage_evaluate(cfg, recs)
        path = report.write(cfg, frame)
        print(f"\n{report.build(cfg, frame)}", flush=True)
        print(f"wrote {cfg.results / 'predictions.csv'} and {path}", flush=True)


def cmd_report(args) -> None:
    import pandas as pd

    cfg = load_config(args.config, dict(args.set or []))
    csv = cfg.results / "predictions.csv"
    if not csv.exists():
        raise SystemExit(f"{csv} not found; run the evaluate stage first")
    frame = pd.read_csv(csv)
    print(report.build(cfg, frame))
    print(f"\nwrote {report.write(cfg, frame)}", flush=True)


def cmd_meet(args) -> None:
    from nmesc import meet

    meet.run(args.meet_config, token=args.token)


def cmd_inspect(args) -> None:
    cfg = load_config(args.config, dict(args.set or []))
    rec = args.recording

    payload = pipeline.stage_segment(cfg, rec)
    print(f"=== {rec} ===")
    print(f"duration        : {payload['duration']:.2f} s")
    print(f"reference K     : {payload['k_true']}  ({payload['n_reference_turns']} turns)")
    print(f"oracle speech   : {payload['speech_seconds']:.2f} s "
          f"({100 * payload['speech_seconds'] / payload['duration']:.1f}%)")
    print(f"speech intervals: {len(payload['intervals'])}")
    print(f"segments        : {len(payload['segments'])}")

    emb_path = cfg.work / "embeddings" / f"{rec}.npy"
    if emb_path.exists():
        embeddings = np.load(emb_path)
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"embeddings      : {embeddings.shape} dtype={embeddings.dtype} "
              f"L2 norm in [{norms.min():.4f}, {norms.max():.4f}]")

    npz_path = cfg.work / "clustering" / f"{rec}.npz"
    if not npz_path.exists():
        print("clustering      : not run yet")
        return

    with np.load(npz_path) as data:
        print(f"K̂               : {int(data['k_pred'])}   (true {payload['k_true']})")
        print(f"p*              : {int(data['p_star'])}")
        print(f"mean off-diag cos: {float(data['mean_cosine']):.4f}")
        print(f"clustering time : {float(data['elapsed_sec']):.2f} s")

        p_values, r_values = data["p_values"], data["r_values"]
        if len(p_values):
            best = int(np.argmin(r_values))
            print(f"\nr(p) sweep over p in [{p_values[0]}, {p_values[-1]}] "
                  f"({len(p_values)} points), min at p={p_values[best]}:")
            step = max(1, len(p_values) // args.rows)
            shown = sorted(set(list(range(0, len(p_values), step)) + [best]))
            print(f"  {'p':>6} {'r(p)':>12} {'g_p':>12} {'max gap':>12} {'k@p':>5}")
            for i in shown:
                mark = "  <- p*" if i == best else ""
                print(f"  {p_values[i]:>6} {r_values[i]:>12.4f} {data['g_values'][i]:>12.6f} "
                      f"{data['max_gap_values'][i]:>12.6f} {data['k_values'][i]:>5}{mark}")

        eigenvalues, eigengaps = data["eigenvalues"], data["eigengaps"]
        if len(eigenvalues):
            top = min(len(eigengaps), cfg.clustering.k_max + 3)
            print(f"\nsmallest eigenvalues and gaps at p* (k_max={cfg.clustering.k_max}):")
            print(f"  {'i':>4} {'lambda_i':>14} {'gap_i = l(i+1)-l(i)':>22}")
            for i in range(top):
                mark = "  <- K̂" if i + 1 == int(data["k_pred"]) else ""
                capped = "" if i < cfg.clustering.k_max else "  (beyond k_max)"
                print(f"  {i + 1:>4} {eigenvalues[i]:>14.6f} {eigengaps[i]:>22.6f}{mark}{capped}")

        labels = data["labels"]
        if len(labels):
            uniq, counts = np.unique(labels, return_counts=True)
            print(f"\nsegment labels  : {dict(zip(uniq.tolist(), counts.tolist()))}")

    hyp = cfg.work / "rttm_hyp" / f"{rec}.rttm"
    print(f"hypothesis rttm : {hyp if hyp.exists() else 'not written yet'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--config", required=True)
        p.add_argument("--set", action="append", type=_parse_override, metavar="KEY=VALUE",
                       help="dotted config override, e.g. --set clustering.laplacian=sym_norm")

    run = sub.add_parser("run", help="execute pipeline stages")
    common(run)
    run.add_argument("--stages", default=",".join(pipeline.STAGES))
    run.add_argument("--only", nargs="+", metavar="REC_ID")
    run.add_argument("--limit", type=int)
    run.add_argument("--force", action="store_true", help="recompute even if artifacts exist")
    run.set_defaults(func=cmd_run)

    inspect = sub.add_parser("inspect", help="dump intermediates for one recording")
    common(inspect)
    inspect.add_argument("recording")
    inspect.add_argument("--rows", type=int, default=20, help="rows of the r(p) sweep to print")
    inspect.set_defaults(func=cmd_inspect)

    rep = sub.add_parser("report", help="rebuild summary.md from predictions.csv")
    common(rep)
    rep.set_defaults(func=cmd_report)

    meet = sub.add_parser("meet", help="render results_meet plots and relative-error accuracy")
    meet.add_argument("--meet-config", default="results_meet/config.yaml")
    meet.add_argument("--token", default=None, help="HF token for pyannote/segmentation-3.0")
    meet.set_defaults(func=cmd_meet)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
