"""Conformal prediction experiments with the tuned models of scripts/tune.py.

For every trained model (dataset x model x seed) and score function, scores all test
queries once, then calibrates on random halves of the test queries and evaluates on the
other halves (--calibration test; each seed gets its own splits, the same for all models).
With --calibration valid, it calibrates on the validation split and evaluates on the whole
test split, as in earlier work. Known answers are filtered from splits outside the
calibration and evaluation data: train + valid, or train only.

    python scripts/conformal.py --datasets nations umls kinship --models distmult complex rescal

Writes one row per (seed, score, grouping, alpha) to
results/conformal/<dataset>-<model>-<calibration>.csv, with metrics pooled over the
splits, and prints a summary (mean ± std over seeds).
"""

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import torch

import kgcp

ROOT = Path(__file__).resolve().parents[1]
SCORES = {
    "neg": kgcp.Neg(),
    "minmax": kgcp.MinMax(),
    "softmax": kgcp.Softmax(),
    "aps": kgcp.APS(),
    "raps": kgcp.RAPS(),
    "rank": kgcp.Rank(),
}
GROUPS = {"marginal": kgcp.marginal, "relation": kgcp.by_relation, "direction": kgcp.by_direction}
SUMMARY = ["coverage", "size", "singleton", "covgap", "ucr", "cond_size"]


def model_folders(dataset: str, model: str, seeds: list[int] | None) -> list[Path]:
    """The trained models of the selected configuration, one per seed."""
    result = json.loads((ROOT / "results/hpo" / f"{dataset}-{model}.json").read_text())
    folder = ROOT / "runs/hpo/final" / f"{dataset}-{model}" / f"top{result['selected']}"
    folders = sorted(folder.glob("seed*"))
    return [f for f in folders if seeds is None or int(f.name[4:]) in seeds]


def calibration_splits(model, score, calibration: str, num_splits: int, seed: int) -> list:
    """(calibration, evaluation) scores. `seed` makes splits differ between models."""
    kg = model.kg
    test = kgcp.Queries.from_triples(kg.split("test"))
    if calibration == "test":
        known = kg.known("train", "valid")
        scores = kgcp.nonconformity(model, test, score, known, seed=seed)
        seeds = range(seed * num_splits, (seed + 1) * num_splits)
        return [scores.split(0.5, torch.Generator().manual_seed(s)) for s in seeds]
    valid = kgcp.Queries.from_triples(kg.split("valid"))
    known = kg.known("train")
    return [
        (
            kgcp.nonconformity(model, valid, score, known, seed=seed),
            kgcp.nonconformity(model, test, score, known, seed=seed),
        )
    ]


def run(dataset: str, model_name: str, args) -> list[dict]:
    rows = []
    for folder in model_folders(dataset, model_name, args.seeds):
        model, seed = kgcp.LibKGE(folder, device=args.device), int(folder.name[4:])
        for score_name, score in SCORES.items():
            splits = calibration_splits(model, score, args.calibration, args.splits, seed)
            for alpha in args.alphas:
                for groups_name, groups in GROUPS.items():
                    cp = kgcp.SplitConformal(alpha, groups)
                    sets = (cp.calibrate(cal).predict(test) for cal, test in splits)
                    rows.append(
                        {
                            "dataset": dataset,
                            "model": model_name,
                            "seed": seed,
                            "calibration": args.calibration,
                            "score": score_name,
                            "groups": groups_name,
                            "alpha": alpha,
                            **kgcp.evaluate(sets, alpha, args.min_queries),
                        }
                    )
    return rows


def summarize(rows: list[dict]) -> str:
    """Markdown table: mean ± std over seeds, per alpha, grouping and score."""
    lines = [
        "| alpha | groups | score | " + " | ".join(SUMMARY) + " |",
        "|---" * (3 + len(SUMMARY)) + "|",
    ]
    seeds = {}
    for r in rows:
        seeds.setdefault((r["alpha"], r["groups"], r["score"]), []).append(r)
    for (alpha, groups, score), selected in seeds.items():
        cells = []
        for metric in SUMMARY:
            values = [r[metric] for r in selected]
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            cells.append(f"{statistics.mean(values):.3f} ± {std:.3f}")
        lines.append(f"| {alpha} | {groups} | {score} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--datasets", nargs="+", default=["nations", "umls", "kinship"])
    parser.add_argument("--models", nargs="+", default=["distmult", "complex", "rescal"])
    parser.add_argument("--seeds", nargs="+", type=int, help="model seeds (default: all)")
    parser.add_argument("--calibration", choices=["test", "valid"], default="test")
    parser.add_argument("--splits", type=int, default=100, help="random calibration splits")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.05, 0.1, 0.2])
    parser.add_argument("--min-queries", type=int, default=10, help="for relation metrics")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, mps or auto")
    parser.add_argument("--out", type=Path, default=ROOT / "results/conformal")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        for model_name in args.models:
            start = time.time()
            rows = run(dataset, model_name, args)
            path = args.out / f"{dataset}-{model_name}-{args.calibration}.csv"
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(f"## {dataset} {model_name} ({time.time() - start:.0f} s, {path})\n")
            print(summarize(rows) + "\n", flush=True)


if __name__ == "__main__":
    main()
