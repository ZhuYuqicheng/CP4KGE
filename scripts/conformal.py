"""Conformal prediction experiments with trained LibKGE models.

For every model and score function, scores all test queries once, then calibrates on
random halves of the test queries and evaluates on the other halves (--calibration test;
each seed gets its own splits, the same for all models). With --calibration valid, it
calibrates on the validation split and evaluates on the whole test split, as in earlier
work. Known answers are filtered from splits outside the calibration and evaluation data:
train + valid, or train only. Each model runs once per split of the data; the score
functions run on its device.

    python scripts/conformal.py  # the tuned models, models/<dataset>-<model>/seed<N>.pt
    python scripts/conformal.py models/libkge/*.pt --out results/conformal/libkge

A model is a folder of seed<N>.pt files, named by the folder, or a single file (seed 0),
named by the file; names are <dataset>-<model>. Writes one row per (seed, score, grouping,
alpha) to <out>/<name>-<calibration>.csv, with metrics pooled over the splits, and prints
a summary (mean ± std over seeds).
"""

import argparse
import csv
import statistics
import time
from pathlib import Path

import torch
from tqdm import tqdm

import kgcp

ROOT = Path(__file__).resolve().parents[1]
TUNED = [
    ROOT / "models" / f"{dataset}-{model}"
    for dataset in ("nations", "umls", "kinship")
    for model in ("distmult", "complex", "rescal")
]
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


def model_files(path: Path, seeds: list[int] | None) -> list[tuple[Path, int]]:
    """(file, seed) of a model: the seed<N>.pt files of a folder, or a single file."""
    if path.is_file():
        return [(path, 0)]
    files = sorted(path.glob("seed*.pt"))
    if not files:
        raise FileNotFoundError(f"no models in {path} (see README.md#trained-models)")
    return [(f, int(f.stem[4:])) for f in files if seeds is None or int(f.stem[4:]) in seeds]


@torch.no_grad()
def logits(model: kgcp.LibKGE, queries: kgcp.Queries, batch_size: int) -> torch.Tensor:
    """[n, E] logits of the queries, on the model's device."""
    out = torch.empty(len(queries), model.kg.num_entities, device=model.device)
    for start in range(0, len(queries), batch_size):
        out[start : start + batch_size] = model.score(queries[start : start + batch_size])
    return out


def evaluate(score_split, score, seed: int, args) -> list[dict]:
    """Rows of one score function: per alpha and grouping, metrics pooled over the splits.

    `score_split(split, score)` scores the queries of a split. `seed` makes splits differ
    between models.
    """
    if args.calibration == "test":
        scores = score_split("test", score)
        seeds = range(seed * args.splits, (seed + 1) * args.splits)
        splits = [scores.split(0.5, torch.Generator().manual_seed(s)) for s in seeds]
    else:
        splits = [(score_split("valid", score), score_split("test", score))]
    rows = []
    for alpha in args.alphas:
        for groups_name, groups in GROUPS.items():
            cp = kgcp.SplitConformal(alpha, groups)
            sets = (cp.calibrate(cal).predict(test) for cal, test in splits)
            metrics = kgcp.evaluate(sets, alpha, args.min_queries)
            rows.append({"groups": groups_name, "alpha": alpha} | metrics)
    return rows


def evaluate_model(file: Path, seed: int, args, bar: tqdm) -> list[dict]:
    """Rows of a trained model: per score function, alpha and grouping."""
    model = kgcp.LibKGE(file, device=args.device)
    kg = model.kg
    if args.calibration == "test":
        known, splits = kg.known("train", "valid"), ["test"]
    else:
        known, splits = kg.known("train"), ["valid", "test"]
    data = {}  # the model runs once per split
    for split in splits:
        queries = kgcp.Queries.from_triples(kg.split(split))
        data[split] = queries, logits(model, queries, args.batch_size)

    def score_split(split, score):
        queries, split_logits = data[split]
        return kgcp.nonconformity(split_logits, queries, score, known, seed=seed)

    rows = []
    for name, score in SCORES.items():
        rows += [{"score": name} | row for row in evaluate(score_split, score, seed, args)]
        bar.update()
    return rows


def run(path: Path, files: list[tuple[Path, int]], args, bar: tqdm) -> list[dict]:
    name = path.stem if path.is_file() else path.name
    dataset, model = name.rsplit("-", 1)
    rows = []
    for file, seed in files:
        bar.set_postfix_str(f"{name} seed {seed}")
        key = {"dataset": dataset, "model": model, "seed": seed, "calibration": args.calibration}
        rows += [key | row for row in evaluate_model(file, seed, args, bar)]
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
            cell = f"{statistics.mean(values):.3f}"
            if len(values) > 1:
                cell += f" ± {statistics.stdev(values):.3f}"
            cells.append(cell)
        lines.append(f"| {alpha} | {groups} | {score} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("models", nargs="*", type=Path, default=TUNED, help="folders or files")
    parser.add_argument("--seeds", nargs="+", type=int, help="model seeds (default: all)")
    parser.add_argument("--calibration", choices=["test", "valid"], default="test")
    parser.add_argument("--splits", type=int, default=100, help="random calibration splits")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.05, 0.1, 0.2])
    parser.add_argument("--min-queries", type=int, default=10, help="for relation metrics")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, mps or auto")
    parser.add_argument("--batch-size", type=int, default=1024, help="queries per model call")
    parser.add_argument("--out", type=Path, default=ROOT / "results/conformal")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    models = {path: model_files(path, args.seeds) for path in args.models}
    total = len(SCORES) * sum(map(len, models.values()))
    with tqdm(total=total, unit="score", smoothing=0) as bar:  # per model and score function
        for path, files in models.items():
            start = time.time()
            rows = run(path, files, args, bar)
            name = f"{rows[0]['dataset']}-{rows[0]['model']}"
            out = args.out / f"{name}-{args.calibration}.csv"
            with open(out, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            tqdm.write(f"## {name} ({time.time() - start:.0f} s, {out})\n")
            tqdm.write(summarize(rows) + "\n")


if __name__ == "__main__":
    main()
