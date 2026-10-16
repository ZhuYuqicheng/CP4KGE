"""Conformal prediction experiments with trained LibKGE models.

For every model and score function, scores all test queries once, then calibrates on
random halves of the test queries and evaluates on the other halves (--calibration test;
each seed gets its own splits, the same for all models). With --calibration valid, it
calibrates on the validation split and evaluates on the whole test split, as in earlier
work. Known answers are filtered from splits outside the calibration and evaluation data:
train + valid, or train only. Each model runs once per split of the data (with
--low-memory, once per split and score function, without keeping its logits); the score
functions run on its device.

Methods: marginal split CP; Mondrian CP per relation and per direction; CondKGCP, and its
ablations without merging relations (per relation) and without rank calibration (Mondrian
CP over the merged relations), for every setting of the search space:

- gamma, the share of the top-k miscoverage that the score threshold gives back: --gammas
- the minimum group size of merged relations (--min-sizes, or adapted to the data): 8
  sizes, evenly spaced on a log scale, from the smallest group that can estimate a top-k
  miscoverage below alpha, floor(1 / alpha) + 1 calibration queries, to the expected
  calibration queries of the most frequent relation, beyond which all relations form one
  group. The counts are those of the queries that calibration sets are drawn from.

With --select, each of these three methods gets its setting per score function and alpha
from --select-splits random splits of the validation queries (filtered with train only),
each calibrating on as many queries as the test calibration halves. Of the settings that
cover at least 1 - alpha, or at least as much as marginal CP on the same splits (which
shares their noise), those within --covgap-tolerance of the smallest CovGap; of these,
the one with the smallest mean set size per relation (cond_size). If none covers enough,
the one that covers the most. Only the selected settings run on test.

    python scripts/conformal.py  # the tuned models, models/<dataset>-<model>/seed<N>.pt
    python scripts/conformal.py models/libkge/*.pt --out results/conformal/libkge

A model is a folder of seed<N>.pt files, named by the folder, or a single file (seed 0),
named by the file; names are <dataset>-<model>. Writes one row per (seed, score, method,
alpha) to <out>/<name>-<calibration>.csv, with metrics pooled over the splits, the
validation rows of --select to <out>/<name>-selection.csv, and prints a summary at alpha
0.1 (or the first alpha): mean ± std over seeds.
"""

import argparse
import csv
import functools
import math
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
GROUPINGS = {"relation": kgcp.by_relation, "direction": kgcp.by_direction}
NUM_MIN_SIZES = 8
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


def key(method: str, groups: str, min_size: int | None = None, gamma: float | None = None):
    """A method and its setting: the row key of its results."""
    return {"method": method, "groups": groups, "min_size": min_size, "gamma": gamma}


BASELINES = [key("split", "marginal"), key("mondrian", "relation"), key("mondrian", "direction")]


def search_space(min_sizes: list[int], gammas: list[float]) -> dict[str, list[dict]]:
    """The settings of each method that has some."""
    return {
        "condkgcp": [key("condkgcp", "merged", s, g) for s in min_sizes for g in gammas],
        "without merging": [key("condkgcp", "relation", gamma=g) for g in gammas],
        "without rank calibration": [key("mondrian", "merged", s) for s in min_sizes],
    }


def min_sizes(pool: kgcp.Queries, calibration_size: int, alpha: float) -> list[int]:
    """Minimum group sizes adapted to the data, for calibration sets of `calibration_size`
    queries drawn from `pool` (see the module docstring)."""
    smallest = math.floor(1 / alpha + 1e-9) + 1
    largest = torch.bincount(pool.relation).max().item() * calibration_size / len(pool)
    if largest <= smallest:
        return [smallest]
    sizes = torch.logspace(math.log10(smallest), math.log10(largest), NUM_MIN_SIZES)
    return sorted(set(sizes.round().long().tolist()))


def predictor(row: dict, alpha: float, merged):
    """The conformal predictor of a row key; `merged(min_size)` gives MergedRelations."""
    if row["method"] == "split":
        return kgcp.SplitConformal(alpha)
    groups = merged(row["min_size"]) if row["groups"] == "merged" else GROUPINGS[row["groups"]]
    if row["method"] == "mondrian":
        return kgcp.MondrianConformal(alpha, groups)
    return kgcp.CondKGCP(alpha, groups, row["gamma"])


def random_splits(scores: kgcp.Scores, fraction: float, seed: int, num: int) -> list[tuple]:
    """`num` random splits of the queries; `seed` makes them differ between models."""
    seeds = range(seed * num, (seed + 1) * num)
    return [scores.split(fraction, torch.Generator().manual_seed(s)) for s in seeds]


def pooled(splits: list[tuple], rows: list[dict], alpha: float, merged, args) -> list[dict]:
    """The rows with their metrics, pooled over the (calibration, evaluation) splits."""
    out = []
    for row in rows:
        cp = predictor(row, alpha, merged)
        sets = (cp.calibrate(cal).predict(test) for cal, test in splits)
        out.append(row | {"alpha": alpha} | kgcp.evaluate(sets, alpha, args.min_queries))
    return out


def best(rows: list[dict], alpha: float, reference: float, tolerance: float) -> dict:
    """Of the rows that cover at least 1 - alpha or `reference` (marginal CP's coverage),
    those within `tolerance` of the smallest CovGap; of these, the one with the smallest
    sets per relation. If none covers enough, the one that covers the most."""
    floor = min(1 - alpha, reference) - 1e-12  # tolerance: rounding
    covering = [r for r in rows if r["coverage"] >= floor]
    if not covering:
        return max(rows, key=lambda r: r["coverage"])
    lowest = min(r["covgap"] for r in covering)
    near = [r for r in covering if r["covgap"] <= lowest + tolerance + 1e-12]
    return min(near, key=lambda r: (r["cond_size"], r["size"]))


def select(valid: kgcp.Scores, calibration_size: int, seed: int, merged, args) -> list[dict]:
    """Rows of every setting on random splits of the validation queries, with calibration
    sets of `calibration_size`, and of marginal CP (`selected` empty) as a reference;
    `selected` marks the best of each method and alpha."""
    splits = random_splits(valid, calibration_size / len(valid), seed, args.select_splits)
    rows = []
    for alpha in args.alphas:
        reference = pooled(splits, BASELINES[:1], alpha, merged, args)[0]
        rows.append(reference | {"selected": None})
        sizes = args.min_sizes or min_sizes(valid.queries, calibration_size, alpha)
        for settings in search_space(sizes, args.gammas).values():
            tuned = pooled(splits, settings, alpha, merged, args)
            chosen = best(tuned, alpha, reference["coverage"], args.covgap_tolerance)
            rows += [r | {"selected": r is chosen} for r in tuned]
    return rows


def evaluate(score_split, score, seed: int, merged, selected: dict, args) -> list[dict]:
    """Rows of one score function: per alpha and method, metrics pooled over the splits.

    `score_split(split, score)` scores the queries of a split. `seed` makes splits differ
    between models. `selected` maps alpha to the settings selected on the validation
    queries; without, all settings run.
    """
    if args.calibration == "test":
        pool = score_split("test", score)
        splits = random_splits(pool, 0.5, seed, args.splits)
    else:
        pool = score_split("valid", score)
        splits = [(pool, score_split("test", score))]
    rows = []
    for alpha in args.alphas:
        if selected:
            settings = selected[alpha]
        else:
            sizes = args.min_sizes or min_sizes(pool.queries, len(splits[0][0]), alpha)
            settings = sum(search_space(sizes, args.gammas).values(), [])
        rows += pooled(splits, BASELINES + settings, alpha, merged, args)
    return rows


def evaluate_model(file: Path, seed: int, args, bar: tqdm) -> tuple[list[dict], list[dict]]:
    """Rows of a trained model per score function, alpha and method; and with --select,
    the rows of every setting on the validation queries."""
    model = kgcp.LibKGE(file, device=args.device)
    kg = model.kg
    embeddings = model.relation_embeddings()
    merged = functools.cache(lambda size: kgcp.MergedRelations(embeddings, size))
    if args.calibration == "test":
        known = {"test": kg.known("train", "valid"), "valid": kg.known("train")}
        splits = ["test", "valid"] if args.select else ["test"]
    else:
        known = {"valid": kg.known("train"), "test": kg.known("train")}
        splits = ["valid", "test"]
    data = {}  # the model runs once per split, unless --low-memory
    for split in splits:
        queries = kgcp.Queries.from_triples(kg.split(split))
        data[split] = queries, None if args.low_memory else logits(model, queries, args.batch_size)

    def score_split(split, score):
        queries, split_logits = data[split]
        return kgcp.nonconformity(
            model if split_logits is None else split_logits,
            queries,
            score,
            known[split],
            args.batch_size,
            seed=seed,
            keep_matrix=False,  # coverage and set sizes need only the sorted scores
        )

    rows, selection = [], []
    for name, score in SCORES.items():
        selected = {}
        if args.select:  # the validation scores are freed before the test scores are made
            calibration_size = round(0.5 * len(data["test"][0]))
            tuned = select(score_split("valid", score), calibration_size, seed, merged, args)
            selection += [{"score": name} | row for row in tuned]
            for row in tuned:
                if row["selected"] is True:
                    setting = {k: row[k] for k in ("method", "groups", "min_size", "gamma")}
                    selected.setdefault(row["alpha"], []).append(setting)
        rows += [
            {"score": name} | row
            for row in evaluate(score_split, score, seed, merged, selected, args)
        ]
        bar.update()
    return rows, selection


def run(path: Path, files: list[tuple[Path, int]], args, bar: tqdm):
    """Rows of a model on test, and with --select on validation, for all its seeds."""
    name = path.stem if path.is_file() else path.name
    dataset, model = name.rsplit("-", 1)
    rows, selection = [], []
    for file, seed in files:
        bar.set_postfix_str(f"{name} seed {seed}")
        key = {"dataset": dataset, "model": model, "seed": seed, "calibration": args.calibration}
        test, valid = evaluate_model(file, seed, args, bar)
        rows += [key | row for row in test]
        selection += [key | row for row in valid]
    return rows, selection


def write(rows: list[dict], out: Path) -> None:
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def label(row: dict) -> str:
    """The method of a row, such as "condkgcp merged min_size=50 gamma=0.25"."""
    options = [f"{name}={row[name]}" for name in ("min_size", "gamma") if row[name] is not None]
    return " ".join([row["method"], row["groups"], *options])


def summarize(rows: list[dict], alpha: float, select: bool) -> str:
    """Markdown table at level alpha: mean ± std over seeds, per score and method. With
    --select, the settings of a method may differ between seeds."""
    lines = [
        "| score | method | " + " | ".join(SUMMARY) + " |",
        "|---" * (2 + len(SUMMARY)) + "|",
    ]
    seeds = {}
    for r in rows:
        if r["alpha"] == alpha:
            method = " ".join([r["method"], r["groups"]]) if select else label(r)
            seeds.setdefault((r["score"], method), []).append(r)
    for (score, method), selected in seeds.items():
        cells = []
        for metric in SUMMARY:
            values = [r[metric] for r in selected]
            cell = f"{statistics.mean(values):.3f}"
            if len(values) > 1:
                cell += f" ± {statistics.stdev(values):.3f}"
            cells.append(cell)
        lines.append(f"| {score} | {method} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("models", nargs="*", type=Path, default=TUNED, help="folders or files")
    parser.add_argument("--seeds", nargs="+", type=int, help="model seeds (default: all)")
    parser.add_argument("--calibration", choices=["test", "valid"], default="test")
    parser.add_argument("--splits", type=int, default=100, help="random calibration splits")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.05, 0.1, 0.2])
    parser.add_argument("--min-queries", type=int, default=10, help="for relation metrics")
    parser.add_argument(
        "--gammas", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0], help="CondKGCP"
    )
    parser.add_argument(
        "--min-sizes", nargs="+", type=int, help="of merged relations (default: adapted)"
    )
    parser.add_argument("--select", action="store_true", help="select the settings on validation")
    parser.add_argument(
        "--select-splits", type=int, default=10, help="random validation splits for --select"
    )
    parser.add_argument(
        "--covgap-tolerance", type=float, default=0.005, help="of --select, for smaller sets"
    )
    parser.add_argument(
        "--low-memory", action="store_true", help="score per score function, keep no logits"
    )
    parser.add_argument("--device", default="cpu", help="cpu, cuda, mps or auto")
    parser.add_argument("--batch-size", type=int, default=1024, help="queries per model call")
    parser.add_argument("--out", type=Path, default=ROOT / "results/conformal")
    args = parser.parse_args()
    if args.select and args.calibration == "valid":
        parser.error("--select needs --calibration test: it tunes on the validation queries")

    args.out.mkdir(parents=True, exist_ok=True)
    models = {path: model_files(path, args.seeds) for path in args.models}
    total = len(SCORES) * sum(map(len, models.values()))
    with tqdm(total=total, unit="score", smoothing=0) as bar:  # per model and score function
        for path, files in models.items():
            start = time.time()
            rows, selection = run(path, files, args, bar)
            name = f"{rows[0]['dataset']}-{rows[0]['model']}"
            out = args.out / f"{name}-{args.calibration}.csv"
            write(rows, out)
            if selection:
                write(selection, args.out / f"{name}-selection.csv")
            tqdm.write(f"## {name} ({time.time() - start:.0f} s, {out})\n")
            alpha = 0.1 if 0.1 in args.alphas else args.alphas[0]
            tqdm.write(f"alpha {alpha}:\n{summarize(rows, alpha, args.select)}\n")


if __name__ == "__main__":
    main()
