"""CondKGCP's search space on validation and on test, and a page to review it.

For every model and score function, runs every setting of the search space of
scripts/conformal.py (CondKGCP and its two ablations, with marginal and Mondrian CP for
reference) twice: on the validation splits of its --select, which mark the selected
settings, and on its first --test-splits test splits. Writes one row per split, score
function, alpha and setting to <out>/<name>-search.csv, with metrics pooled over the
splits and their standard deviation over single splits. A progress bar per model counts
its steps, a setting on a split, and estimates the time left.

Then builds <out>/condkgcp-search.html from all <out>/*-search.csv, a page to open in a
browser. For a dataset, model, score function and alpha, it shows the selection rule
step by step; the Pareto front of the CondKGCP settings (CovGap against conditional set
size); the same settings by min_size and gamma; and a table of the selected settings on
test against Mondrian and marginal CP.

    python scripts/condkgcp_search.py models/libkge/wn18-*.pt models/libkge/fb15k-*.pt
    python scripts/condkgcp_search.py  # only rebuild the page

A model is a single file (seed 0), named <dataset>-<model>.pt.
"""

import argparse
import csv
import functools
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import conformal  # scripts/conformal.py: the search space, splits and selection rule
from tqdm import tqdm

import kgcp

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(__file__).with_suffix(".html")
SEED = 0
TEST_SPLITS = 100  # of scripts/conformal.py; this uses the first --test-splits
METRICS = ["coverage", "size", "cond_size", "covgap", "ucg", "ucr", "singleton", "empty"]
SPREAD = ["coverage", "cond_size", "covgap"]


def pooled(
    splits: list[tuple], rows: list[dict], alpha: float, merged, args, bar: tqdm
) -> list[dict]:
    """The rows with their metrics pooled over the (calibration, evaluation) splits, and
    the standard deviation of the metrics over single splits."""
    out = []
    for row in rows:
        cp = conformal.predictor(row, alpha, merged)
        parts = []
        for calibration, evaluation in splits:
            sets = cp.calibrate(calibration).predict(evaluation)
            # what kgcp.evaluate reads, computed once for pooled and single splits
            parts.append(
                SimpleNamespace(covered=sets.covered, size=sets.size, queries=sets.queries)
            )
            bar.update()
        metrics = kgcp.evaluate(parts, alpha, args.min_queries)
        single = [kgcp.evaluate([p], alpha, args.min_queries) for p in parts]
        spread = {f"{m}_sd": statistics.stdev(s[m] for s in single) for m in SPREAD}
        out.append(row | {"alpha": alpha} | {m: metrics[m] for m in METRICS} | spread)
    return out


def setting(row: dict, selected: bool | None) -> dict:
    """A row with its setting first, then whether it was selected, then its metrics."""
    keys = ("method", "groups", "min_size", "gamma", "alpha")
    rest = {k: v for k, v in row.items() if k not in keys}
    return {k: row[k] for k in keys} | {"selected": selected} | rest


def search(file: Path, args) -> list[dict]:
    """Rows of every setting, score function and alpha of a model, on validation and test."""
    dataset, model_name = file.stem.rsplit("-", 1)
    model = kgcp.LibKGE(file, device=args.device)
    kg = model.kg
    embeddings = model.relation_embeddings()
    merged = functools.cache(lambda size: kgcp.MergedRelations(embeddings, size))
    known = {"valid": kg.known("train"), "test": kg.known("train", "valid")}
    queries = {split: kgcp.Queries.from_triples(kg.split(split)) for split in known}
    calibration_size = round(0.5 * len(queries["test"]))
    space = {  # per alpha, the settings of each method
        alpha: conformal.search_space(
            conformal.min_sizes(queries["valid"], calibration_size, alpha), args.gammas
        )
        for alpha in args.alphas
    }
    # a step is a setting on a split; the two reference methods are settings too
    settings = sum(2 + sum(map(len, methods.values())) for methods in space.values())
    steps = len(conformal.SCORES) * settings * (args.select_splits + args.test_splits)
    bar = tqdm(total=steps, desc=file.stem, unit="step", smoothing=0)
    rows = []
    for score_name, score in conformal.SCORES.items():
        for split in ("valid", "test"):  # one split's scores in memory at a time
            bar.set_postfix_str(f"{score_name} {split}")
            scores = kgcp.nonconformity(
                model,
                queries[split],
                score,
                known[split],
                args.batch_size,
                seed=SEED,
                keep_matrix=False,
            )
            if split == "valid":  # as --select: calibration sets as large as the test halves
                num = args.select_splits
                splits = conformal.random_splits(scores, calibration_size / len(scores), SEED, num)
            else:
                num = args.test_splits
                splits = conformal.random_splits(scores, 0.5, SEED, TEST_SPLITS)[:num]
            key = {
                "dataset": dataset,
                "model": model_name,
                "split": split,
                "splits": num,
                "score": score_name,
            }
            for alpha in args.alphas:
                reference = pooled(splits, conformal.BASELINES[:2], alpha, merged, args, bar)
                rows += [key | setting(row, None) for row in reference]
                for settings in space[alpha].values():
                    tuned = pooled(splits, settings, alpha, merged, args, bar)
                    chosen = None
                    if split == "valid":
                        floor = reference[0]["coverage"]  # marginal CP's
                        chosen = conformal.best(tuned, alpha, floor, args.covgap_tolerance)
                    rows += [
                        key | setting(row, None if chosen is None else row is chosen)
                        for row in tuned
                    ]
            del scores, splits
    bar.close()
    return rows


# ---- the page ----

LABELS = {
    "wn18": "WN18",
    "wnrr": "WN18RR",
    "fb15k": "FB15k",
    "fb15k-237": "FB15k-237",
    "distmult": "DistMult",
    "complex": "ComplEx",
    "rescal": "RESCAL",
    "transe": "TransE",
    "rotate": "RotatE",
    "conve": "ConvE",
}
FAMILIES = {
    ("split", "marginal"): "marginal",
    ("mondrian", "relation"): "mondrian",
    ("condkgcp", "merged"): "condkgcp",
}
COLUMNS = [
    "split",
    "score",
    "alpha",
    "family",
    "min_size",
    "gamma",
    "selected",
    "coverage",
    "size",
    "cond_size",
    "covgap",
]


def number(text: str):
    if text in ("", "None"):
        return None
    if text in ("True", "False"):
        return text == "True"
    value = float(text)
    return int(value) if value.is_integer() and "." not in text else float(f"{value:.6g}")


def page(out: Path, tolerance: float) -> Path | None:
    """Builds <out>/condkgcp-search.html from <out>/*-search.csv; None without any."""
    models = {}
    for path in sorted(out.glob("*-search.csv")):
        with open(path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if (r["method"], r["groups"]) in FAMILIES]
        dataset, model = rows[0]["dataset"], rows[0]["model"]
        grid = {}
        for r in rows:
            r["family"] = FAMILIES[r["method"], r["groups"]]
            if r["family"] == "condkgcp":
                grid.setdefault(r["alpha"], set()).add(int(r["min_size"]))
        models[f"{dataset}-{model}"] = {
            "dataset": dataset,
            "model": model,
            "label": LABELS.get(dataset, dataset),
            "model_label": LABELS.get(model, model),
            "splits": {r["split"]: int(r["splits"]) for r in rows},
            "grid": {str(number(a)): sorted(s) for a, s in grid.items()},
            "rows": [
                [
                    number(r[c]) if c != "family" and c not in ("split", "score") else r[c]
                    for c in COLUMNS
                ]
                for r in rows
            ],
        }
    if not models:
        return None
    gammas = sorted(
        {r[COLUMNS.index("gamma")] for m in models.values() for r in m["rows"]} - {None}
    )
    data = {"cols": COLUMNS, "tolerance": tolerance, "gammas": gammas, "models": models}
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = out / "condkgcp-search.html"
    html.write_text(TEMPLATE.read_text().replace("__DATA__", payload))
    return html


def parser() -> argparse.ArgumentParser:
    """The options; cloud/modal/condkgcp.py runs the search with the same defaults."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("models", nargs="*", type=Path, help="files <dataset>-<model>.pt")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.05, 0.1, 0.2])
    parser.add_argument("--gammas", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--select-splits", type=int, default=10, help="validation splits")
    parser.add_argument("--test-splits", type=int, default=20, help="test splits")
    parser.add_argument("--covgap-tolerance", type=float, default=0.005, help="of --select")
    parser.add_argument("--min-queries", type=int, default=10, help="for relation metrics")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, mps or auto")
    parser.add_argument("--batch-size", type=int, default=1024, help="queries per model call")
    parser.add_argument("--out", type=Path, default=ROOT / "results/conformal/libkge")
    return parser


def main():
    args = parser().parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for file in args.models:
        start = time.time()
        out = args.out / f"{file.stem}-search.csv"
        conformal.write(search(file, args), out)
        print(f"{file.stem}: {time.time() - start:.0f} s, {out}")
    html = page(args.out, args.covgap_tolerance)
    print(html or f"no *-search.csv in {args.out}")


if __name__ == "__main__":
    main()
