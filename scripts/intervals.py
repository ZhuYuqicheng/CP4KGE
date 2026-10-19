"""Conformal prediction intervals for the confidences of triples, with UnKGCP's models of
uncertain knowledge graphs (third_party/unkge).

For every model and residual score, predicts the confidences of the test triples once,
then calibrates on random halves of the test triples and evaluates on the other halves
(--calibration test; each seed gets its own splits, the same for all models). With
--calibration valid, it calibrates on the validation triples and evaluates on all test
triples, as in UnKGCP's experiments. The triples are those of the datasets' test and
validation files, which have no negatives.

Scores: the absolute residual |c - M| (CP, UnKGCP's baseline) and the residual scaled by
the binary entropy of the prediction, |c - M| / H(M) (UnKGCP). Methods: marginal split CP,
as in UnKGCP, and Mondrian CP per relation.

    python scripts/intervals.py  # models/unkgcp/<dataset>-{ukge_logi,passleaf,beurre}/

A model is a folder of seed<N>.pt files, named <dataset>-<model> by the folder. Writes
one row per (seed, score, method, alpha) to <out>/<name>-<calibration>.csv, with metrics
pooled over the splits, and prints a summary at alpha 0.1 (or the first alpha): mean ±
std over seeds, with UnKGCP's Table 1 for marginal CP.
"""

import argparse
import statistics
import time
from pathlib import Path

import conformal  # scripts/conformal.py: model files, random splits, CSV output
from tqdm import tqdm

import kgcp

ROOT = Path(__file__).resolve().parents[1]
MODELS = [
    ROOT / "models" / "unkgcp" / f"{dataset}-{model}"
    for dataset in ("cn15k", "nl27k", "ppi5k")
    for model in ("ukge_logi", "passleaf", "beurre")
]
SCORES = {"absolute": kgcp.AbsoluteResidual(), "entropy": kgcp.EntropyResidual()}
METHODS = [("split", "marginal"), ("mondrian", "relation")]
SUMMARY = ["coverage", "width", "full", "covgap", "ucr", "cond_width"]
# UnKGCP's Table 1 (alpha 0.1, mean of 10 models): (dataset, model) -> score -> coverage, width.
# Its UKGE values are those of the ukge_rect models, and its CN15k UnKGCP values use the
# entropy of 0.5 M + 0.5 (docs/prediction_intervals.md).
TABLE_1 = {
    ("cn15k", "ukge_logi"): {"absolute": (0.90, 0.88), "entropy": (0.90, 0.82)},
    ("ppi5k", "ukge_logi"): {"absolute": (0.90, 0.16), "entropy": (0.90, 0.16)},
    ("nl27k", "ukge_logi"): {"absolute": (0.90, 0.27), "entropy": (0.91, 0.43)},
    ("cn15k", "passleaf"): {"absolute": (0.90, 0.86), "entropy": (0.90, 0.84)},
    ("ppi5k", "passleaf"): {"absolute": (0.90, 0.21), "entropy": (0.90, 0.20)},
    ("nl27k", "passleaf"): {"absolute": (0.90, 0.54), "entropy": (0.90, 0.44)},
    ("cn15k", "beurre"): {"absolute": (0.90, 0.86), "entropy": (0.90, 0.81)},
    ("ppi5k", "beurre"): {"absolute": (0.90, 0.25), "entropy": (0.90, 0.26)},
    ("nl27k", "beurre"): {"absolute": (0.90, 0.42), "entropy": (0.90, 0.38)},
}


def predictor(method: tuple[str, str], alpha: float) -> kgcp.SplitConformal:
    if method == ("split", "marginal"):
        return kgcp.SplitConformal(alpha)
    return kgcp.MondrianConformal(alpha, kgcp.by_relation)


def evaluate_model(file: Path, seed: int, args) -> list[dict]:
    """Rows of a model per score, method and alpha, with metrics pooled over the splits."""
    model = kgcp.LibKGE(file, device=args.device)
    triples = {s: model.kg.uncertain_triples(s) for s in ("valid", "test")}
    rows = []
    for name, score in SCORES.items():
        residuals = {s: kgcp.residuals(model, t, score) for s, t in triples.items()}
        if args.calibration == "test":
            splits = conformal.random_splits(residuals["test"], 0.5, seed, args.splits)
        else:
            splits = [(residuals["valid"], residuals["test"])]
        for alpha in args.alphas:
            for method in METHODS:
                cp = predictor(method, alpha)
                intervals = (cp.calibrate(cal).predict(evaluation) for cal, evaluation in splits)
                metrics = kgcp.evaluate_intervals(intervals, alpha, args.min_triples)
                key = {"score": name, "method": method[0], "groups": method[1], "alpha": alpha}
                rows.append(key | metrics)
    return rows


def summarize(rows: list[dict], alpha: float) -> str:
    """Markdown table at level alpha: mean ± std over seeds, per score and method, with
    UnKGCP's Table 1 (coverage / width) for marginal CP at alpha 0.1."""
    lines = [
        "| score | method | " + " | ".join(SUMMARY) + " | Table 1 |",
        "|---" * (3 + len(SUMMARY)) + "|",
    ]
    seeds = {}
    for r in rows:
        if r["alpha"] == alpha:
            seeds.setdefault((r["score"], r["method"], r["groups"]), []).append(r)
    for (score, method, groups), selected in seeds.items():
        cells = []
        for metric in SUMMARY:
            values = [r[metric] for r in selected]
            cell = f"{statistics.mean(values):.3f}"
            if len(values) > 1:
                cell += f" ± {statistics.stdev(values):.3f}"
            cells.append(cell)
        paper = None
        if method == "split" and alpha == 0.1:
            paper = TABLE_1.get((selected[0]["dataset"], selected[0]["model"]), {}).get(score)
        cells.append(f"{paper[0]:.2f} / {paper[1]:.2f}" if paper else "")
        lines.append(f"| {score} | {method} {groups} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("models", nargs="*", type=Path, default=MODELS, help="model folders")
    parser.add_argument("--seeds", nargs="+", type=int, help="model seeds (default: all)")
    parser.add_argument("--calibration", choices=["test", "valid"], default="test")
    parser.add_argument("--splits", type=int, default=100, help="random calibration splits")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.05, 0.1, 0.2])
    parser.add_argument("--min-triples", type=int, default=10, help="for relation metrics")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, mps or auto")
    parser.add_argument("--out", type=Path, default=ROOT / "results/conformal/unkgcp")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    models = {path: conformal.model_files(path, args.seeds) for path in args.models}
    with tqdm(total=sum(map(len, models.values())), unit="model", smoothing=0) as bar:
        for path, files in models.items():
            start = time.time()
            dataset, name = path.name.split("-", 1)
            rows = []
            for file, seed in files:
                bar.set_postfix_str(f"{path.name} seed {seed}")
                key = {"dataset": dataset, "model": name, "seed": seed}
                key["calibration"] = args.calibration
                rows += [key | row for row in evaluate_model(file, seed, args)]
                bar.update()
            out = args.out / f"{path.name}-{args.calibration}.csv"
            conformal.write(rows, out)
            tqdm.write(f"## {path.name} ({time.time() - start:.0f} s, {out.name})\n")
            alpha = 0.1 if 0.1 in args.alphas else args.alphas[0]
            tqdm.write(f"alpha {alpha}:\n{summarize(rows, alpha)}\n")


if __name__ == "__main__":
    main()
