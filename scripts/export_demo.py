"""Export the model outputs that the demo page (docs/demo/) calibrates in the browser.

    uv run scripts/export_demo.py

The page runs no model: it loads what the models predicted and repeats kgcp's calibration
in JavaScript. This script writes those predictions to docs/demo/data/, one file per
model:

- answer sets (sets-<dataset>-<model>.js): the logits of all candidates of every test
  query of one tuned model, as 16-bit floats, with the known answers of train and valid
  filtered out; and the L1 distances of the model's relation embeddings (MergedRelations)
- intervals (intervals-<dataset>-<model>.js): the predicted and the true confidence of a
  sample of the test triples of one of UnKGCP's models

Each file also holds `reference`: the metrics that kgcp computes from exactly these
numbers for fixed calibration splits (the even rows calibrate, the odd rows are tested).
The page recomputes them to check its JavaScript against kgcp. index.js lists the files
and adds the benchmark results that the page shows without recomputing them.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
from pathlib import Path

import torch

import kgcp

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "third_party" / "libkge" / "data"
OUT = ROOT / "docs" / "demo" / "data"

# (dataset, models): one seed of each tuned model; Kinship's entities are numbered
# persons, so one model is enough to show a graph with many queries per relation
SETS = [
    ("nations", ["distmult", "complex", "rescal"]),
    ("umls", ["distmult", "complex", "rescal"]),
    ("kinship", ["complex"]),
]
SETS_SEED = 1
INTERVALS = [(d, ["ukge_logi", "passleaf", "beurre"]) for d in ("nl27k", "cn15k", "ppi5k")]
INTERVALS_SEED = 0
SAMPLE = 6000  # test triples per uncertain graph: half calibrate, half are tested
EXAMPLES = 400  # of them, with the names of their entities

SCORES = {
    "neg": kgcp.Neg(),
    "minmax": kgcp.MinMax(),
    "softmax": kgcp.Softmax(),
    "aps": kgcp.APS(randomized=False),  # the page draws its own u for the randomized scores
    "raps": kgcp.RAPS(randomized=False),
    "rank": kgcp.Rank(),
}
RESIDUALS = {"absolute": kgcp.AbsoluteResidual(), "entropy": kgcp.EntropyResidual()}
MIN_SIZE, GAMMA = 20, 0.5  # CondKGCP's settings of the reference results


def encode(tensor: torch.Tensor, dtype: torch.dtype) -> str:
    """Little-endian values of the tensor, base64."""
    return base64.b64encode(tensor.to(dtype).contiguous().numpy().tobytes()).decode()


def names(dataset: str, kind: str, ids: list[str]) -> list[str]:
    """The strings of the ids, for the datasets whose ids are numbers."""
    file = DATA / dataset / f"{kind}_strings.del"
    if not file.exists():
        return ids
    return [line.rstrip("\n").split("\t", 1)[1] for line in file.open()]


def layout(distances: torch.Tensor) -> list[list[float]]:
    """[R, 2] positions in the unit square whose distances approximate the given ones
    (classical multidimensional scaling), to draw the relations."""
    squared = distances.double() ** 2
    n = len(squared)
    centering = torch.eye(n, dtype=torch.double) - 1 / n
    values, vectors = torch.linalg.eigh(-centering @ squared @ centering / 2)
    xy = vectors[:, -2:].flip(1) * values[-2:].flip(0).clamp_min(0).sqrt()
    xy = (xy - xy.amin(0)) / (xy.amax(0) - xy.amin(0)).clamp_min(1e-12)
    return [[round(v, 4) for v in row] for row in xy.tolist()]


def rounded(metrics: dict[str, float]) -> dict[str, float]:
    return {name: round(value, 9) for name, value in metrics.items()}


def export_sets(dataset: str, model_name: str) -> dict:
    model = kgcp.LibKGE(ROOT / "models" / f"{dataset}-{model_name}" / f"seed{SETS_SEED}.pt", "cpu")
    kg = model.kg
    queries = kgcp.Queries.from_triples(kg.split("test"))
    known = kg.known("train", "valid")
    logits = model.score(queries).float().masked_fill(known(queries), -torch.inf)
    logits = logits.half()  # what the page gets: the reference below starts from it too
    distances = kgcp.MergedRelations(model.relation_embeddings(), MIN_SIZE).distances

    calibration = torch.arange(len(queries)) % 2 == 0
    reference = []
    for score_name, score in SCORES.items():
        scores = kgcp.nonconformity(logits.float(), queries, score, known=known)
        cal, test = scores[calibration], scores[~calibration]
        merged = kgcp.MergedRelations(model.relation_embeddings(), MIN_SIZE)
        predictors = {
            "split": kgcp.SplitConformal,
            "mondrian-relation": lambda a: kgcp.MondrianConformal(a, kgcp.by_relation),
            "mondrian-direction": lambda a: kgcp.MondrianConformal(a, kgcp.by_direction),
            "mondrian-merged": lambda a, g=merged: kgcp.MondrianConformal(a, g),
            "condkgcp-merged": lambda a, g=merged: kgcp.CondKGCP(a, g, GAMMA),
            "condkgcp-relation": lambda a: kgcp.CondKGCP(a, kgcp.by_relation, 1.0),
        }
        for alpha in (0.1, 0.2) if score_name == "softmax" else (0.1,):
            for method, predictor in predictors.items():
                sets = predictor(alpha).calibrate(cal).predict(test)
                metrics = kgcp.evaluate(sets, alpha)
                reference.append(
                    {"score": score_name, "method": method, "alpha": alpha} | rounded(metrics)
                )

    recorded = json.loads((ROOT / "results" / "hpo" / f"{dataset}-{model_name}.json").read_text())
    test_metrics = recorded["test"][str(SETS_SEED)]
    return {
        "kind": "sets",
        "dataset": dataset,
        "model": model_name,
        "seed": SETS_SEED,
        "mrr": round(test_metrics["mean_reciprocal_rank_filtered"], 4),
        "hits1": round(test_metrics["hits_at_1_filtered"], 4),
        "entities": kg.entities,
        "relations": kg.relations,
        "anchor": encode(queries.anchor, torch.int16),
        "relation": encode(queries.relation, torch.int16),
        "target": encode(queries.target, torch.int16),
        "direction": encode(queries.direction, torch.uint8),
        "logits": encode(logits, torch.float16),
        "distances": encode(distances, torch.float32),
        "layout": layout(distances),
        "referenceSettings": {"minSize": MIN_SIZE, "gamma": GAMMA},
        "reference": reference,
    }


def export_intervals(dataset: str, model_name: str) -> dict:
    path = ROOT / "models" / "unkgcp" / f"{dataset}-{model_name}" / f"seed{INTERVALS_SEED}.pt"
    model = kgcp.LibKGE(path, "cpu")
    kg = model.kg
    triples = kg.uncertain_triples("test")
    predicted = kgcp.residuals(model, triples, RESIDUALS["absolute"]).predicted
    errors = (predicted - triples.confidence).abs()

    # the same triples for every model of the dataset
    rows = torch.randperm(len(triples), generator=torch.Generator().manual_seed(0))[:SAMPLE]
    triples = triples[rows.sort().values]
    predicted = predicted[rows.sort().values].float()  # what the page gets
    confidence = triples.confidence.float()
    exact = kgcp.UncertainTriples(
        triples.head, triples.relation, triples.tail, confidence.double(), triples.triple
    )

    calibration = torch.arange(len(exact)) % 2 == 0
    reference = []
    for score_name, score in RESIDUALS.items():
        residuals = kgcp.residuals(predicted.double(), exact, score)
        cal, test = residuals[calibration], residuals[~calibration]
        for alpha in (0.05, 0.1, 0.2):
            for method, predictor in (
                ("split", kgcp.SplitConformal(alpha)),
                ("mondrian-relation", kgcp.MondrianConformal(alpha, kgcp.by_relation)),
            ):
                intervals = predictor.calibrate(cal).predict(test)
                metrics = kgcp.evaluate_intervals(intervals, alpha)
                reference.append(
                    {"score": score_name, "method": method, "alpha": alpha} | rounded(metrics)
                )

    entities = names(dataset, "entity", kg.entities)
    shown = torch.randperm(len(exact), generator=torch.Generator().manual_seed(1))[:EXAMPLES]
    examples = [
        [i, entities[exact.head[i]], entities[exact.tail[i]]] for i in shown.sort().values.tolist()
    ]
    return {
        "kind": "intervals",
        "dataset": dataset,
        "model": model_name,
        "seed": INTERVALS_SEED,
        "testTriples": len(errors),
        "mae": round(errors.mean().item(), 4),
        "relations": names(dataset, "relation", kg.relations),
        "relation": encode(exact.relation, torch.int16),
        "predicted": encode(predicted, torch.float32),
        "confidence": encode(confidence, torch.float32),
        "examples": examples,  # [row, head, tail] of some triples
        "reference": reference,
    }


def benchmark() -> dict:
    """Results that the page shows as they are in results/: the 24 LibKGE models at
    alpha 0.1 with the Softmax score (test MRR and marginal set size), and CondKGCP with
    the settings selected on validation against marginal and Mondrian CP on WN18 and FB15k."""
    folder = ROOT / "results" / "conformal" / "libkge"
    mrr = {}
    for line in (ROOT / "results" / "libkge" / "test.jsonl").read_text().splitlines():
        record = json.loads(line)
        mrr[record["model"]] = record["mrr"]

    models = []
    for file in sorted(folder.glob("*-test.csv")):
        for row in csv.DictReader(file.open()):
            if (row["score"], row["groups"], row["alpha"]) == ("softmax", "marginal", "0.1"):
                name = f"{row['dataset']}-{row['model']}"
                models.append(
                    {
                        "dataset": row["dataset"],
                        "model": row["model"],
                        "mrr": round(mrr[name], 4),
                        "size": round(float(row["size"]), 2),
                        "coverage": round(float(row["coverage"]), 4),
                    }
                )

    conditional = []
    for file in sorted(folder.glob("*-search.csv")):
        rows = [r for r in csv.DictReader(file.open()) if r["score"] == "softmax"]
        rows = [r for r in rows if r["alpha"] == "0.1"]
        picked = {
            (r["min_size"], r["gamma"])
            for r in rows
            if r["split"] == "valid" and r["selected"] == "True"
            if (r["method"], r["groups"]) == ("condkgcp", "merged")
        }
        entry = {"dataset": rows[0]["dataset"], "model": rows[0]["model"]}
        for r in (r for r in rows if r["split"] == "test"):
            key = (r["method"], r["groups"])
            if key == ("condkgcp", "merged") and (r["min_size"], r["gamma"]) in picked:
                method = "condkgcp"
                entry["minSize"], entry["gamma"] = int(float(r["min_size"])), float(r["gamma"])
            elif key in (("split", "marginal"), ("mondrian", "relation")):
                method = key[0]
            else:
                continue
            entry[method] = {
                "coverage": round(float(r["coverage"]), 4),
                "covgap": round(float(r["covgap"]), 4),
                "condSize": round(float(r["cond_size"]), 2),
            }
        conditional.append(entry)
    return {"alpha": 0.1, "score": "softmax", "models": models, "conditional": conditional}


def write(name: str, key: str, value: dict) -> int:
    """docs/demo/data/<name>.js, which registers the value under KGCP_DATA[key]."""
    text = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    file = OUT / f"{name}.js"
    file.write_text(
        "// Written by scripts/export_demo.py: do not edit.\n"
        f"(window.KGCP_DATA = window.KGCP_DATA || {{}})[{json.dumps(key)}] = {text};\n"
    )
    return file.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    index = {"sets": [], "intervals": [], "benchmark": benchmark()}
    for kind, pairs, export in (
        ("sets", SETS, export_sets),
        ("intervals", INTERVALS, export_intervals),
    ):
        for dataset, models in pairs:
            for model in models:
                key = f"{kind}-{dataset}-{model}"
                value = export(dataset, model)
                size = write(key, key, value)
                listed = {"key": key, "dataset": dataset, "model": model, "bytes": size}
                listed |= {k: value[k] for k in ("mrr", "hits1", "mae") if k in value}
                index[kind].append(listed)
                print(f"{key}: {size / 1024:.0f} KiB, {len(value['reference'])} reference results")
    write("index", "index", index)
    total = sum(f.stat().st_size for f in OUT.glob("*.js"))
    print(f"{len(list(OUT.glob('*.js')))} files, {total / 1e6:.2f} MB in {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
