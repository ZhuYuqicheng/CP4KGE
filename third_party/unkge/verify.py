"""Checks the converted models of convert.py.

    uv run python third_party/unkge/verify.py --datasets ppi5k
    uv run python third_party/unkge/verify.py --beurre-code <UnKGCP checkout>/beurre-main

For every converted model: loads it with kgcp.LibKGE, and computes the mean squared and
absolute error of its confidences on the test triples (test) and on the test triples
with as many negatives (test_with_neg); prints them per model (mean ± std over seeds)
with the UnKGCP paper's Table 2, which reports the errors on the test triples.

With --beurre-code, the folder of the BEUrRE code used by UnKGCP (bigumbel.py), also
compares each converted BEUrRE model whose original is still in models/unkgcp/raw/ with
it, on all test_with_neg triples scored by BEUrRE's own code. This runs that code;
param.py, which needs wandb, is replaced by a stand-in.

Exits with an error if a confidence is outside [0, 1] by more than 1e-5 or a converted
BEUrRE model differs from its original by more than 1e-6.
"""

import argparse
import statistics
import sys
import types
from pathlib import Path

import torch
from tqdm import tqdm

import kgcp

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "third_party/libkge/data"
MODELS = ROOT / "models/unkgcp"
DATASETS = ["cn15k", "nl27k", "ppi5k"]
TOLERANCE = 1e-5  # for confidences outside [0, 1]
PAPER = {  # Table 2 of UnKGCP: (model, dataset) -> test MSE, MAE
    ("ukge_logi", "cn15k"): (0.24, 0.41),
    ("ukge_logi", "ppi5k"): (0.01, 0.04),
    ("ukge_logi", "nl27k"): (0.05, 0.11),
    ("passleaf", "cn15k"): (0.24, 0.41),
    ("passleaf", "ppi5k"): (0.01, 0.03),
    ("passleaf", "nl27k"): (0.06, 0.11),
    ("beurre", "cn15k"): (0.12, 0.28),
    ("beurre", "ppi5k"): (0.01, 0.06),
    ("beurre", "nl27k"): (0.03, 0.12),
}


def read_split(dataset: str, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """[m, 3] triples and [m] confidences of a split of prepare_data.py."""
    with open(DATA / dataset / f"{split}.del") as f:
        rows = [line.split("\t") for line in f]
    triples = torch.tensor([[int(x) for x in row[:3]] for row in rows])
    return triples, torch.tensor([float(row[3]) for row in rows], dtype=torch.float64)


@torch.no_grad()
def confidences(model: kgcp.LibKGE, triples: torch.Tensor) -> torch.Tensor:
    s, p, o = triples.to(model.device).T
    return model.model.confidence(s, p, o).double().cpu()


def errors(file: Path, dataset: str) -> dict[str, float]:
    model = kgcp.LibKGE(file, device="cpu")
    if model.kg.name != dataset:
        raise ValueError(f"{file}: dataset {model.kg.name}")
    row = {}
    for split in ("test", "test_with_neg"):
        triples, truth = read_split(dataset, split)
        predicted = confidences(model, triples)
        row[f"{split}_mse"] = (predicted - truth).square().mean().item()
        row[f"{split}_mae"] = (predicted - truth).abs().mean().item()
        # BEUrRE's own float32 scores reach 1 + 1e-6 (log P > 0) for some triples
        outside = (predicted < -TOLERANCE) | (predicted > 1 + TOLERANCE)
        row[f"{split}_outside"] = outside.sum().item()
    return row


def compare_beurre(file: Path, dataset: str, code: Path) -> float:
    """Max difference of the converted model's confidences to BEUrRE's own."""
    sys.modules.setdefault("param", types.SimpleNamespace(Params=type("Params", (), {})))
    sys.path.insert(0, str(code))
    try:
        original = torch.load(
            next((MODELS / "raw" / f"{dataset}-beurre" / file.stem).glob("*.pt")),
            map_location="cpu",
            weights_only=False,
        )
    finally:
        sys.path.remove(str(code))
    triples, truth = read_split(dataset, "test_with_neg")
    with torch.no_grad():
        log_probability, _ = original(triples, truth.float())
    converted = confidences(kgcp.LibKGE(file, device="cpu"), triples)
    return (converted - log_probability.exp().double()).abs().max().item()


def mean_std(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--beurre-code", type=Path, help="folder of BEUrRE's bigumbel.py")
    args = parser.parse_args()

    files = [f for d in args.datasets for f in sorted(MODELS.glob(f"{d}-*/seed*.pt"))]
    rows: dict[tuple[str, str], list[dict]] = {}
    with tqdm(files, unit="model") as bar:
        for file in bar:
            dataset, name = file.parent.name.split("-", 1)
            bar.set_postfix_str(f"{dataset}-{name} {file.stem}")
            row = errors(file, dataset)
            raw = MODELS / "raw" / file.parent.name / file.stem
            if name == "beurre" and args.beurre_code is not None and raw.exists():
                row["difference"] = compare_beurre(file, dataset, args.beurre_code)
            rows.setdefault((dataset, name), []).append(row)

    print(
        "| dataset | model | seeds | test MSE | test MAE | paper MSE / MAE | test_with_neg MSE |"
        " test_with_neg MAE | outside [0, 1] ± 1e-5 | max difference to BEUrRE's code |"
    )
    print("|---" * 10 + "|")
    failed = False
    for (dataset, name), selected in rows.items():
        cells = [
            mean_std([r[key] for r in selected])
            for key in ("test_mse", "test_mae", "test_with_neg_mse", "test_with_neg_mae")
        ]
        paper = PAPER.get((name, dataset))
        paper = f"{paper[0]:.2f} / {paper[1]:.2f}" if paper else "-"
        outside = sum(r["test_outside"] + r["test_with_neg_outside"] for r in selected)
        differences = [r["difference"] for r in selected if "difference" in r]
        difference = f"{max(differences):.1e}" if differences else "-"
        print(
            f"| {dataset} | {name} | {len(selected)} | {cells[0]} | {cells[1]} | {paper} | "
            f"{cells[2]} | {cells[3]} | {outside} | {difference} |"
        )
        failed |= outside > 0 or any(d > 1e-6 for d in differences)
    if failed:
        sys.exit("verification failed: confidences outside [0, 1] or BEUrRE differs")


if __name__ == "__main__":
    main()
