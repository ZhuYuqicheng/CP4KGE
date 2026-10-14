"""The LibKGE backend scores and filters exactly like LibKGE's own evaluation."""

import json
from pathlib import Path

import pytest
import torch

import kgcp

ROOT = Path(__file__).parents[1]
PAIRS = [
    f"{d}-{m}" for d in ("nations", "umls", "kinship") for m in ("distmult", "complex", "rescal")
]
SEEDS = range(1, 6)


def load(pair: str, seed: int) -> kgcp.LibKGE:
    path = ROOT / "models" / pair / f"seed{seed}.pt"
    if not path.exists():
        pytest.skip(f"no trained model {path} (see README.md#trained-models)")
    return kgcp.LibKGE(path, device="cpu")


def libkge_ranks(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Ranks as LibKGE computes them (rounded_mean_rank): candidates within its tolerance
    of the target's logit, the target included, are ties and add half their number."""
    true = logits[torch.arange(len(logits)), target][:, None]
    ties = torch.isclose(logits, true, rtol=1e-4, atol=1e-5)
    return ((logits > true) & ~ties).sum(1) + ties.sum(1) // 2 + 1


def test_graph():
    kg = load("nations-distmult", 1).kg
    assert (kg.name, kg.num_entities, kg.num_relations) == ("nations", 14, 55)
    assert [len(kg.split(s)) for s in ("train", "valid", "test")] == [1592, 199, 201]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("pair", PAIRS)
def test_filtered_ranks_match_libkge(pair, seed):
    """MRR and Hits@1 from our filtered logits equal LibKGE's test metrics (which filter
    train, valid and test triples), as recorded in results/hpo/."""
    model = load(pair, seed)
    kg = model.kg
    queries = kgcp.Queries.from_triples(kg.split("test"))
    known = kg.known("train", "valid", "test")
    # batches of 100: tail queries, then one mixed batch, then head queries
    logits = -kgcp.nonconformity(model, queries, kgcp.Neg(), known, batch_size=100).values
    ranks = libkge_ranks(logits, queries.target).double()
    expected = json.loads((ROOT / f"results/hpo/{pair}.json").read_text())["test"][str(seed)]
    assert (1 / ranks).mean().item() == pytest.approx(
        expected["mean_reciprocal_rank_filtered"], abs=1e-6
    )
    assert (ranks == 1).double().mean().item() == pytest.approx(
        expected["hits_at_1_filtered"], abs=1e-6
    )
