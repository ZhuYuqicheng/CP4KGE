"""The LibKGE backend scores and filters exactly like LibKGE's own evaluation."""

import json
from pathlib import Path

import pytest

import kgcp

ROOT = Path(__file__).parents[1]
RESULTS = ROOT / "results/hpo/nations-distmult.json"
SEED = 1


@pytest.fixture(scope="module")
def model():
    rank = json.loads(RESULTS.read_text())["selected"]
    folder = ROOT / f"runs/hpo/final/nations-distmult/top{rank}/seed{SEED}"
    if not folder.exists():
        pytest.skip(f"no trained model in {folder} (see docs/hyperparameter_tuning.md)")
    return kgcp.LibKGE(folder, device="cpu")


def test_graph(model):
    kg = model.kg
    assert (kg.name, kg.num_entities, kg.num_relations) == ("nations", 14, 55)
    assert [len(kg.split(s)) for s in ("train", "valid", "test")] == [1592, 199, 201]


def test_filtered_ranks_match_libkge(model):
    """MRR and Hits@1 from our filtered ranks equal LibKGE's test metrics (which filter
    train, valid and test triples)."""
    kg = model.kg
    queries = kgcp.Queries.from_triples(kg.split("test"))
    known = kg.known("train", "valid", "test")
    # batches of 100 of the 201 tail and 201 head queries: mixed, then head only
    ranks = kgcp.nonconformity(model, queries, kgcp.Rank(), known, batch_size=100).true
    expected = json.loads(RESULTS.read_text())["test"][str(SEED)]
    mrr = (1 / ranks).mean().item()
    hits_at_1 = (ranks == 1).float().mean().item()
    assert mrr == pytest.approx(expected["mean_reciprocal_rank_filtered"], abs=1e-6)
    assert hits_at_1 == pytest.approx(expected["hits_at_1_filtered"], abs=1e-6)
