"""Coverage and size of prediction sets."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from kgcp.conformal import PredictionSets
from kgcp.data import HEAD, TAIL, Queries


def evaluate(
    sets: PredictionSets | Iterable[PredictionSets], alpha: float, min_queries: int = 10
) -> dict[str, float]:
    """Coverage and size of prediction sets, overall, per direction and per relation.

    Several prediction sets (e.g. one per random calibration split) are pooled, so a
    relation's coverage is its average over the splits. Relation metrics use the
    relations with at least `min_queries` distinct queries:

    - covgap: mean |coverage - (1 - alpha)|
    - ucg: mean shortfall of the under-covered relations (0 if none)
    - ucr: fraction of relations that are under-covered
    - cond_size: mean set size
    """
    if isinstance(sets, PredictionSets):
        sets = [sets]
    covered, size, queries = [], [], []
    for s in sets:  # keeps only one mask in memory if `sets` is a generator
        covered.append(s.covered)
        size.append(s.size)
        queries.append(s.queries)
    covered, size = torch.cat(covered).double(), torch.cat(size).double()
    queries = Queries.cat(queries)

    metrics = {
        "coverage": covered.mean(),
        "size": size.mean(),
        "empty": (size == 0).double().mean(),
        "singleton": (size == 1).double().mean(),
    }
    for name, direction in (("tail", TAIL), ("head", HEAD)):
        mask = queries.direction == direction
        metrics[f"coverage_{name}"] = covered[mask].mean()
        metrics[f"size_{name}"] = size[mask].mean()

    _, relation = queries.relation.unique(return_inverse=True)  # 0, 1, ... per relation
    query = queries.triple * 2 + queries.direction
    stride = query.max() + 1
    distinct = torch.bincount((relation * stride + query).unique() // stride)
    keep = distinct >= min_queries
    num_queries = torch.bincount(relation).double()
    relation_coverage = (torch.bincount(relation, weights=covered) / num_queries)[keep]
    relation_size = (torch.bincount(relation, weights=size) / num_queries)[keep]
    gaps = (1 - alpha) - relation_coverage
    under = gaps > 1e-12  # tolerance: coverage may equal 1 - alpha up to rounding
    metrics |= {
        "covgap": gaps.abs().mean(),
        "ucg": gaps[under].mean() if under.any() else torch.tensor(0.0),
        "ucr": under.double().mean(),
        "cond_size": relation_size.mean(),
        "relations": keep.sum(),
    }
    return {name: value.item() for name, value in metrics.items()}
