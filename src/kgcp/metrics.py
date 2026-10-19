"""Coverage and size of prediction sets and intervals."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor

from kgcp.data import HEAD, TAIL, Queries, UncertainTriples
from kgcp.intervals import PredictionIntervals
from kgcp.scores import PredictionSets


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

    query = queries.triple * 2 + queries.direction
    metrics |= _per_relation(queries.relation, query, covered, size, alpha, min_queries, "size")
    return {name: value.item() for name, value in metrics.items()}


def evaluate_intervals(
    intervals: PredictionIntervals | Iterable[PredictionIntervals],
    alpha: float,
    min_triples: int = 10,
) -> dict[str, float]:
    """Coverage and width of prediction intervals, overall and per relation.

    Several prediction intervals (e.g. one per random calibration split) are pooled, as
    in `evaluate`. Relation metrics use the relations with at least `min_triples` distinct
    triples: covgap, ucg, ucr as in `evaluate`, and cond_width, the mean width. full is
    the fraction of intervals that are all of [0, 1].
    """
    if isinstance(intervals, PredictionIntervals):
        intervals = [intervals]
    covered, width, triples = [], [], []
    for i in intervals:
        covered.append(i.covered)
        width.append(i.width)
        triples.append(i.queries)
    covered, width = torch.cat(covered).double(), torch.cat(width).double()
    triples = UncertainTriples.cat(triples)
    metrics = {
        "coverage": covered.mean(),
        "width": width.mean(),
        "full": (width >= 1).double().mean(),
    }
    metrics |= _per_relation(
        triples.relation, triples.triple, covered, width, alpha, min_triples, "width"
    )
    return {name: value.item() for name, value in metrics.items()}


def _per_relation(
    relation: Tensor,
    unit: Tensor,
    covered: Tensor,
    size: Tensor,
    alpha: float,
    min_units: int,
    name: str,
) -> dict[str, Tensor]:
    """Coverage gaps and mean `name` (`size`) per relation, over the relations with at
    least `min_units` distinct units (queries or triples, by their ids `unit`)."""
    _, relation = relation.unique(return_inverse=True)  # 0, 1, ... per relation
    stride = unit.max() + 1
    distinct = torch.bincount((relation * stride + unit).unique() // stride)
    keep = distinct >= min_units
    num_units = torch.bincount(relation).double()
    relation_coverage = (torch.bincount(relation, weights=covered) / num_units)[keep]
    relation_size = (torch.bincount(relation, weights=size) / num_units)[keep]
    gaps = (1 - alpha) - relation_coverage
    under = gaps > 1e-12  # tolerance: coverage may equal 1 - alpha up to rounding
    return {
        "covgap": gaps.abs().mean(),
        "ucg": gaps[under].mean() if under.any() else torch.tensor(0.0),
        "ucr": under.double().mean(),
        f"cond_{name}": relation_size.mean(),
        "relations": keep.sum(),
    }
