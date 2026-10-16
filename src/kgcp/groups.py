"""Groupings of queries for group-conditional calibration: queries -> [n] group ids.

A grouping is fixed (a function of the queries), or learned from the calibration data:
an object whose `fit(calibration)` returns the grouping to use.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch
from torch import Tensor

from kgcp.data import Queries

if TYPE_CHECKING:
    from kgcp.scores import Scores

Grouping = Callable[[Queries], Tensor]


@runtime_checkable
class LearnedGrouping(Protocol):
    def fit(self, calibration: Scores) -> Grouping:
        """The grouping for these calibration scores."""
        ...


def marginal(queries: Queries) -> Tensor:
    return torch.zeros_like(queries.relation)


def by_relation(queries: Queries) -> Tensor:
    return queries.relation


def by_direction(queries: Queries) -> Tensor:
    return queries.direction


@dataclass(frozen=True)
class RelationGroups:
    """Groups of relations: the queries of relation r are in group `table[r]`."""

    table: Tensor  # [R] long

    def __call__(self, queries: Queries) -> Tensor:
        return self.table[queries.relation]


class MergedRelations:
    """Relations with fewer than `min_size` calibration queries join the most similar
    relation that has at least `min_size` (Algorithm 1 of CondKGCP; Zhu et al., 2025).

    Similarity is the negative L1 distance of the relations' embeddings [R, d]. A group
    is named by its relation with enough queries. If no relation has `min_size`
    queries, all relations form one group.
    """

    def __init__(self, embeddings: Tensor, min_size: int):
        self.min_size = min_size
        embeddings = embeddings.float()
        self.distances = torch.cdist(embeddings, embeddings, p=1)  # [R, R], once per model

    def fit(self, calibration: Scores) -> RelationGroups:
        num_relations = len(self.distances)
        counts = torch.bincount(calibration.queries.relation, minlength=num_relations)
        large = counts >= self.min_size
        if not large.any():
            return RelationGroups(torch.full((num_relations,), counts.argmax().item()))
        nearest = large.nonzero().squeeze(1)[self.distances[:, large].argmin(1)]
        return RelationGroups(torch.where(large, torch.arange(num_relations), nearest))
