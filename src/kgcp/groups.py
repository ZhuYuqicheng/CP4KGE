"""Groupings of queries for group-conditional calibration: queries -> [n] group ids."""

from collections.abc import Callable

import torch
from torch import Tensor

from kgcp.data import Queries

Grouping = Callable[[Queries], Tensor]


def marginal(queries: Queries) -> Tensor:
    return torch.zeros_like(queries.relation)


def by_relation(queries: Queries) -> Tensor:
    return queries.relation


def by_direction(queries: Queries) -> Tensor:
    return queries.direction
