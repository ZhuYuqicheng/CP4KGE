"""Split conformal prediction of answer sets, optionally per group of queries."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from kgcp.data import Queries
from kgcp.groups import Grouping, marginal
from kgcp.scores import Scores


def conformal_quantile(scores: Tensor, alpha: float) -> float:
    """The ceil((n + 1)(1 - alpha))-th smallest of n scores; +inf if that exceeds n.

    A new exchangeable score is at most this value with probability >= 1 - alpha.
    """
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha) - 1e-9)  # tolerance: (n + 1)(1 - alpha) may be integral
    return scores.kthvalue(k).values.item() if k <= n else math.inf


@dataclass(frozen=True)
class PredictionSets:
    """Answer sets as an [n, E] bool `mask` over the candidate entities."""

    mask: Tensor
    queries: Queries

    @property
    def covered(self) -> Tensor:
        """[n] bool: the set contains the query's target."""
        return self.mask[torch.arange(len(self.mask)), self.queries.target]

    @property
    def size(self) -> Tensor:
        return self.mask.sum(1)


class SplitConformal:
    """Split conformal prediction, calibrated separately within each group of queries.

    A query's set holds the candidates scoring at most its group's threshold, the
    conformal quantile of the group's calibration scores. With one group (`marginal`),
    sets cover the target with probability >= 1 - alpha; with `by_relation` (Mondrian
    or predicate-wise CP), they do so for every relation.
    """

    def __init__(self, alpha: float, groups: Grouping = marginal):
        self.alpha, self.groups = alpha, groups
        self.thresholds: dict[int, float] = {}

    def calibrate(self, scores: Scores) -> SplitConformal:
        groups, true = self.groups(scores.queries), scores.true
        self.thresholds = {
            group: conformal_quantile(true[groups == group], self.alpha)
            for group in groups.unique().tolist()
        }
        return self

    def predict(self, scores: Scores) -> PredictionSets:
        groups = self.groups(scores.queries).tolist()
        thresholds = torch.tensor([self.thresholds.get(g, math.inf) for g in groups])
        mask = (scores.values <= thresholds[:, None]) & scores.values.isfinite()
        return PredictionSets(mask, scores.queries)
