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
    """Answer sets: the candidates whose score is at most their query's threshold."""

    scores: Scores
    thresholds: Tensor  # [n]

    @property
    def queries(self) -> Queries:
        return self.scores.queries

    @property
    def mask(self) -> Tensor:
        """[n, E] bool over the candidate entities."""
        values = self.scores.values
        return (values <= self.thresholds[:, None]) & values.isfinite()

    @property
    def covered(self) -> Tensor:
        """[n] bool: the set contains the query's target."""
        true = self.scores.true
        return (true <= self.thresholds) & true.isfinite()

    @property
    def size(self) -> Tensor:
        return self.scores.count(self.thresholds)


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
        groups, order = self.groups(scores.queries).sort(stable=True)
        keys, counts = groups.unique_consecutive(return_counts=True)
        per_group = scores.true[order].split(counts.tolist())
        self.thresholds = {
            group: conformal_quantile(true, self.alpha)
            for group, true in zip(keys.tolist(), per_group, strict=True)
        }
        return self

    def predict(self, scores: Scores) -> PredictionSets:
        groups = self.groups(scores.queries).tolist()
        thresholds = torch.tensor([self.thresholds.get(g, math.inf) for g in groups])
        return PredictionSets(scores, thresholds)
