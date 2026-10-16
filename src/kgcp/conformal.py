"""Split conformal prediction of answer sets: marginal, per group of queries (Mondrian),
and CondKGCP."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import Tensor

from kgcp.data import Queries
from kgcp.groups import Grouping, LearnedGrouping, by_relation, marginal
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
    """Split conformal prediction: one threshold for all queries, the conformal quantile
    of the calibration scores. Sets cover the target with probability >= 1 - alpha.

    Subclasses change how queries are grouped (`_partition`), what is calibrated per
    group (`_calibrate`), and the threshold of each query (`_thresholds`).
    """

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.grouping: Grouping = marginal  # of the last calibration
        self.thresholds: dict[int, float] = {}  # group -> score threshold

    def calibrate(self, scores: Scores) -> SplitConformal:
        self.grouping = self._partition(scores)
        self._calibrate(scores, self.grouping(scores.queries))
        return self

    def predict(self, scores: Scores) -> PredictionSets:
        return PredictionSets(scores, self._thresholds(scores, self.grouping(scores.queries)))

    def _partition(self, calibration: Scores) -> Grouping:
        return marginal

    def _calibrate(self, scores: Scores, groups: Tensor) -> None:
        self.thresholds = {
            group: conformal_quantile(true, self.alpha)
            for group, true in _by_group(groups, scores.true)
        }

    def _thresholds(self, scores: Scores, groups: Tensor) -> Tensor:
        """[n] score threshold of each query; +inf (all candidates) for groups without
        calibration queries."""
        return torch.tensor([self.thresholds.get(g, math.inf) for g in groups.tolist()])


class MondrianConformal(SplitConformal):
    """Split conformal prediction calibrated separately within each group of queries
    (Mondrian CP; Vovk et al., 2005): sets cover the target with probability
    >= 1 - alpha within every group. With `by_relation`, this is predicate-wise CP.

    `groups` is a fixed grouping, or one learned on each calibration set (such as
    `MergedRelations`).
    """

    def __init__(self, alpha: float, groups: Grouping | LearnedGrouping = by_relation):
        super().__init__(alpha)
        self.groups = groups

    def _partition(self, calibration: Scores) -> Grouping:
        if isinstance(self.groups, LearnedGrouping):
            return self.groups.fit(calibration)
        return self.groups


class CondKGCP(MondrianConformal):
    """Predicate-conditional conformal prediction (CondKGCP; Zhu et al., 2025): Mondrian
    CP over groups of relations (`MergedRelations`), with rank calibration (Shi et al.,
    2024).

    Within each group g, k(g) is the smallest k such that the target ranks beyond k in
    fewer than a fraction alpha of the calibration queries, a fraction eps(g). The set
    holds the candidates ranked within k(g) whose score is at most the conformal
    quantile at level alpha - gamma * eps(g). Ranks count ties against a candidate.

    If eps(g) were the group's true top-k(g) miscoverage, coverage within the group would
    be >= 1 - alpha - (1 - gamma) * eps(g): >= 1 - alpha for gamma = 1, while a smaller
    gamma gives smaller sets. Estimated on few calibration queries, eps(g) is optimistic,
    and coverage can fall short even for gamma = 1: merging keeps groups large enough.
    """

    def __init__(self, alpha: float, groups: Grouping | LearnedGrouping, gamma: float = 0.01):
        super().__init__(alpha, groups)
        self.gamma = gamma
        self.top_k: dict[int, int] = {}  # group -> k(g)
        self.top_k_miscoverage: dict[int, float] = {}  # group -> eps(g)

    def _calibrate(self, scores: Scores, groups: Tensor) -> None:
        ranks = scores.count(scores.true)  # the targets' ranks, ties counted against them
        self.thresholds, self.top_k, self.top_k_miscoverage = {}, {}, {}
        for group, true, rank in _by_group(groups, scores.true, ranks):
            n = len(rank)
            # more than n (1 - alpha) targets within k: the (floor(n (1 - alpha)) + 1)-th rank
            k = rank.kthvalue(min(math.floor(n * (1 - self.alpha) + 1e-9) + 1, n)).values.item()
            miscoverage = (rank > k).double().mean().item()
            self.top_k[group], self.top_k_miscoverage[group] = k, miscoverage
            level = self.alpha - self.gamma * miscoverage
            self.thresholds[group] = conformal_quantile(true, level)

    def _thresholds(self, scores: Scores, groups: Tensor) -> Tensor:
        """[n] the lower of the score threshold and the threshold of the top k(g)."""
        num_candidates = scores.sorted.shape[1]
        top_k = torch.tensor([self.top_k.get(g, num_candidates) for g in groups.tolist()])
        return super()._thresholds(scores, groups).minimum(scores.top_k_threshold(top_k))


def _by_group(groups: Tensor, *values: Tensor) -> Iterator[tuple]:
    """(group, its values, ...) for every group, in increasing order of group."""
    groups, order = groups.sort(stable=True)
    keys, counts = groups.unique_consecutive(return_counts=True)
    parts = [v[order].split(counts.tolist()) for v in values]
    return zip(keys.tolist(), *parts, strict=True)
