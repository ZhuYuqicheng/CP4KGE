"""Nonconformity scores of candidate answers, lower means more plausible; and answer sets.

A score function maps a batch of logits [B, E] (higher = more plausible; -inf for
filtered candidates) to nonconformity scores [B, E]. Randomized scores draw from the
given generator.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from kgcp.data import KnownAnswers, Queries, _Split

if TYPE_CHECKING:
    from kgcp.backends import KGEModel

ScoreFunction = Callable[[Tensor, torch.Generator], Tensor]


@dataclass(frozen=True)
class Neg:
    """Negative logit."""

    def __call__(self, logits: Tensor, generator: torch.Generator) -> Tensor:
        return -logits


@dataclass(frozen=True)
class MinMax:
    """One minus the logit min-max normalized over the candidates of a query."""

    def __call__(self, logits: Tensor, generator: torch.Generator) -> Tensor:
        low = logits.masked_fill(logits.isneginf(), torch.inf).amin(1, keepdim=True)
        high = logits.amax(1, keepdim=True)
        return (high - logits) / (high - low).clamp_min(1e-12)


@dataclass(frozen=True)
class Softmax:
    """One minus the softmax probability (LAC; Sadinle et al., 2019)."""

    def __call__(self, logits: Tensor, generator: torch.Generator) -> Tensor:
        return 1 - logits.softmax(1)


@dataclass(frozen=True)
class APS:
    """Softmax mass of the candidates ranked above, plus u times the own mass, with
    u ~ U(0, 1) per query, or u = 1 if not randomized (Romano et al., 2020)."""

    randomized: bool = True

    def __call__(self, logits: Tensor, generator: torch.Generator) -> Tensor:
        probs, order = logits.softmax(1).sort(1, descending=True)
        u = 1.0
        if self.randomized:  # drawn on the CPU: the same on every device
            u = torch.rand(len(probs), 1, generator=generator).to(probs.device)
        ranked = probs.cumsum(1) - (1 - u) * probs + self._penalty(probs)
        return torch.empty_like(ranked).scatter_(1, order, ranked)

    def _penalty(self, probs: Tensor) -> Tensor | float:
        return 0.0


@dataclass(frozen=True)
class RAPS(APS):
    """APS plus `lam` per rank beyond `k_reg` (Angelopoulos et al., 2021)."""

    lam: float = 0.01
    k_reg: int = 5

    def _penalty(self, probs: Tensor) -> Tensor:
        ranks = torch.arange(1, probs.shape[1] + 1, device=probs.device)
        return self.lam * (ranks - self.k_reg).clamp_min(0)


@dataclass(frozen=True)
class Rank:
    """Rank among the candidates (1 = best): conformal top-k prediction."""

    def __call__(self, logits: Tensor, generator: torch.Generator) -> Tensor:
        order = logits.argsort(dim=1, descending=True, stable=True)
        ranks = torch.arange(1, logits.shape[1] + 1, dtype=logits.dtype, device=logits.device)
        ranks = ranks.expand_as(logits)
        return torch.empty_like(logits).scatter_(1, order, ranks)


@dataclass(frozen=True)
class Scores(_Split):
    """Nonconformity of every candidate entity for each query: [n, E] `values`.

    Filtered candidates score +inf and never enter a prediction set. Subsets of the
    queries (indexing, `split`) share the score matrices: they hold their rows' indices.
    Without `matrix` (half the memory), coverage and set sizes are available, but not
    `values` or the sets' masks.

    `split` splits the queries individually, which keeps them exchangeable. The head and
    tail query of a triple are correlated, so splitting by triple would weaken the
    guarantee to ceil((n + 1)(1 - alpha)) / (n + 2).
    """

    queries: Queries
    sorted: Tensor  # [N, E] each query's scores sorted, for set sizes and ranks
    target_scores: Tensor  # [N] score of each query's target
    matrix: Tensor | None = None  # [N, E] scores of the queries that subsets are taken from
    rows: Tensor | None = None  # [n] rows of the queries that subsets are taken from; None: all

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, index) -> Scores:
        return dataclasses.replace(self, queries=self.queries[index], rows=self._rows[index])

    @property
    def _rows(self) -> Tensor:
        return torch.arange(len(self.sorted)) if self.rows is None else self.rows

    @property
    def values(self) -> Tensor:
        """[n, E] scores; a copy for a subset of the queries."""
        if self.matrix is None:
            raise ValueError("scores kept without their matrix (keep_matrix=False)")
        return self.matrix if self.rows is None else self.matrix[self.rows]

    @property
    def true(self) -> Tensor:
        """[n] nonconformity of each query's target."""
        return self.target_scores if self.rows is None else self.target_scores[self.rows]

    def count(self, thresholds: Tensor) -> Tensor:
        """[n] number of candidates whose score is finite and at most the query's threshold.

        A binary search in each sorted row: no pass over the scores.
        """
        rows, num_candidates = self._rows, self.sorted.shape[1]
        thresholds = thresholds.clamp(max=torch.finfo(self.sorted.dtype).max)  # +inf: finite
        low, high = torch.zeros_like(rows), torch.full_like(rows, num_candidates)
        for _ in range(num_candidates.bit_length()):  # the first `low` scores are at most it
            active = low < high
            middle = (low + high) // 2
            below = self.sorted[rows, middle.clamp(max=num_candidates - 1)] <= thresholds
            low = torch.where(active & below, middle + 1, low)
            high = torch.where(active & ~below, middle, high)
        return low

    def top_k_threshold(self, k: Tensor) -> Tensor:
        """[n] the threshold whose set holds the candidates ranked within the query's `k`:
        those scoring below its (k + 1)-th smallest score. A candidate's rank is the
        number of candidates scoring at most it (`count`), so ties count against it.
        """
        rows, num_candidates = self._rows, self.sorted.shape[1]
        above = self.sorted[rows, k.clamp(max=num_candidates - 1)]
        below = torch.nextafter(above, torch.full_like(above, -torch.inf))
        return torch.where(k < num_candidates, below, torch.inf)

    def predict(self, thresholds: Tensor) -> PredictionSets:
        """The answer sets of the queries' score thresholds [n]."""
        return PredictionSets(self, thresholds)


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


@torch.no_grad()
def nonconformity(
    model: KGEModel | Tensor,
    queries: Queries,
    score: ScoreFunction,
    known: KnownAnswers | None = None,
    batch_size: int = 1024,
    seed: int = 0,
    keep_matrix: bool = True,
) -> Scores:
    """Score all candidate answers of the queries.

    `model` is a KGE model or precomputed [n, E] logits; the scores are computed on its
    device and kept on the CPU. Known answers other than the target are removed from the
    candidates before scoring. Without `keep_matrix`, only the sorted scores and the
    targets' scores are kept: enough for coverage and set sizes, in half the memory.
    """
    generator = torch.Generator().manual_seed(seed)
    num_candidates = model.shape[1] if isinstance(model, Tensor) else model.kg.num_entities
    matrix = torch.empty(len(queries), num_candidates) if keep_matrix else None
    ordered = torch.empty(len(queries), num_candidates)
    targets = torch.empty(len(queries))
    for start in range(0, len(queries), batch_size):
        batch = queries[start : start + batch_size]
        if isinstance(model, Tensor):
            logits = model[start : start + batch_size]
        else:
            logits = model.score(batch)
        logits = logits.float()
        filtered = torch.zeros_like(logits, dtype=torch.bool)
        if known is not None:
            filtered = known(batch).to(logits.device)
        scores = score(logits.masked_fill(filtered, -torch.inf), generator)
        scores = scores.masked_fill(filtered, torch.inf)
        if matrix is not None:
            matrix[start : start + batch_size] = scores
        ordered[start : start + batch_size] = scores.sort(1).values
        targets[start : start + batch_size] = scores.gather(
            1, batch.target[:, None].to(scores.device)
        ).squeeze(1)
    return Scores(queries, ordered, targets, matrix)
