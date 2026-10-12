"""Nonconformity scores of candidate answers: lower means more plausible.

A score function maps a batch of logits [B, E] (higher = more plausible; -inf for
filtered candidates) to nonconformity scores [B, E]. Randomized scores draw from the
given generator.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from kgcp.data import KnownAnswers, Queries, _Rows

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
        u = torch.rand(len(probs), 1, generator=generator) if self.randomized else 1.0
        ranked = probs.cumsum(1) - (1 - u) * probs + self._penalty(probs.shape[1])
        return torch.empty_like(ranked).scatter_(1, order, ranked)

    def _penalty(self, num_candidates: int) -> Tensor | float:
        return 0.0


@dataclass(frozen=True)
class RAPS(APS):
    """APS plus `lam` per rank beyond `k_reg` (Angelopoulos et al., 2021)."""

    lam: float = 0.01
    k_reg: int = 5

    def _penalty(self, num_candidates: int) -> Tensor:
        return self.lam * (torch.arange(1, num_candidates + 1) - self.k_reg).clamp_min(0)


@dataclass(frozen=True)
class Rank:
    """Rank among the candidates (1 = best): conformal top-k prediction."""

    def __call__(self, logits: Tensor, generator: torch.Generator) -> Tensor:
        order = logits.argsort(dim=1, descending=True, stable=True)
        ranks = torch.arange(1, logits.shape[1] + 1, dtype=logits.dtype).expand_as(logits)
        return torch.empty_like(logits).scatter_(1, order, ranks)


@dataclass(frozen=True)
class Scores(_Rows):
    """Nonconformity of every candidate entity for each query: [n, E] `values`.

    Filtered candidates score +inf and never enter a prediction set.
    """

    queries: Queries
    values: Tensor

    @property
    def true(self) -> Tensor:
        """[n] nonconformity of each query's target."""
        return self.values[torch.arange(len(self)), self.queries.target]

    def split(self, fraction: float = 0.5, generator: torch.Generator | None = None):
        """Random split of the queries, `fraction` of them into the first part.

        Queries are split individually, which keeps them exchangeable. The head and tail
        query of a triple are correlated, so splitting by triple would weaken the
        guarantee to ceil((n + 1)(1 - alpha)) / (n + 2).
        """
        first = torch.zeros(len(self), dtype=torch.bool)
        first[torch.randperm(len(self), generator=generator)[: round(fraction * len(self))]] = True
        return self[first], self[~first]


@torch.no_grad()
def nonconformity(
    model: KGEModel | Tensor,
    queries: Queries,
    score: ScoreFunction,
    known: KnownAnswers | None = None,
    batch_size: int = 1024,
    seed: int = 0,
) -> Scores:
    """Score all candidate answers of the queries.

    `model` is a KGE model or precomputed [n, E] logits. Known answers other than the
    target are removed from the candidates before scoring.
    """
    generator = torch.Generator().manual_seed(seed)
    values = []
    for start in range(0, len(queries), batch_size):
        batch = queries[start : start + batch_size]
        if isinstance(model, Tensor):
            logits = model[start : start + batch_size]
        else:
            logits = model.score(batch)
        logits = logits.float().cpu()
        filtered = torch.zeros_like(logits, dtype=torch.bool)
        if known is not None:
            filtered = known(batch)
        scores = score(logits.masked_fill(filtered, -torch.inf), generator)
        values.append(scores.masked_fill(filtered, torch.inf))
    return Scores(queries, torch.cat(values))
