"""Nonconformity of the confidences of triples of uncertain knowledge graphs, and
prediction intervals for them.

A model predicts the confidence c in [0, 1] of a triple: M. The nonconformity of c is the
residual |c - M| divided by a scale s(M) > 0 that the residual score gives. Calibrated
with the threshold t, the interval M ± t s(M) (within [0, 1]) holds the confidences whose
nonconformity is at most t.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch
from torch import Tensor

from kgcp.data import UncertainTriples, _Rows

if TYPE_CHECKING:
    from kgcp.backends import ConfidenceModel


class ResidualScore(Protocol):
    def scale(self, predicted: Tensor) -> Tensor:
        """[n] scale s(M) > 0 of the residuals of the predicted confidences M [n]."""
        ...


@dataclass(frozen=True)
class AbsoluteResidual:
    """|c - M|: intervals of equal width, M ± t (the CP baseline of UnKGCP)."""

    def scale(self, predicted: Tensor) -> Tensor:
        return torch.ones_like(predicted)


@dataclass(frozen=True)
class EntropyResidual:
    """|c - M| / H(M), where H is the binary entropy of the prediction: intervals
    M ± t H(M), wider where the model is uncertain (UnKGCP; Zhu et al., 2025).

    M is clipped to [eps, 1 - eps] for the entropy, as in the authors' code, so that
    H > 0. The base of the logarithm does not matter: it scales t inversely.
    """

    eps: float = 1e-6

    def scale(self, predicted: Tensor) -> Tensor:
        p = predicted.clamp(self.eps, 1 - self.eps)
        return -p * p.log() - (1 - p) * (1 - p).log()


@dataclass(frozen=True)
class Residuals(_Rows):
    """Nonconformity of the true confidence of each triple: `true`.

    `queries` are the triples: as in UnKGCP, a triple is a query, whose answer is its
    confidence. Groupings by relation therefore apply to them as to queries.
    """

    queries: UncertainTriples
    predicted: Tensor  # [n] double: the model's confidence M
    scale: Tensor  # [n] double: s(M) > 0

    @property
    def true(self) -> Tensor:
        """[n] nonconformity of each triple's confidence."""
        return (self.queries.confidence - self.predicted).abs() / self.scale

    def predict(self, thresholds: Tensor) -> PredictionIntervals:
        """The intervals of the triples' thresholds [n]."""
        return PredictionIntervals(self, thresholds)


@dataclass(frozen=True)
class PredictionIntervals:
    """Intervals for the confidences of triples: M ± t s(M), within [0, 1]."""

    residuals: Residuals
    thresholds: Tensor  # [n]

    @property
    def queries(self) -> UncertainTriples:
        return self.residuals.queries

    @property
    def lower(self) -> Tensor:
        r = self.residuals
        return (r.predicted - self.thresholds * r.scale).clamp(0, 1)

    @property
    def upper(self) -> Tensor:
        r = self.residuals
        return (r.predicted + self.thresholds * r.scale).clamp(0, 1)

    @property
    def width(self) -> Tensor:
        return self.upper - self.lower

    @property
    def covered(self) -> Tensor:
        """[n] bool: the interval contains the triple's confidence (its nonconformity is
        at most the threshold; up to rounding, lower <= c <= upper)."""
        return self.residuals.true <= self.thresholds


@torch.no_grad()
def residuals(
    model: ConfidenceModel | Tensor,
    triples: UncertainTriples,
    score: ResidualScore,
    batch_size: int = 4096,
) -> Residuals:
    """Nonconformity of the triples' confidences.

    `model` is a model of an uncertain knowledge graph or precomputed [m] confidences.
    The residuals are kept on the CPU, in double precision.
    """
    if isinstance(model, Tensor):
        predicted = model
    else:
        batches = (triples[i : i + batch_size] for i in range(0, len(triples), batch_size))
        predicted = torch.cat([model.confidence(batch).cpu() for batch in batches])
    predicted = predicted.double().cpu()
    return Residuals(triples, predicted, score.scale(predicted))
