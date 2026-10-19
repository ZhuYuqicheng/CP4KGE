"""Conformal prediction for knowledge graph embeddings: answer sets for link prediction
queries, and intervals for the confidences of triples of uncertain knowledge graphs."""

from kgcp.backends import ConfidenceModel, KGEModel, LibKGE
from kgcp.conformal import CondKGCP, MondrianConformal, SplitConformal, conformal_quantile
from kgcp.data import HEAD, KG, TAIL, KnownAnswers, Queries, UncertainTriples
from kgcp.groups import MergedRelations, RelationGroups, by_direction, by_relation, marginal
from kgcp.intervals import (
    AbsoluteResidual,
    EntropyResidual,
    PredictionIntervals,
    Residuals,
    residuals,
)
from kgcp.metrics import evaluate, evaluate_intervals
from kgcp.scores import (
    APS,
    RAPS,
    MinMax,
    Neg,
    PredictionSets,
    Rank,
    Scores,
    Softmax,
    nonconformity,
)

__all__ = [
    "APS",
    "AbsoluteResidual",
    "CondKGCP",
    "ConfidenceModel",
    "EntropyResidual",
    "HEAD",
    "KG",
    "KGEModel",
    "KnownAnswers",
    "LibKGE",
    "MergedRelations",
    "MinMax",
    "MondrianConformal",
    "Neg",
    "PredictionIntervals",
    "PredictionSets",
    "Queries",
    "RAPS",
    "Rank",
    "RelationGroups",
    "Residuals",
    "Scores",
    "Softmax",
    "SplitConformal",
    "TAIL",
    "UncertainTriples",
    "by_direction",
    "by_relation",
    "conformal_quantile",
    "evaluate",
    "evaluate_intervals",
    "marginal",
    "nonconformity",
    "residuals",
]
