"""Conformal prediction of answer sets for knowledge graph embeddings."""

from kgcp.backends import KGEModel, LibKGE
from kgcp.conformal import (
    CondKGCP,
    MondrianConformal,
    PredictionSets,
    SplitConformal,
    conformal_quantile,
)
from kgcp.data import HEAD, KG, TAIL, KnownAnswers, Queries
from kgcp.groups import MergedRelations, RelationGroups, by_direction, by_relation, marginal
from kgcp.metrics import evaluate
from kgcp.scores import APS, RAPS, MinMax, Neg, Rank, Scores, Softmax, nonconformity

__all__ = [
    "APS",
    "CondKGCP",
    "HEAD",
    "KG",
    "KGEModel",
    "KnownAnswers",
    "LibKGE",
    "MergedRelations",
    "MinMax",
    "MondrianConformal",
    "Neg",
    "PredictionSets",
    "Queries",
    "RAPS",
    "Rank",
    "RelationGroups",
    "Scores",
    "Softmax",
    "SplitConformal",
    "TAIL",
    "by_direction",
    "by_relation",
    "conformal_quantile",
    "evaluate",
    "marginal",
    "nonconformity",
]
