"""Knowledge graphs, link prediction queries, and their known answers; triples of
uncertain knowledge graphs."""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import torch
from torch import Tensor

TAIL, HEAD = 0, 1  # query directions: (s, p, ?) and (?, p, o)


@dataclass(frozen=True)
class KG:
    """A knowledge graph: named splits of (s, p, o) id triples, and the id names.

    In an uncertain knowledge graph, triples have a confidence in [0, 1]: `confidences`
    holds those of the splits that have them.
    """

    name: str
    splits: dict[str, Tensor]  # split name -> [m, 3] long
    entities: list[str]
    relations: list[str]
    confidences: dict[str, Tensor] = field(default_factory=dict)  # split name -> [m] double

    @property
    def num_entities(self) -> int:
        return len(self.entities)

    @property
    def num_relations(self) -> int:
        return len(self.relations)

    def split(self, *names: str) -> Tensor:
        """The triples of the given splits, concatenated."""
        return torch.cat([self.splits[name] for name in names])

    def known(self, *names: str) -> KnownAnswers:
        """The answers known from the given splits."""
        return KnownAnswers(self.split(*names), self.num_entities, self.num_relations)

    def uncertain_triples(self, name: str) -> UncertainTriples:
        """The triples of a split with their confidences."""
        if name not in self.confidences:
            raise ValueError(f"{self.name}: split {name} has no confidences")
        return UncertainTriples.from_triples(self.splits[name], self.confidences[name])


class _Split:
    """Random splits of the rows of an indexable object."""

    def split(self, fraction: float = 0.5, generator: torch.Generator | None = None):
        """Random split of the rows, `fraction` of them into the first part."""
        first = torch.zeros(len(self), dtype=torch.bool)
        first[torch.randperm(len(self), generator=generator)[: round(fraction * len(self))]] = True
        return self[first], self[~first]


class _Rows(_Split):
    """Dataclass of equally long tensors (or rows), indexable like one tensor."""

    def __len__(self) -> int:
        return len(getattr(self, fields(self)[0].name))

    def __getitem__(self, index):
        return type(self)(**{f.name: getattr(self, f.name)[index] for f in fields(self)})

    @classmethod
    def cat(cls, rows: list):
        return cls(**{f.name: torch.cat([getattr(r, f.name) for r in rows]) for f in fields(cls)})


@dataclass(frozen=True)
class Queries(_Rows):
    """Link prediction queries: given `anchor` and `relation`, predict `target`.

    A tail query (`direction` TAIL) asks (anchor, relation, ?), a head query (HEAD)
    asks (?, relation, anchor). `triple` is the row of the triple it was made from.
    """

    anchor: Tensor
    relation: Tensor
    target: Tensor
    direction: Tensor
    triple: Tensor

    @classmethod
    def from_triples(cls, triples: Tensor) -> Queries:
        """The tail query, then the head query, of every triple."""
        s, p, o = triples.long().T
        m = len(triples)
        return cls(
            anchor=torch.cat([s, o]),
            relation=torch.cat([p, p]),
            target=torch.cat([o, s]),
            direction=torch.cat([torch.full((m,), TAIL), torch.full((m,), HEAD)]),
            triple=torch.arange(m).repeat(2),
        )


@dataclass(frozen=True)
class UncertainTriples(_Rows):
    """Triples (head, relation, tail) of an uncertain knowledge graph and their
    `confidence` in [0, 1], which models predict. `triple` is the row of the triple in its
    split."""

    head: Tensor
    relation: Tensor
    tail: Tensor
    confidence: Tensor
    triple: Tensor

    @classmethod
    def from_triples(cls, triples: Tensor, confidences: Tensor) -> UncertainTriples:
        s, p, o = triples.long().T
        return cls(s, p, o, confidences.double(), torch.arange(len(triples)))


class KnownAnswers:
    """Answers of queries that are known from a set of triples."""

    def __init__(self, triples: Tensor, num_entities: int, num_relations: int):
        self.num_entities, self.num_relations = num_entities, num_relations
        queries = Queries.from_triples(triples)
        self._codes = (self._key(queries) * num_entities + queries.target).unique()

    def __call__(self, queries: Queries) -> Tensor:
        """[n, E] bool: entity e is a known answer of query i, other than its target."""
        # the codes of a query's answers are key * E + answer: a range of the sorted codes
        key = self._key(queries) * self.num_entities
        start = torch.searchsorted(self._codes, key)
        counts = torch.searchsorted(self._codes, key + self.num_entities) - start
        rows = torch.repeat_interleave(torch.arange(len(queries)), counts)
        ranks = torch.arange(len(rows)) - (counts.cumsum(0) - counts)[rows]  # 0, 1, ... per query
        known = torch.zeros(len(queries), self.num_entities, dtype=torch.bool)
        known[rows, self._codes[start[rows] + ranks] - key[rows]] = True
        known[torch.arange(len(queries)), queries.target] = False
        return known

    def _key(self, queries: Queries) -> Tensor:
        return (queries.anchor * self.num_relations + queries.relation) * 2 + queries.direction
