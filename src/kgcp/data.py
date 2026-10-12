"""Knowledge graphs, link prediction queries, and their known answers."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import Tensor

TAIL, HEAD = 0, 1  # query directions: (s, p, ?) and (?, p, o)


@dataclass(frozen=True)
class KG:
    """A knowledge graph: named splits of (s, p, o) id triples, and the id names."""

    name: str
    splits: dict[str, Tensor]  # split name -> [m, 3] long
    entities: list[str]
    relations: list[str]

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


class _Rows:
    """Dataclass of equally long tensors, indexable like one tensor."""

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


class KnownAnswers:
    """Answers of queries that are known from a set of triples."""

    def __init__(self, triples: Tensor, num_entities: int, num_relations: int):
        self.num_entities, self.num_relations = num_entities, num_relations
        queries = Queries.from_triples(triples)
        self._codes = (self._key(queries) * num_entities + queries.target).unique()

    def __call__(self, queries: Queries) -> Tensor:
        """[n, E] bool: entity e is a known answer of query i, other than its target."""
        candidates = torch.arange(self.num_entities)
        codes = self._key(queries)[:, None] * self.num_entities + candidates
        known = torch.isin(codes, self._codes)
        known[torch.arange(len(queries)), queries.target] = False
        return known

    def _key(self, queries: Queries) -> Tensor:
        return (queries.anchor * self.num_relations + queries.relation) * 2 + queries.direction
