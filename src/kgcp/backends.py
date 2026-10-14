"""KGE models that score all candidate answers of queries."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor

from kgcp.data import KG, TAIL, Queries


class KGEModel(Protocol):
    kg: KG  # the graph the model was trained on, with the model's entity/relation ids

    def score(self, queries: Queries) -> Tensor:
        """[n, E] logits of all candidate answers: higher = more plausible."""
        ...


class LibKGE:
    """A trained LibKGE model, from a checkpoint or packaged model file (such as
    models/<dataset>-<model>/seed<N>.pt), or a training folder.

    The model's dataset must be in LibKGE's data folder: packaged models hold the
    entity and relation ids, but not the triples.
    """

    def __init__(self, checkpoint: str | Path, device: str = "auto"):
        from kge.model import KgeModel
        from kge.util.io import load_checkpoint

        path = Path(checkpoint)
        if path.is_dir():
            path = path / "checkpoint_best.pt"
        checkpoint = load_checkpoint(str(path), device=device)
        checkpoint["config"].set("console.quiet", True)
        self.model = KgeModel.create_from(checkpoint)
        self.device = self.model.config.get("job.device")

        dataset = self.model.dataset
        self.kg = KG(
            name=dataset.config.get("dataset.name"),
            splits={s: dataset.split(s).long() for s in ("train", "valid", "test")},
            entities=list(dataset.entity_ids()),
            relations=list(dataset.relation_ids()),
        )

    @torch.no_grad()
    def score(self, queries: Queries) -> Tensor:
        anchor, relation = queries.anchor.to(self.device), queries.relation.to(self.device)
        tail = (queries.direction == TAIL).to(self.device)
        logits = torch.empty(len(queries), self.kg.num_entities, device=self.device)
        if tail.any():  # libkge fails on empty batches
            logits[tail] = self.model.score_sp(anchor[tail], relation[tail])
        if not tail.all():
            logits[~tail] = self.model.score_po(relation[~tail], anchor[~tail])
        return logits
