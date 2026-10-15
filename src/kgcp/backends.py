"""KGE models that score all candidate answers of queries."""

from __future__ import annotations

import tempfile
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

    Checkpoints of older LibKGE versions (such as LibKGE's published models of
    2019-2020) load too: their options are read into today's configuration, and options
    that LibKGE no longer has are kept but unused.
    """

    def __init__(self, checkpoint: str | Path, device: str = "auto"):
        from kge import Config, Dataset
        from kge.model import KgeModel
        from kge.util.io import load_checkpoint

        path = Path(checkpoint)
        if path.is_dir():
            path = path / "checkpoint_best.pt"
        checkpoint = load_checkpoint(str(path), device=device)

        # as KgeModel.create_from, which rejects options that LibKGE no longer has
        config = Config(folder=checkpoint["folder"])
        config.load_config(checkpoint["config"], create=True)
        config.set("console.quiet", True)
        config.log_folder = tempfile.mkdtemp(prefix="kge-")
        _use_existing_dataset_files(config)
        dataset = Dataset.create_from(checkpoint, config)
        for kind, ids, n in (
            ("entity", dataset.entity_ids(), dataset.num_entities()),
            ("relation", dataset.relation_ids(), dataset.num_relations()),
        ):
            if len(ids) != n:
                raise ValueError(f"{path}: {n} {kind} embeddings, but {len(ids)} {kind} ids")
        self.model = KgeModel.create(config, dataset, init_for_load_only=True)
        self.model.load(checkpoint["model"])
        self.model.eval()
        self.device = config.get("job.device")

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


def _use_existing_dataset_files(config) -> None:
    """Replaces the dataset file names of a LibKGE config that do not exist in the
    dataset's folder by those of its dataset.yaml.

    Checkpoints of 2019-2020 name the id files entity_map.del and relation_map.del;
    LibKGE now writes entity_ids.del and relation_ids.del, and keeps a checkpoint's
    file names over those of dataset.yaml.
    """
    import yaml
    from kge import Config
    from kge.misc import kge_base_dir

    folder = Path(kge_base_dir()) / "data" / config.get("dataset.name")
    if not (folder / "dataset.yaml").exists():
        return  # LibKGE reports the missing dataset
    options = yaml.safe_load((folder / "dataset.yaml").read_text())["dataset"]
    for key, filename in Config.flatten(options).items():
        key = f"dataset.{key}"
        if key.endswith(".filename") and config.exists(key):
            if not (folder / config.get(key)).exists():
                config.set(key, filename)
