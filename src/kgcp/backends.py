"""KGE models that score all candidate answers of queries.

kgcp needs only PyTorch. A backend imports its KGE library when a model is loaded, so the
library is needed only by those who use the backend.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor

from kgcp.data import KG, TAIL, Queries, UncertainTriples


class KGEModel(Protocol):
    kg: KG  # the graph the model was trained on, with the model's entity/relation ids

    def score(self, queries: Queries) -> Tensor:
        """[n, E] logits of all candidate answers: higher = more plausible."""
        ...

    def relation_embeddings(self) -> Tensor:
        """[R, d] a vector per relation, for groupings by relation similarity."""
        ...


class ConfidenceModel(Protocol):
    """A model of an uncertain knowledge graph: it predicts the confidence of triples."""

    kg: KG

    def confidence(self, triples: UncertainTriples) -> Tensor:
        """[m] predicted confidence in [0, 1] of the triples."""
        ...


class LibKGE:
    """A trained LibKGE model, from a checkpoint or packaged model file (such as
    models/<dataset>-<model>/seed<N>.pt), or a training folder.

    The model's dataset must be in LibKGE's data folder: packaged models hold the
    entity and relation ids, but not the triples.

    Checkpoints of older LibKGE versions (such as LibKGE's published models of
    2019-2020) load too: their options are read into today's configuration, and options
    that LibKGE no longer has are kept but unused.

    Models of uncertain knowledge graphs (UKGE, PASSLEAF and BEUrRE of third_party/unkge)
    also predict confidences, and their dataset's splits have a fourth column, the
    confidence of each triple (`kg.confidences`).
    """

    def __init__(self, checkpoint: str | Path, device: str = "auto"):
        try:
            from kge import Config, Dataset
            from kge.model import KgeModel
            from kge.util.io import load_checkpoint
        except ImportError as error:
            raise ImportError(
                "kgcp.LibKGE needs LibKGE, which is not on PyPI: it is installed from a clone "
                "of the kgcp repository, where it finds its datasets, with `uv sync` (or "
                "`pip install -e third_party/libkge -e third_party/unkge`)"
            ) from error

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

        splits = {s: dataset.split(s).long() for s in ("train", "valid", "test")}
        self.kg = KG(
            name=dataset.config.get("dataset.name"),
            splits=splits,
            entities=list(dataset.entity_ids()),
            relations=list(dataset.relation_ids()),
            confidences=_confidences(dataset, splits),
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

    @torch.no_grad()
    def confidence(self, triples: UncertainTriples) -> Tensor:
        """[m] predicted confidence of the triples, on the model's device; for models of
        uncertain knowledge graphs."""
        if not hasattr(self.model, "confidence"):
            raise TypeError(f"{type(self.model).__name__} models predict no confidences")
        s, p, o = (x.to(self.device) for x in (triples.head, triples.relation, triples.tail))
        return self.model.confidence(s, p, o)

    @torch.no_grad()
    def relation_embeddings(self) -> Tensor:
        """[R, d] relation embeddings as the model stores them (e.g. real and imaginary
        parts for ComplEx, flattened matrices for RESCAL), on the CPU. With reciprocal
        relations, a relation's embeddings for tail and for head queries, concatenated."""
        from kge.model import ReciprocalRelationsModel

        num_relations = self.kg.num_relations
        embeddings = self.model.get_p_embedder().embed_all()
        if isinstance(self.model, ReciprocalRelationsModel):
            embeddings = torch.cat([embeddings[:num_relations], embeddings[num_relations:]], 1)
        return embeddings.reshape(num_relations, -1).float().cpu()


class PyKEEN:
    """A trained PyKEEN model with its dataset: a `pykeen.datasets.Dataset`, or any object
    whose `training`, `testing` and (optional) `validation` are the triples factories with
    the ids that the model was trained with. They become the splits train, test and valid.

    The scores are those of PyKEEN's `predict_t` and `predict_h`: a model trained with
    inverse triples scores head queries with the inverse relations, as in PyKEEN's own
    evaluation. `device` moves the model; by default it stays where it is.
    """

    def __init__(self, model, dataset, device: str | None = None):
        if device is not None:
            model = model.to(device)
        self.model = model.eval()
        self.device = model.device

        factories = {"train": dataset.training, "test": dataset.testing}
        if getattr(dataset, "validation", None) is not None:
            factories["valid"] = dataset.validation
        training = dataset.training
        num_relations = training.real_num_relations  # without the inverse relations
        for kind, n, expected in (
            ("entities", model.num_entities, training.num_entities),
            ("relations", model.num_real_relations, num_relations),
        ):
            if n != expected:
                raise ValueError(f"the model has {n} {kind}, but the dataset {expected}")
        name = getattr(dataset, "get_normalized_name", lambda: type(dataset).__name__)()
        self.kg = KG(
            name=name,
            splits={split: factory.mapped_triples.long() for split, factory in factories.items()},
            entities=_labels(training, "entity_id_to_label", training.num_entities),
            relations=_labels(training, "relation_id_to_label", num_relations),
        )

    @torch.no_grad()
    def score(self, queries: Queries) -> Tensor:
        anchor, relation = queries.anchor, queries.relation
        tail = queries.direction == TAIL
        logits = torch.empty(len(queries), self.kg.num_entities, device=self.device)
        on_device = tail.to(self.device)
        if tail.any():
            pairs = torch.stack([anchor[tail], relation[tail]], dim=1)
            logits[on_device] = self.model.predict_t(pairs).float()
        if not tail.all():
            pairs = torch.stack([relation[~tail], anchor[~tail]], dim=1)
            logits[~on_device] = self.model.predict_h(pairs).float()
        return logits

    @torch.no_grad()
    def relation_embeddings(self) -> Tensor:
        """[R, d] the model's first representation of each relation, flattened, on the CPU:
        real and imaginary parts for complex embeddings. With inverse triples, a relation's
        representation and that of its inverse (which scores head queries), concatenated."""
        representations = getattr(self.model, "relation_representations", None)
        if not representations:
            raise TypeError(f"{type(self.model).__name__} models have no relation embeddings")
        embeddings = representations[0](indices=None)
        if embeddings.is_complex():
            embeddings = torch.view_as_real(embeddings)
        # PyKEEN stores relation r in row 2r and its inverse in row 2r + 1
        return embeddings.reshape(self.kg.num_relations, -1).float().cpu()


def _labels(factory, attribute: str, n: int) -> list[str]:
    """The labels of ids 0, ..., n - 1 of a PyKEEN triples factory; the ids themselves for
    a factory without labels."""
    labels = getattr(factory, attribute, None)
    return [str(i) if labels is None else str(labels[i]) for i in range(n)]


def _confidences(dataset, splits: dict[str, Tensor]) -> dict[str, Tensor]:
    """[m] confidences of the splits whose files have a fourth column: the confidence of
    each triple in an uncertain knowledge graph. LibKGE reads only the first three."""
    confidences = {}
    for split, triples in splits.items():
        file = Path(dataset.folder) / dataset.config.get(f"dataset.files.{split}.filename")
        with open(file) as f:
            if len(f.readline().rstrip("\n").split("\t")) < 4:
                continue
            f.seek(0)
            values = [float(line.split("\t")[3]) for line in f]
        if len(values) != len(triples):
            raise ValueError(f"{file}: {len(values)} confidences for {len(triples)} triples")
        confidences[split] = torch.tensor(values, dtype=torch.float64)
    return confidences


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
