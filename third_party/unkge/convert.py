"""Converts the authors' models of UnKGCP (download.py models) into LibKGE packaged models.

    uv run --with tensorflow python third_party/unkge/convert.py --datasets ppi5k

Reads models/unkgcp/raw/<dataset>-<model>/seed<N>/ and writes
models/unkgcp/<dataset>-<model>/seed<N>.pt and config.yaml, in the ids of the LibKGE
datasets of prepare_data.py:

- ukge_logi, ukge_rect, passleaf: TensorFlow 1 checkpoints (read with TensorFlow 2) of
  UKGE models. UKGE numbers the entities and relations in the order in which they first
  occur in train, val, test, test_with_neg and softlogic (heads before tails); the rows
  of the embeddings are reordered to the dataset's ids. A converted model must give the
  confidences of UKGE's formula on the original arrays, for all test_with_neg triples.
- beurre: BEUrRE models pickled whole (torch.save of the module). They are unpickled
  with stand-ins for BEUrRE's classes, so none of its code runs; only the tensors and the
  Gumbel beta are kept. Their ids are the dataset's.

Existing converted models are kept (delete them to convert again).
"""

import _compat_pickle
import argparse
import json
import pickle
from pathlib import Path

import torch
from kge import Config, Dataset
from kge.model import KgeModel
from tqdm import tqdm
from unkge import MODULES

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "third_party/libkge/data"
RAW = ROOT / "models/unkgcp/raw"
OUT = ROOT / "models/unkgcp"
DATASETS = ["cn15k", "nl27k", "ppi5k"]
MODELS = {  # model -> (LibKGE model, options)
    "ukge_logi": ("ukge", {"ukge.output": "logi"}),
    "ukge_rect": ("ukge", {"ukge.output": "rect"}),
    "passleaf": ("ukge", {"ukge.output": "logi"}),
    "beurre": ("beurre", {}),
}
UKGE_FILES = ["train", "valid", "test", "test_with_neg", "softlogic"]  # UKGE's order


def read_del(dataset: str, name: str) -> torch.Tensor:
    """[m, 3] long: the triples of a .del file of prepare_data.py, in its order."""
    with open(DATA / dataset / f"{name}.del") as f:
        return torch.tensor([[int(x) for x in line.split("\t")[:3]] for line in f])


def ukge_vocabulary(dataset: str) -> tuple[torch.Tensor, torch.Tensor]:
    """The dataset ids of UKGE's entities and relations, in UKGE's order."""
    entities, relations = {}, {}
    for name in UKGE_FILES:
        for h, r, t in read_del(dataset, name).tolist():
            entities.setdefault(h, len(entities))
            entities.setdefault(t, len(entities))
            relations.setdefault(r, len(relations))
    return torch.tensor(list(entities)), torch.tensor(list(relations))


def read_tensorflow(folder: Path) -> tuple[dict[str, torch.Tensor], int]:
    """UKGE's variables (without the optimizer's) and the checkpoint's epoch."""
    import tensorflow as tf

    index = next(folder.glob("model.bin-*.index"))
    prefix = str(index.with_suffix(""))
    reader = tf.train.load_checkpoint(prefix)
    names = {"graph/ht": "entities", "graph/r": "relations", "weights": "w", "bias": "b"}
    arrays = {key: torch.as_tensor(reader.get_tensor(name)) for name, key in names.items()}
    return arrays, int(prefix.rsplit("-", 1)[1])


def ukge_state(dataset: str, arrays: dict, num_entities: int, num_relations: int):
    """LibKGE state of a UKGE model: the embeddings in the dataset's ids. Entities and
    relations that UKGE never saw keep zero embeddings."""
    entity_order, relation_order = ukge_vocabulary(dataset)
    for kind, order, table in (
        ("entities", entity_order, arrays["entities"]),
        ("relations", relation_order, arrays["relations"]),
    ):
        if len(order) != len(table):
            raise ValueError(f"{dataset}: UKGE knows {len(order)} {kind}, the model {len(table)}")
    entities = torch.zeros(num_entities, arrays["entities"].shape[1])
    relations = torch.zeros(num_relations, arrays["relations"].shape[1])
    entities[entity_order] = arrays["entities"]
    relations[relation_order] = arrays["relations"]
    state = {
        "_entity_embedder._embeddings.weight": entities,
        "_relation_embedder._embeddings.weight": relations,
        "_scorer.weight": arrays["w"].reshape(()),
        "_scorer.bias": arrays["b"].reshape(()),
    }
    return state, (entity_order, relation_order)


def check_ukge(model, dataset: str, arrays: dict, orders, output: str) -> float:
    """Max difference to UKGE's confidences on the test_with_neg triples, computed as
    UKGE's tester does, on the original arrays with UKGE's ids."""
    triples = read_del(dataset, "test_with_neg")
    entity_index = {e: i for i, e in enumerate(orders[0].tolist())}
    relation_index = {r: i for i, r in enumerate(orders[1].tolist())}
    h = torch.tensor([entity_index[e] for e in triples[:, 0].tolist()])
    r = torch.tensor([relation_index[e] for e in triples[:, 1].tolist()])
    t = torch.tensor([entity_index[e] for e in triples[:, 2].tolist()])
    ht, rel = arrays["entities"].double(), arrays["relations"].double()
    score = arrays["w"].double() * (ht[h] * ht[t] * rel[r]).sum(1) + arrays["b"].double()
    original = score.sigmoid() if output == "logi" else score.clamp(0, 1)
    with torch.no_grad():
        converted = model.confidence(triples[:, 0], triples[:, 1], triples[:, 2]).double()
    return (converted - original).abs().max().item()


class _Stub:
    """Stands in for a class of a pickled BEUrRE model, keeping the object's state."""

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__["state"] = state


class _StubUnpickler(pickle.Unpickler):
    """Unpickles torch, numpy and plain Python objects; any other class is a _Stub."""

    SAFE_BUILTINS = {"set", "frozenset", "slice", "range", "complex", "object"}

    def find_class(self, module, name):
        # Python 2 names of protocol 2 pickles, such as __builtin__.set
        module, name = _compat_pickle.NAME_MAPPING.get((module, name), (module, name))
        module = _compat_pickle.IMPORT_MAPPING.get(module, module)
        if module.split(".")[0] in ("torch", "collections", "numpy", "_codecs", "copyreg"):
            return super().find_class(module, name)
        if module == "builtins" and name in self.SAFE_BUILTINS:
            return super().find_class(module, name)
        return type(name, (_Stub,), {"__module__": module})


class _StubPickle:
    Unpickler = _StubUnpickler
    __name__ = "unkge_stub_pickle"

    @staticmethod
    def load(file, **kwargs):
        return _StubUnpickler(file, **kwargs).load()


def read_beurre(folder: Path) -> tuple[dict[str, torch.Tensor], float]:
    """BEUrRE's parameters and Gumbel beta."""
    file = next(folder.glob("*.pt"))
    model = torch.load(file, map_location="cpu", weights_only=False, pickle_module=_StubPickle)
    parameters = {k: v.detach().float() for k, v in model.state["_parameters"].items()}
    return parameters, float(model.state["gumbel_beta"])


def beurre_state(parameters: dict) -> dict[str, torch.Tensor]:
    entities = torch.cat([parameters["min_embedding"], parameters["delta_embedding"]], 1)
    relations = torch.cat(
        [
            parameters[f"rel_{kind}_for_{side}"]
            for side in ("head", "tail")
            for kind in ("trans", "scale")
        ],
        1,
    )
    return {
        "_entity_embedder._embeddings.weight": entities,
        "_relation_embedder._embeddings.weight": relations,
    }


def build(dataset: str, model: str, options: dict, state: dict):
    """A LibKGE model with the given state, and its config and dataset."""
    config = Config()
    config.set("modules", MODULES)
    config.set("job.device", "cpu")
    config.set("console.quiet", True)
    config.set("dataset.name", dataset)
    config.set("model", model)
    config._import(model)
    for key, value in options.items():
        config.set(key, value)
    for embedder, key in (("entity", "_entity_embedder"), ("relation", "_relation_embedder")):
        config.set(f"{model}.{embedder}_embedder.dim", state[f"{key}._embeddings.weight"].shape[1])
        config.set(f"{model}.{embedder}_embedder.regularize", "")
    data = Dataset.create(config, preload_data=False)
    kge_model = KgeModel.create(config, data, init_for_load_only=True)
    kge_model.load_state_dict(state)
    kge_model.eval()
    return kge_model, config, data


def convert(dataset: str, name: str, seed_folder: Path) -> None:
    target = OUT / f"{dataset}-{name}" / f"{seed_folder.name}.pt"
    if target.exists():
        return
    model, options = MODELS[name]
    data = Dataset.create(_dataset_config(dataset), preload_data=False)
    num_entities, num_relations = data.num_entities(), data.num_relations()
    report = {}
    if model == "ukge":
        arrays, epoch = read_tensorflow(seed_folder)
        state, orders = ukge_state(dataset, arrays, num_entities, num_relations)
        kge_model, config, data = build(dataset, model, options, state)
        report["max_difference"] = check_ukge(
            kge_model, dataset, arrays, orders, options["ukge.output"]
        )
        if report["max_difference"] > 1e-5:
            raise ValueError(f"{target}: differs from UKGE by {report['max_difference']}")
        report["unseen_entities"] = num_entities - len(orders[0])
        report["unseen_relations"] = num_relations - len(orders[1])
    else:
        parameters, beta = read_beurre(seed_folder)
        shapes = {k: tuple(v.shape) for k, v in parameters.items()}
        if shapes["min_embedding"][0] != num_entities:
            raise ValueError(f"{seed_folder}: parameter shapes {shapes}")
        if shapes["rel_trans_for_head"][0] != num_relations:
            raise ValueError(f"{seed_folder}: parameter shapes {shapes}")
        kge_model, config, data = build(
            dataset, model, options | {"beurre.gumbel_beta": beta}, beurre_state(parameters)
        )
        epoch = None
    source = json.loads((seed_folder / "source.json").read_text())
    package = {
        "type": "package",
        "model": (kge_model.state_dict(), {}),
        "epoch": epoch,
        "job_id": None,
        "valid_trace": [],
        "source": {"files": source, "converted_by": "third_party/unkge/convert.py"},
    }
    package = config.save_to(package)
    package = data.save_to(package, ["entity_ids", "relation_ids"])
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(package, target)
    config.save(target.parent / "config.yaml")
    tqdm.write(f"{target.relative_to(ROOT)}: {report}")


def _dataset_config(dataset: str) -> Config:
    config = Config()
    config.set("console.quiet", True)
    config.set("dataset.name", dataset)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    args = parser.parse_args()
    runs = [
        (dataset, name, seed_folder)
        for dataset in args.datasets
        for name in args.models
        for seed_folder in sorted((RAW / f"{dataset}-{name}").glob("seed*"))
    ]
    with tqdm(runs, unit="model") as bar:
        for dataset, name, seed_folder in bar:
            bar.set_postfix_str(f"{dataset}-{name} {seed_folder.name}")
            convert(dataset, name, seed_folder)


if __name__ == "__main__":
    main()
