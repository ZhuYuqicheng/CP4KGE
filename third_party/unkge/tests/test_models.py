"""The UKGE and BEUrRE models on LibKGE's toy dataset, randomly initialized.

uv run pytest third_party/unkge/tests
"""

import pytest
import torch
from kge import Config, Dataset
from kge.model import KgeModel
from unkge import MODULES


def create(model: str, options: dict) -> KgeModel:
    torch.manual_seed(0)
    config = Config()
    config.set("modules", MODULES)
    config.set("console.quiet", True)
    config.set("job.device", "cpu")
    config.set("dataset.name", "toy")
    config.set("model", model)
    config._import(model)
    for key, value in options.items():
        config.set(key, value)
    kge_model = KgeModel.create(config, Dataset.create(config))
    with torch.no_grad():  # nonzero scale factors
        for parameter in kge_model._scorer.parameters():
            parameter.normal_()
    return kge_model.eval()


MODELS = {
    "ukge_logi": ("ukge", {"ukge.output": "logi"}),
    "ukge_rect": ("ukge", {"ukge.output": "rect"}),
    "beurre": ("beurre", {"beurre.entity_embedder.dim": 6, "beurre.relation_embedder.dim": 12}),
}


@pytest.fixture(params=list(MODELS))
def model(request):
    return create(*MODELS[request.param])


@torch.no_grad()
def test_all_candidates_score_as_triples(model):
    """score_sp and score_po give the scores of score_spo, for every candidate."""
    triples = model.dataset.split("test")[:5].long()
    s, p, o = triples.T
    entities = torch.arange(model.dataset.num_entities())
    tails, heads = model.score_sp(s, p), model.score_po(p, o)
    for i in range(len(triples)):
        n = len(entities)
        expected = model.score_spo(s[i].repeat(n), p[i].repeat(n), entities)
        torch.testing.assert_close(tails[i], expected)
        expected = model.score_spo(entities, p[i].repeat(n), o[i].repeat(n))
        torch.testing.assert_close(heads[i], expected)


@torch.no_grad()
def test_ukge_score():
    model = create(*MODELS["ukge_logi"])
    s, p, o = model.dataset.split("test")[:5].long().T
    distmult = (
        model.get_s_embedder().embed(s)
        * model.get_p_embedder().embed(p)
        * model.get_o_embedder().embed(o)
    ).sum(1)
    expected = model._scorer.weight * distmult + model._scorer.bias
    torch.testing.assert_close(model.score_spo(s, p, o), expected)
    torch.testing.assert_close(model.confidence(s, p, o), expected.sigmoid())


@torch.no_grad()
def test_confidences_in_unit_interval(model):
    s, p, o = model.dataset.split("test").long().T
    confidence = model.confidence(s, p, o)
    assert ((confidence >= 0) & (confidence <= 1 + 1e-6)).all()


def test_beurre_dimensions():
    with pytest.raises(ValueError, match="2d"):
        create("beurre", {"beurre.entity_embedder.dim": 6, "beurre.relation_embedder.dim": 6})
