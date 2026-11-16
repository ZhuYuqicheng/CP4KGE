"""The PyKEEN backend scores and filters exactly like PyKEEN's own evaluation.

The models are trained here, in a few seconds, on the Nations dataset that PyKEEN ships.
"""

import logging
import math

import pytest
import torch

import kgcp

pytest.importorskip("pykeen")

ALPHA = 0.1
# (model, trained with inverse triples): real and complex embeddings, both ways to score heads
MODELS = [("DistMult", False), ("DistMult", True), ("ComplEx", True)]


@pytest.fixture(scope="module", params=MODELS, ids=lambda p: f"{p[0]}{'-inverse' if p[1] else ''}")
def trained(request):
    from pykeen.datasets import Nations
    from pykeen.pipeline import pipeline

    name, inverse = request.param
    logging.getLogger("pykeen").setLevel(logging.ERROR)
    dataset = Nations(create_inverse_triples=inverse)
    result = pipeline(
        dataset=dataset,
        model=name,
        model_kwargs={"embedding_dim": 32},
        training_kwargs={"num_epochs": 30, "use_tqdm": False},
        evaluation_kwargs={"use_tqdm": False},
        random_seed=0,
        device="cpu",
    )
    return dataset, result.model


def test_graph(trained):
    dataset, model = trained
    kg = kgcp.PyKEEN(model, dataset).kg
    assert (kg.name, kg.num_entities, kg.num_relations) == ("nations", 14, 55)  # no inverses
    assert [len(kg.split(s)) for s in ("train", "valid", "test")] == [1592, 199, 201]
    assert torch.equal(kg.split("test"), dataset.testing.mapped_triples)
    assert kg.entities[dataset.entity_to_id["usa"]] == "usa"
    assert kg.relations[dataset.relation_to_id["embassy"]] == "embassy"


def test_scores(trained):
    """[n, E] logits, without gradients; head and tail queries may come in any order."""
    dataset, model = trained
    backend = kgcp.PyKEEN(model, dataset)
    queries = kgcp.Queries.from_triples(backend.kg.split("test")[:20])
    logits = backend.score(queries)
    assert logits.shape == (40, 14) and not logits.requires_grad
    mixed = torch.randperm(40, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(backend.score(queries[mixed]), logits[mixed], atol=1e-6)
    assert not model.training


def pykeen_ranks(logits: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    """Ranks as PyKEEN defines them: the candidates scored above the target plus one
    (optimistic), those scored at least as high (pessimistic), and their mean."""
    true = logits[torch.arange(len(logits)), target][:, None]
    optimistic, pessimistic = (logits > true).sum(1) + 1, (logits >= true).sum(1)
    return {
        "optimistic": optimistic.double(),
        "pessimistic": pessimistic.double(),
        "realistic": (optimistic + pessimistic) / 2,
    }


def test_filtered_ranks_match_pykeen(trained):
    """Metrics from our filtered logits equal those of PyKEEN's evaluator, which filters
    the train, valid and test triples: for head and tail queries, and every kind of rank."""
    from pykeen.evaluation import RankBasedEvaluator

    dataset, model = trained
    expected = RankBasedEvaluator().evaluate(
        model,
        dataset.testing.mapped_triples,
        additional_filter_triples=[
            dataset.training.mapped_triples,
            dataset.validation.mapped_triples,
        ],
        use_tqdm=False,
    )
    backend = kgcp.PyKEEN(model, dataset)
    kg = backend.kg
    queries = kgcp.Queries.from_triples(kg.split("test"))
    known = kg.known("train", "valid", "test")
    # batches of 100: tail queries, then one mixed batch, then head queries
    logits = -kgcp.nonconformity(backend, queries, kgcp.Neg(), known, batch_size=100).values
    sides = {
        "tail": queries.direction == kgcp.TAIL,
        "head": queries.direction == kgcp.HEAD,
        "both": torch.ones(len(queries), dtype=torch.bool),
    }
    for kind, ranks in pykeen_ranks(logits, queries.target).items():
        for side, mask in sides.items():
            ours = {
                "inverse_harmonic_mean_rank": (1 / ranks[mask]).mean(),
                "arithmetic_mean_rank": ranks[mask].mean(),
                "hits_at_1": (ranks[mask] <= 1).double().mean(),
                "hits_at_3": (ranks[mask] <= 3).double().mean(),
            }
            for metric, value in ours.items():
                theirs = expected.get_metric(f"{side}.{kind}.{metric}")
                assert value.item() == pytest.approx(theirs, abs=1e-6), (side, kind, metric)


def test_relation_embeddings(trained):
    """A real vector per relation; with inverse triples, joined with its inverse's."""
    dataset, model = trained
    embeddings = kgcp.PyKEEN(model, dataset).relation_embeddings()
    stored = model.relation_representations[0](indices=None).detach()
    if stored.is_complex():
        stored = torch.view_as_real(stored).flatten(1)
    if model.use_inverse_triples:  # rows 2r and 2r + 1
        stored = torch.cat([stored[0::2], stored[1::2]], dim=1)
    assert embeddings.shape[0] == 55 and not embeddings.is_complex()
    assert torch.equal(embeddings, stored)


def test_conformal_prediction(trained):
    """Answer sets from a PyKEEN model cover as the theory says, and CondKGCP groups the
    relations by the model's embeddings."""
    dataset, model = trained
    backend = kgcp.PyKEEN(model, dataset)
    kg = backend.kg
    queries = kgcp.Queries.from_triples(kg.split("test"))
    coverage = {}
    for score in (kgcp.Neg(), kgcp.Softmax()):
        scores = kgcp.nonconformity(backend, queries, score, known=kg.known("train", "valid"))
        splits = [scores.split(0.5, torch.Generator().manual_seed(i)) for i in range(300)]
        sets = (kgcp.SplitConformal(ALPHA).calibrate(cal).predict(test) for cal, test in splits)
        metrics = kgcp.evaluate(sets, ALPHA)
        coverage[type(score).__name__] = metrics["coverage"]
        assert 1 <= metrics["size"] <= kg.num_entities
    n = len(splits[0][0])
    expected = math.ceil((n + 1) * (1 - ALPHA)) / (n + 1)
    assert coverage["Neg"] == pytest.approx(expected, abs=0.01)  # logits have no ties
    # ties only add coverage: the softmax of a briefly trained model's logits can be so
    # peaked that many candidates share the score 1.0
    assert coverage["Softmax"] >= expected - 0.01

    merged = kgcp.MergedRelations(backend.relation_embeddings(), min_size=20)
    cal, test = splits[0]
    cp = kgcp.CondKGCP(ALPHA, merged, gamma=1.0).calibrate(cal)
    assert set(cp.thresholds) <= set(range(kg.num_relations))
    assert cp.predict(test).size.max() <= kg.num_entities


def test_wrong_dataset(trained):
    from pykeen.datasets import Kinships

    _, model = trained
    with pytest.raises(ValueError, match="entities"):
        kgcp.PyKEEN(model, Kinships())
