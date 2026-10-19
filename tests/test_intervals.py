"""Prediction intervals for the confidences of triples: guarantees and edge cases on
synthetic predictions, and the converted UnKGCP models (third_party/unkge)."""

import math
from pathlib import Path

import pytest
import torch

import kgcp

R, M, ALPHA = 4, 600, 0.1
SCORES = [kgcp.AbsoluteResidual(), kgcp.EntropyResidual()]
ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def toy():
    """Triples with confidences, and predictions whose error grows with the relation id."""
    generator = torch.Generator().manual_seed(0)
    triples = torch.stack(
        [
            torch.randint(50, (M,), generator=generator),
            torch.randint(R, (M,), generator=generator),
            torch.randint(50, (M,), generator=generator),
        ],
        dim=1,
    )
    confidences = torch.rand(M, generator=generator, dtype=torch.float64)
    triples = kgcp.UncertainTriples.from_triples(triples, confidences)
    noise = torch.randn(M, generator=generator, dtype=torch.float64)
    predicted = (confidences + 0.05 * (1 + triples.relation) * noise).clamp(0, 1)
    return triples, predicted


def random_splits(residuals, n, fraction=0.5):
    return [residuals.split(fraction, torch.Generator().manual_seed(i)) for i in range(n)]


def test_residual_scores():
    predicted = torch.tensor([0.0, 0.5, 0.9, 1.0], dtype=torch.float64)
    assert torch.equal(kgcp.AbsoluteResidual().scale(predicted), torch.ones(4, dtype=torch.float64))
    scale = kgcp.EntropyResidual().scale(predicted)
    assert scale[1].item() == pytest.approx(math.log(2))
    assert scale[2].item() == pytest.approx(-0.9 * math.log(0.9) - 0.1 * math.log(0.1))
    assert scale[0] > 0 and scale[0].item() == pytest.approx(scale[3].item())  # clipped


@pytest.mark.parametrize("score", SCORES, ids=lambda s: type(s).__name__)
def test_marginal_coverage(toy, score):
    """Expected coverage is exactly k / (n + 1), k = ceil((n + 1)(1 - alpha)), without ties."""
    triples, predicted = toy
    residuals = kgcp.residuals(predicted, triples, score)
    splits = random_splits(residuals, 1000, fraction=20 / M)
    intervals = (kgcp.SplitConformal(ALPHA).calibrate(cal).predict(test) for cal, test in splits)
    coverage = kgcp.evaluate_intervals(intervals, ALPHA)["coverage"]
    n = len(splits[0][0])
    expected = math.ceil((n + 1) * (1 - ALPHA)) / (n + 1)
    assert expected - 0.01 <= coverage <= expected + 0.01


def relation_coverage(residuals, cp):
    covered, relation = [], []
    for cal, test in random_splits(residuals, 300):
        covered.append(cp.calibrate(cal).predict(test).covered)
        relation.append(test.queries.relation)
    covered, relation = torch.cat(covered).float(), torch.cat(relation)
    return torch.stack([covered[relation == r].mean() for r in range(R)])


@pytest.mark.parametrize("score", SCORES, ids=lambda s: type(s).__name__)
def test_relation_conditional_coverage(toy, score):
    triples, predicted = toy
    residuals = kgcp.residuals(predicted, triples, score)
    assert relation_coverage(residuals, kgcp.MondrianConformal(ALPHA)).min() >= 1 - ALPHA - 0.01
    # marginal calibration under-covers the relation with the largest errors
    assert relation_coverage(residuals, kgcp.SplitConformal(ALPHA)).min() < 1 - ALPHA - 0.02


@pytest.mark.parametrize("score", SCORES, ids=lambda s: type(s).__name__)
def test_intervals(toy, score):
    triples, predicted = toy
    residuals = kgcp.residuals(predicted, triples, score)
    cal, test = residuals.split(0.5, torch.Generator().manual_seed(0))
    intervals = kgcp.SplitConformal(ALPHA).calibrate(cal).predict(test)
    t = intervals.thresholds
    assert t.dtype == torch.float64 and (t == t[0]).all()
    scale = score.scale(test.predicted)
    assert torch.equal(intervals.lower, (test.predicted - t * scale).clamp(0, 1))
    assert torch.equal(intervals.upper, (test.predicted + t * scale).clamp(0, 1))
    c = test.queries.confidence
    assert torch.equal(intervals.covered, (intervals.lower <= c) & (c <= intervals.upper))
    # too few calibration triples (8 < 1 / alpha - 1): all of [0, 1]
    full = kgcp.SplitConformal(ALPHA).calibrate(cal[:8]).predict(test)
    assert (full.width == 1).all() and full.covered.all()
    assert kgcp.evaluate_intervals(full, ALPHA)["full"] == 1


def test_residuals_split(toy):
    triples, predicted = toy
    residuals = kgcp.residuals(predicted, triples, kgcp.EntropyResidual())
    cal, test = residuals.split(0.25, torch.Generator().manual_seed(0))
    rows = [cal.queries.triple, test.queries.triple]
    assert (len(rows[0]), len(rows[1])) == (M // 4, 3 * M // 4)
    assert set(rows[0].tolist()) | set(rows[1].tolist()) == set(range(M))
    for part, row in zip((cal, test), rows, strict=True):
        assert torch.equal(part.predicted, predicted[row])
        assert torch.equal(part.true, residuals.true[row])


def test_residuals_of_a_model(toy):
    """A model's confidences, in batches, give the residuals of precomputed confidences."""
    triples, predicted = toy

    class Model:
        def confidence(self, batch):
            return predicted.float()[batch.triple]

    score = kgcp.EntropyResidual()
    from_model = kgcp.residuals(Model(), triples, score, batch_size=64)
    assert torch.equal(from_model.true, kgcp.residuals(predicted.float(), triples, score).true)


def test_mondrian_over_merged_relations(toy):
    triples, predicted = toy
    residuals = kgcp.residuals(predicted, triples, kgcp.EntropyResidual())
    cal, _ = residuals.split(0.5, torch.Generator().manual_seed(0))
    merged = kgcp.MergedRelations(torch.randn(R, 8), min_size=1)
    by_relation = kgcp.MondrianConformal(ALPHA).calibrate(cal)
    assert kgcp.MondrianConformal(ALPHA, merged).calibrate(cal).thresholds == by_relation.thresholds


def test_condkgcp_predicts_answer_sets_only(toy):
    triples, predicted = toy
    residuals = kgcp.residuals(predicted, triples, kgcp.AbsoluteResidual())
    with pytest.raises(TypeError):
        kgcp.CondKGCP(ALPHA, kgcp.by_relation).calibrate(residuals)


def test_evaluate_intervals(toy):
    triples, predicted = toy
    residuals = kgcp.residuals(predicted, triples, kgcp.AbsoluteResidual())
    splits = random_splits(residuals, 3)
    intervals = [kgcp.MondrianConformal(ALPHA).calibrate(cal).predict(t) for cal, t in splits]
    metrics = kgcp.evaluate_intervals(intervals, ALPHA)
    covered = torch.cat([i.covered for i in intervals]).double()
    width = torch.cat([i.width for i in intervals])
    relation = torch.cat([i.queries.relation for i in intervals])
    assert metrics["coverage"] == pytest.approx(covered.mean().item())
    assert metrics["width"] == pytest.approx(width.mean().item())
    gaps = torch.stack([(1 - ALPHA) - covered[relation == r].mean() for r in range(R)])
    widths = torch.stack([width[relation == r].mean() for r in range(R)])
    assert metrics["covgap"] == pytest.approx(gaps.abs().mean().item())
    assert metrics["ucr"] == pytest.approx((gaps > 0).double().mean().item())
    assert metrics["cond_width"] == pytest.approx(widths.mean().item())
    assert metrics["relations"] == R


def test_uncertain_triples():
    triples = torch.tensor([[0, 1, 2], [2, 0, 1]])
    kg = kgcp.KG("toy", {"test": triples}, ["a", "b", "c"], ["r", "s"])
    with pytest.raises(ValueError):
        kg.uncertain_triples("test")
    kg = kgcp.KG(
        "toy", {"test": triples}, ["a", "b", "c"], ["r", "s"], {"test": torch.tensor([0.5, 1])}
    )
    test = kg.uncertain_triples("test")
    assert test.head.tolist() == [0, 2] and test.relation.tolist() == [1, 0]
    assert test.tail.tolist() == [2, 1] and test.triple.tolist() == [0, 1]
    assert test.confidence.dtype == torch.float64 and test.confidence.tolist() == [0.5, 1]


UNKGCP = ["ukge_logi", "passleaf", "beurre"]
# UnKGCP's Table 1 on PPI5k (mean of 10 models): model -> score -> coverage, width. Its UKGE
# values are those of the ukge_rect models (docs/prediction_intervals.md), not ukge_logi.
TABLE_1 = {
    "passleaf": {"absolute": (0.90, 0.21), "entropy": (0.90, 0.20)},
    "beurre": {"absolute": (0.90, 0.25), "entropy": (0.90, 0.26)},
}


def load_unkgcp(model: str) -> kgcp.LibKGE:
    path = ROOT / "models" / "unkgcp" / f"ppi5k-{model}" / "seed0.pt"
    if not path.exists():
        pytest.skip(f"no converted model {path} (see third_party/unkge/README.md)")
    return kgcp.LibKGE(path, device="cpu")


@pytest.mark.parametrize("name", UNKGCP)
def test_unkgcp_model(name):
    """The confidences of the dataset, and the paper's protocol: calibrated on the
    validation triples, the intervals of the test triples are those of Table 1."""
    model = load_unkgcp(name)
    kg = model.kg
    assert {s: len(c) for s, c in kg.confidences.items()} == {
        s: len(kg.split(s)) for s in ("train", "valid", "test")
    }
    valid, test = kg.uncertain_triples("valid"), kg.uncertain_triples("test")
    assert ((test.confidence > 0) & (test.confidence <= 1)).all()
    for score_name, score in zip(("absolute", "entropy"), SCORES, strict=True):
        cp = kgcp.SplitConformal(ALPHA).calibrate(kgcp.residuals(model, valid, score))
        metrics = kgcp.evaluate_intervals(cp.predict(kgcp.residuals(model, test, score)), ALPHA)
        assert metrics["coverage"] == pytest.approx(1 - ALPHA, abs=0.01)
        if name in TABLE_1:  # one model; the table's are means over 10
            coverage, width = TABLE_1[name][score_name]
            assert metrics["coverage"] == pytest.approx(coverage, abs=0.01)
            assert metrics["width"] == pytest.approx(width, abs=0.015)
