"""Conformal guarantees and edge cases, on a small synthetic graph."""

import math

import pytest
import torch

import kgcp

E, R, M, ALPHA = 20, 4, 300, 0.1
SCORES = [kgcp.Neg(), kgcp.MinMax(), kgcp.Softmax(), kgcp.APS(), kgcp.RAPS(), kgcp.Rank()]
CONTINUOUS = (kgcp.Neg, kgcp.Softmax, kgcp.APS)  # no ties: coverage is also bounded above


def random_triples(m, generator):
    return torch.stack(
        [
            torch.randint(E, (m,), generator=generator),
            torch.randint(R, (m,), generator=generator),
            torch.randint(E, (m,), generator=generator),
        ],
        dim=1,
    )


@pytest.fixture(scope="module")
def toy():
    """A graph, its test queries, and logits that favor the targets."""
    generator = torch.Generator().manual_seed(0)
    kg = kgcp.KG(
        name="toy",
        splits={"train": random_triples(500, generator), "test": random_triples(M, generator)},
        entities=[f"e{i}" for i in range(E)],
        relations=[f"r{i}" for i in range(R)],
    )
    queries = kgcp.Queries.from_triples(kg.split("test"))
    logits = torch.randn(len(queries), E, generator=generator)
    # the targets' advantage grows with the relation id, so coverage varies per relation
    logits[torch.arange(len(queries)), queries.target] += 2 + queries.relation
    return kg, queries, logits


def random_splits(scores, n, fraction=0.5):
    return [scores.split(fraction, torch.Generator().manual_seed(i)) for i in range(n)]


def test_conformal_quantile():
    scores = torch.arange(1.0, 20.0)  # 1, ..., 19
    assert kgcp.conformal_quantile(scores, 0.05) == 19  # ceil(20 * 0.95) = 19
    assert kgcp.conformal_quantile(scores[:9], 0.1) == 9  # ceil(10 * 0.9) = 9
    assert kgcp.conformal_quantile(scores[:8], 0.1) == math.inf  # ceil(8.1) = 9 > 8
    assert kgcp.conformal_quantile(scores[:0], 0.1) == math.inf


@pytest.mark.parametrize("score", SCORES, ids=lambda s: type(s).__name__)
def test_marginal_coverage(toy, score):
    """Expected coverage is exactly k / (n + 1), k = ceil((n + 1)(1 - alpha)), without ties.

    A small calibration set (n = 20) makes an off-by-one in k visible (-0.05).
    """
    kg, queries, logits = toy
    scores = kgcp.nonconformity(logits, queries, score, known=kg.known("train"))
    splits = random_splits(scores, 1000, fraction=10 / M)
    sets = (kgcp.SplitConformal(ALPHA).calibrate(cal).predict(test) for cal, test in splits)
    coverage = kgcp.evaluate(sets, ALPHA)["coverage"]
    n = len(splits[0][0])
    expected = math.ceil((n + 1) * (1 - ALPHA)) / (n + 1)
    assert coverage >= expected - 0.01
    if isinstance(score, CONTINUOUS):
        assert coverage <= expected + 0.01


def relation_coverage(scores, groups):
    """Coverage per relation, pooled over random splits."""
    cp = kgcp.SplitConformal(ALPHA, groups)
    covered, relation = [], []
    for cal, test in random_splits(scores, 300):
        covered.append(cp.calibrate(cal).predict(test).covered)
        relation.append(test.queries.relation)
    covered, relation = torch.cat(covered).float(), torch.cat(relation)
    return torch.stack([covered[relation == r].mean() for r in range(R)])


@pytest.mark.parametrize("score", SCORES, ids=lambda s: type(s).__name__)
def test_relation_conditional_coverage(toy, score):
    kg, queries, logits = toy
    scores = kgcp.nonconformity(logits, queries, score, known=kg.known("train"))
    assert relation_coverage(scores, kgcp.by_relation).min() >= 1 - ALPHA - 0.01
    # marginal calibration under-covers the relation with the weakest signal
    assert relation_coverage(scores, kgcp.marginal).min() < 1 - ALPHA - 0.02


def test_too_small_group_gets_full_set(toy):
    kg, queries, logits = toy
    scores = kgcp.nonconformity(logits, queries, kgcp.Softmax(), known=kg.known("train"))
    cp = kgcp.SplitConformal(ALPHA).calibrate(scores[:8])  # 8 < 1 / alpha - 1
    sets = cp.predict(scores)
    assert torch.equal(sets.mask, scores.values.isfinite())


def test_known_answers(toy):
    kg, queries, logits = toy
    known = kg.known("train", "test")  # includes every target
    mask = known(queries)
    answers = {}
    for s, p, o in kg.split("train", "test").tolist():
        answers.setdefault((s, p, kgcp.TAIL), set()).add(o)
        answers.setdefault((o, p, kgcp.HEAD), set()).add(s)
    for i in range(len(queries)):
        key = (queries.anchor[i].item(), queries.relation[i].item(), queries.direction[i].item())
        expected = answers[key] - {queries.target[i].item()}
        assert set(mask[i].nonzero().flatten().tolist()) == expected

    scores = kgcp.nonconformity(logits, queries, kgcp.Softmax(), known=known)
    assert scores.true.isfinite().all()  # targets are never filtered
    assert torch.equal(scores.values.isinf(), mask)
    sets = kgcp.SplitConformal(ALPHA).calibrate(scores).predict(scores)
    assert not (sets.mask & mask).any()


def test_aps():
    logits = torch.tensor([[2.0, 0.0, 1.0, -torch.inf]])
    probs = logits.softmax(1)[0]
    scores = kgcp.APS(randomized=False)(logits, torch.Generator())[0]
    expected = [probs[0], probs[0] + probs[2] + probs[1], probs[0] + probs[2], 1.0]
    assert torch.allclose(scores, torch.tensor(expected))


def test_raps_without_penalty_is_aps(toy):
    _, _, logits = toy
    aps = kgcp.APS()(logits, torch.Generator().manual_seed(1))
    raps = kgcp.RAPS(lam=0.0)(logits, torch.Generator().manual_seed(1))
    assert torch.equal(aps, raps)


def test_rank_gives_top_k(toy):
    kg, queries, logits = toy
    known = kg.known("train")
    scores = kgcp.nonconformity(logits, queries, kgcp.Rank(), known=known)
    cal, _ = scores.split(0.5, torch.Generator().manual_seed(0))
    cp = kgcp.SplitConformal(ALPHA).calibrate(cal)
    k = int(cp.thresholds[0])
    filtered = logits.masked_fill(known(queries), -torch.inf)
    top_k = filtered.topk(k, dim=1).indices
    sets = cp.predict(scores)
    expected = torch.zeros_like(sets.mask).scatter_(1, top_k, True) & filtered.isfinite()
    assert torch.equal(sets.mask, expected)


def test_split(toy):
    kg, queries, logits = toy
    scores = kgcp.nonconformity(logits, queries, kgcp.Neg())
    cal, test = scores.split(0.25, torch.Generator().manual_seed(0))
    rows = [q.direction * M + q.triple for q in (cal.queries, test.queries)]  # rows in `scores`
    assert (len(set(rows[0].tolist())), len(set(rows[1].tolist()))) == (M // 2, 3 * M // 2)
    assert set(rows[0].tolist()) | set(rows[1].tolist()) == set(range(2 * M))
    assert torch.equal(cal.values, scores.values[rows[0]])
    assert cal.matrix is scores.matrix and test.matrix is scores.matrix  # no copies
