# Prediction intervals for uncertain knowledge graphs

In an uncertain knowledge graph, every triple (h, r, t) has a confidence c ∈ [0, 1], and
a model predicts it: M. Conformal prediction turns this point prediction into an interval
that contains the true confidence with probability at least 1 − α. This is the task of
UnKGCP (Zhu, Wu et al., "Certainty in Uncertainty: Reasoning over Uncertain Knowledge
Graphs with Statistical Guarantees", EMNLP 2025,
[arXiv:2510.24754](https://arxiv.org/abs/2510.24754)). This document describes the method
in `kgcp`, the protocol and the results for the paper's models: UKGE, PASSLEAF and BEUrRE
on CN15k, NL27k and PPI5k, converted to LibKGE in
[third_party/unkge](../third_party/unkge/README.md). For usage, see the
[README](../README.md#intervals-for-uncertain-knowledge-graphs).

## Results

Split conformal prediction with two residual scores, CP (|c − M|) and UnKGCP
(|c − M| / H(M)), calibrated marginally (as in the paper) or per relation. Each dataset
and model has 5 models (seeds 0–4 of the authors' 10). Each is calibrated on a random
half of the test triples and evaluated on the other half, over 100 random splits per
seed. Metrics are pooled over the splits, then averaged over the 5 seeds. The test
triples are those of the paper's Table 1: the datasets' test files, without negatives.

### Coverage

Marginal coverage is 0.899–0.901 for every model, dataset and score at α = 0.1
(0.949–0.951 at α = 0.05, 0.798–0.802 at α = 0.2). Per-relation calibration covers
0.900–0.902 on CN15k and PPI5k, and 0.912–0.914 on NL27k: many of NL27k's relations have
fewer than 9 calibration triples, too few for a threshold at α = 0.1, so their triples
get the whole interval [0, 1] (9–17% of the triples).

### Width

Mean interval width at α = 0.1, marginal calibration (± std over seeds ≤ 0.016):

| Dataset | Model | CP | UnKGCP | UnKGCP intervals that are all of [0, 1] |
|---|---|---|---|---|
| CN15k | UKGE | 0.845 | 0.917 | 83% |
| CN15k | PASSLEAF | 0.861 | 0.926 | 79% |
| CN15k | BEUrRE | 0.864 | 0.893 | 61% |
| NL27k | UKGE | 0.452 | **0.370** | 0% |
| NL27k | PASSLEAF | 0.542 | **0.435** | 0% |
| NL27k | BEUrRE | 0.426 | **0.389** | 0% |
| PPI5k | UKGE | 0.184 | 0.183 | 0% |
| PPI5k | PASSLEAF | 0.202 | 0.198 | 0% |
| PPI5k | BEUrRE | 0.249 | 0.257 | 0% |

- **NL27k:** UnKGCP's intervals are 9–20% narrower than CP's at the same coverage. The
  gain grows at α = 0.05 (UKGE 0.776 → 0.611, PASSLEAF 0.829 → 0.607) and vanishes at
  α = 0.2.
- **PPI5k:** both scores give the same widths (within 0.01).
- **CN15k:** the models are inaccurate (test MAE 0.28–0.41, see
  [third_party/unkge](../third_party/unkge/README.md#verification)), so both methods give
  wide intervals. UnKGCP's are wider, and 61–83% of them are all of [0, 1] (CP: 18–22%).
  The paper's code computes the entropy differently on CN15k (see
  [Comparison with the paper](#comparison-with-the-paper)).

### Per-relation calibration

At α = 0.1 with UnKGCP's score, marginal calibration (split) vs per relation (Mondrian),
over the relations with at least 10 test triples (CN15k 29, NL27k 136, PPI5k 7):

| Dataset | Model | CovGap split → Mondrian | UCR split → Mondrian | Width split → Mondrian |
|---|---|---|---|---|
| CN15k | UKGE | 0.081 → 0.018 | 0.57 → 0.12 | 0.917 → 0.912 |
| CN15k | PASSLEAF | 0.103 → 0.017 | 0.66 → 0.13 | 0.926 → 0.915 |
| CN15k | BEUrRE | 0.039 → 0.017 | 0.48 → 0.08 | 0.893 → 0.888 |
| NL27k | UKGE | 0.059 → 0.041 | 0.42 → 0.04 | 0.370 → 0.419 |
| NL27k | PASSLEAF | 0.062 → 0.041 | 0.35 → 0.05 | 0.435 → 0.417 |
| NL27k | BEUrRE | 0.056 → 0.041 | 0.45 → 0.05 | 0.389 → 0.453 |
| PPI5k | UKGE | 0.059 → 0.001 | 0.29 → 0.20 | 0.183 → 0.174 |
| PPI5k | PASSLEAF | 0.057 → 0.001 | 0.29 → 0.26 | 0.198 → 0.179 |
| PPI5k | BEUrRE | 0.035 → 0.001 | 0.43 → 0.31 | 0.257 → 0.258 |

Marginal calibration under-covers 29–66% of the relations, by 0.02 to 0.13 on average
(UCG). Per-relation calibration removes the shortfall (UCG ≤ 0.002; the
remaining UCR counts relations a hair below 0.9), and on CN15k and PPI5k its intervals are
as narrow or narrower. On NL27k its CovGap comes from over-coverage: the full intervals
of the small relations. CP's score gives the same picture.

### Comparison with the paper

The paper's protocol (`--calibration valid`): calibrated on all validation triples,
evaluated on all test triples, mean over seeds 0–4, against Table 1 (mean over 10 seeds).
Coverage is 0.895–0.903 (the paper: 0.90–0.91). Width, ours (Table 1):

| Dataset | Model | CP | UnKGCP |
|---|---|---|---|
| CN15k | UKGE | 0.844 (0.88)¹ | 0.914 (0.82)¹ ² |
| CN15k | PASSLEAF | 0.860 (0.86) | 0.926 (0.84)² |
| CN15k | BEUrRE | 0.863 (0.86) | 0.892 (0.81)² |
| NL27k | UKGE | 0.445 (0.27)¹ | 0.361 (0.43)¹ |
| NL27k | PASSLEAF | 0.540 (0.54) | 0.435 (0.44) |
| NL27k | BEUrRE | 0.416 (0.42) | 0.382 (0.38) |
| PPI5k | UKGE | 0.187 (0.16)¹ | 0.187 (0.16)¹ |
| PPI5k | PASSLEAF | 0.207 (0.21) | 0.203 (0.20) |
| PPI5k | BEUrRE | 0.252 (0.25) | 0.259 (0.26) |

9 of the 18 widths agree to the rounding, and a tenth is within the seeds' spread (NL27k
PASSLEAF UnKGCP: 0.4347 ± 0.004, Table 1 0.44). The other 8 have two causes:

1. **Table 1's UKGE values are those of the `ukge_rect` models.** With them, CP's widths
   are 0.879, 0.271 and 0.154 (CN15k, NL27k, PPI5k), and UnKGCP's 0.820 (with note 2),
   0.416 and 0.160: all within 0.015 of Table 1. The paper's Table 2 (test errors), on
   the other hand, matches `ukge_logi` (see [third_party/unkge](../third_party/unkge/README.md#verification)).
   These results are for `ukge_logi`, the model of Table 2.
2. **On CN15k, the authors' code takes the entropy of 0.5 M + 0.5**, not of M as in the
   paper. With it, UnKGCP's CN15k widths are 0.844 (PASSLEAF), 0.805 (BEUrRE) and 0.820
   (`ukge_rect`), as in Table 1. `kgcp` follows the paper.

The `ukge_rect` numbers come from
`scripts/intervals.py models/unkgcp/*-ukge_rect --calibration valid`; the entropy of
0.5 M + 0.5 was a one-off check, not part of `kgcp`.

## Method

**Triples.** The test triples (h, r, t) with their confidence c, from the 4th column of
the dataset's files (`kg.uncertain_triples("test")`). The model's prediction is
M = `LibKGE.confidence(triples)`: the sigmoid of UKGE's and PASSLEAF's score, the
probability of BEUrRE's boxes.

**Residual scores** (`AbsoluteResidual`, `EntropyResidual`): the nonconformity of c is
|c − M| / s(M), and the interval for threshold t is M ± t·s(M), clipped to [0, 1]:

| Method | s(M) | Interval |
|---|---|---|
| CP | 1 | M ± t |
| UnKGCP | H(M) = −M ln M − (1 − M) ln(1 − M) | M ± t·H(M) |

M is clipped to [10⁻⁶, 1 − 10⁻⁶] for the entropy, as in the authors' code, so that
H(M) > 0. The base of the logarithm does not matter: it scales t inversely.

**Calibration.** The same as for answer sets: for n calibration scores, the threshold is
the ⌈(n+1)(1−α)⌉-th smallest, or +∞ (the interval [0, 1]) if that exceeds n
(`SplitConformal`); per relation, each relation gets its own threshold
(`MondrianConformal`). `MergedRelations` works on triples too; `CondKGCP` does not,
since it calibrates ranks of answers.

### Metrics

Computed on the evaluation triples, pooled over all splits:

- **coverage**: fraction of intervals containing the confidence
- **width**: mean interval width; **full**: fraction of intervals that are all of [0, 1]
- per relation, over the relations with at least 10 distinct test triples: **CovGap**,
  **UCR**, **UCG** as for answer sets ([conformal_prediction.md](conformal_prediction.md#metrics)),
  and **cond_width**, the mean width averaged over relations

## Protocol

| Setting | Value |
|---|---|
| Models | UKGE (`ukge_logi`), PASSLEAF, BEUrRE: seeds 0–4 (`models/unkgcp/<dataset>-<model>/seed<N>.pt`, see [third_party/unkge](../third_party/unkge/README.md)) |
| Triples | the test triples (CN15k 19,293, NL27k 14,034, PPI5k 21,720), without negatives |
| Calibration | random half of the test triples; evaluation on the other half |
| Splits | 100 per seed, different for each seed and the same for the three models (500 per dataset and model) |
| α | 0.05, 0.1, 0.2 |
| Scores | CP (absolute residual), UnKGCP (entropy-scaled residual) |
| Groupings | marginal, per relation |
| Relation metrics | relations with at least 10 test triples |
| Reported | metrics pooled over the splits; mean (± std) over the 5 seeds |

Random halves of the test triples are exchangeable by construction. The paper calibrates
on the validation split instead, on which UKGE's training also stops early;
`--calibration valid` reproduces it. The full results, one row per (seed, score,
grouping, α), are in [results/conformal/unkgcp/](../results/conformal/unkgcp/):
`<dataset>-<model>-test.csv`, and `-valid.csv` for validation calibration. Both runs
together take about 3 minutes on a laptop CPU.

## Differences from the authors' code

- **Calibration data:** random halves of the test triples, as for answer sets; the paper
  calibrates on the validation triples (`--calibration valid`).
- **Negatives:** only the test triples, as in Table 1. The authors' code also calibrates
  on random negatives (confidence 0), separately from the positives or together with
  them. Choosing the threshold by whether a triple is a negative needs its true label,
  which a new triple does not have.
- **CN15k:** the entropy of M, as in the paper, not of 0.5 M + 0.5.
- **Coverage** is counted from the nonconformity score (at most the threshold), which is
  the same as lower ≤ c ≤ upper up to rounding.
- **Confidences:** the test files' full-precision confidences. The authors' UKGE and
  PASSLEAF scripts read them from test_with_neg.tsv, rounded to 3 digits.

## Reproduce

```bash
uv run scripts/intervals.py  # models/unkgcp/<dataset>-{ukge_logi,passleaf,beurre}/
uv run scripts/intervals.py --calibration valid
uv run pytest tests/test_intervals.py
```

The tests check the guarantee on synthetic data (expected coverage
⌈(n+1)(1−α)⌉/(n+1), per relation too), the intervals, and that the converted PPI5k
PASSLEAF and BEUrRE models give Table 1's coverage and widths with the paper's protocol.

## References

- Zhu, Wu, et al. Certainty in uncertainty: Reasoning over uncertain knowledge graphs
  with statistical guarantees. EMNLP, 2025.
- Chen, Chen, Shi, Sun, Zaniolo. Embedding uncertain knowledge graphs. AAAI, 2019 (UKGE).
- Chen, Yeh, Kuo. PASSLEAF: A pool-based semi-supervised learning framework for
  uncertain knowledge graph embedding. AAAI, 2021.
- Chen, Boratko, Chen, Dasgupta, Li, McCallum. Probabilistic box embeddings for uncertain
  knowledge graph reasoning. NAACL, 2021 (BEUrRE).
