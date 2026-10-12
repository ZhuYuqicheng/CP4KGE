# Conformal prediction

`kgcp` turns a KGE model's scores into answer sets with a coverage guarantee. For a
query (s, p, ?) or (?, p, o) and a miscoverage level α, it predicts a set of entities
that contains the true answer with probability at least 1 − α. This document describes
the method, the experimental protocol, and results for the tuned DistMult models on
Nations, UMLS and Kinship (see [hyperparameter_tuning.md](hyperparameter_tuning.md)).
For usage, see the [README](../README.md#conformal-prediction).

## Results

Split conformal prediction with six nonconformity scores and three calibration
groupings, at α ∈ {0.05, 0.1, 0.2}. Each of the 5 models per dataset is calibrated on
a random half of the test queries and evaluated on the other half, over 100 random
splits; metrics are pooled over the splits, then averaged over the 5 models.

### Coverage

Marginal coverage is on target everywhere. Range over the five continuous scores, with
Rank separately:

| Dataset | α = 0.05 (target 0.95) | α = 0.1 (target 0.90) | α = 0.2 (target 0.80) |
|---|---|---|---|
| Nations | 0.950–0.951 (Rank 0.960) | 0.901–0.903 (Rank 0.921) | 0.803–0.804 (Rank 0.850) |
| UMLS | 0.949–0.950 (Rank 0.960) | 0.900–0.901 (Rank 0.917) | 0.799–0.801 (Rank 0.840) |
| Kinship | 0.950–0.951 (Rank 0.954) | 0.900–0.901 (Rank 0.910) | 0.799–0.800 (Rank 0.819) |

The expected coverage of split CP with n calibration scores is ⌈(n+1)(1−α)⌉/(n+1)
(e.g. 0.901 on Nations at α = 0.1, n = 201), slightly above 1 − α. The Monte Carlo
error of these means (500 splits) is about ±0.001. Rank over-covers because ranks are
integers: the calibrated k cannot be fractional, so the top-k sets overshoot.

### Set size

Mean set size at α = 0.05 / 0.1 / 0.2, marginal calibration (number of entities in
parentheses):

| Score | Nations (14) | UMLS (135) | Kinship (104) |
|---|---|---|---|
| Neg | 7.22 / 6.01 / 4.34 | 43.63 / 22.02 / 8.73 | 15.89 / 13.99 / 11.06 |
| MinMax | 4.81 / 3.64 / 2.43 | 3.90 / 3.10 / 2.35 | 6.78 / 5.06 / 3.42 |
| Softmax | 4.68 / **3.33** / 2.17 | 3.61 / **2.95** / 2.19 | 7.22 / 5.17 / 3.19 |
| APS | 4.71 / 3.74 / 2.58 | 8.86 / 5.81 / 3.67 | 8.83 / 6.36 / 4.28 |
| RAPS | 4.71 / 3.69 / 2.56 | 6.67 / 4.95 / 3.42 | 8.53 / 6.15 / 4.18 |
| Rank | 5.51 / 4.30 / 3.00 | 6.07 / 4.43 / 3.00 | 11.05 / 7.02 / 4.04 |

Details at α = 0.1, marginal calibration (size: mean ± std over the 5 models; the std
of coverage is at most 0.007 everywhere):

| Dataset | Score | Coverage | Size | Singletons | Empty | CovGap | UCR |
|---|---|---|---|---|---|---|---|
| Nations | Neg | 0.901 | 6.01 ± 0.21 | 0.042 | 0.050 | 0.115 | 0.400 |
| Nations | MinMax | 0.902 | 3.64 ± 0.20 | 0.164 | 0.000 | 0.078 | 0.467 |
| Nations | Softmax | 0.903 | 3.33 ± 0.09 | 0.157 | 0.000 | 0.100 | 0.307 |
| Nations | APS | 0.901 | 3.74 ± 0.07 | 0.111 | 0.002 | 0.094 | 0.347 |
| Nations | RAPS | 0.901 | 3.69 ± 0.04 | 0.103 | 0.002 | 0.093 | 0.347 |
| Nations | Rank | 0.921 | 4.30 ± 0.21 | 0.000 | 0.000 | 0.087 | 0.293 |
| UMLS | Neg | 0.900 | 22.02 ± 1.59 | 0.050 | 0.082 | 0.102 | 0.215 |
| UMLS | MinMax | 0.901 | 3.10 ± 0.06 | 0.220 | 0.000 | 0.070 | 0.363 |
| UMLS | Softmax | 0.900 | 2.95 ± 0.03 | 0.169 | 0.006 | 0.076 | 0.333 |
| UMLS | APS | 0.900 | 5.81 ± 0.26 | 0.108 | 0.011 | 0.061 | 0.393 |
| UMLS | RAPS | 0.900 | 4.95 ± 0.14 | 0.105 | 0.011 | 0.063 | 0.393 |
| UMLS | Rank | 0.917 | 4.43 ± 0.20 | 0.000 | 0.000 | 0.075 | 0.244 |
| Kinship | Neg | 0.900 | 13.99 ± 0.67 | 0.036 | 0.038 | 0.097 | 0.133 |
| Kinship | MinMax | 0.900 | 5.06 ± 0.13 | 0.121 | 0.000 | 0.056 | 0.238 |
| Kinship | Softmax | 0.901 | 5.17 ± 0.15 | 0.037 | 0.001 | 0.077 | 0.190 |
| Kinship | APS | 0.901 | 6.36 ± 0.13 | 0.047 | 0.002 | 0.042 | 0.295 |
| Kinship | RAPS | 0.900 | 6.15 ± 0.13 | 0.040 | 0.002 | 0.047 | 0.257 |
| Kinship | Rank | 0.910 | 7.02 ± 0.10 | 0.000 | 0.000 | 0.083 | 0.162 |

Singletons and Empty are the fractions of sets with one and zero entities. CovGap and
UCR measure per-relation coverage (see [metrics](#metrics)); they use the relations with
at least 10 test queries: 15 of 41 test relations on Nations, 27 of 36 on UMLS, 21 of
23 on Kinship.

### Per-relation and per-direction calibration

Marginal calibration covers 90% on average, but not for every relation: with Softmax
and APS, 19–39% of the relations are under-covered (UCR), and per-relation coverage is
off target by 0.04–0.10 on average (CovGap). Calibrating per relation fixes this where
relations have enough calibration data, at the cost of larger sets. At α = 0.1:

| Dataset | Groups | Softmax: coverage | size | CovGap | UCR | APS: coverage | size | CovGap | UCR |
|---|---|---|---|---|---|---|---|---|---|
| Nations | marginal | 0.903 | 3.33 | 0.100 | 0.307 | 0.901 | 3.74 | 0.094 | 0.347 |
| Nations | direction | 0.905 | 3.41 | 0.097 | 0.320 | 0.903 | 3.81 | 0.091 | 0.360 |
| Nations | relation | 0.964 | 7.17 | 0.060 | 0.000 | 0.964 | 7.21 | 0.060 | 0.013 |
| UMLS | marginal | 0.900 | 2.95 | 0.076 | 0.333 | 0.900 | 5.81 | 0.061 | 0.393 |
| UMLS | direction | 0.901 | 2.98 | 0.076 | 0.356 | 0.901 | 5.87 | 0.062 | 0.385 |
| UMLS | relation | 0.922 | 17.23 | 0.043 | 0.030 | 0.923 | 19.78 | 0.043 | 0.000 |
| Kinship | marginal | 0.901 | 5.17 | 0.077 | 0.190 | 0.901 | 6.36 | 0.042 | 0.295 |
| Kinship | direction | 0.901 | 5.26 | 0.077 | 0.181 | 0.901 | 6.36 | 0.042 | 0.267 |
| Kinship | relation | 0.910 | 6.24 | 0.014 | 0.067 | 0.909 | 7.48 | 0.013 | 0.057 |

- **Kinship** has enough data per relation (about 47 calibration queries on average):
  per-relation calibration cuts CovGap from 0.077 to 0.014 for Softmax, and sets grow
  by only 21%. The relations still counted as under-covered miss by 0.004 on average
  (UCG), which is within sampling noise.
- **Nations and UMLS** have many rare relations. A relation with fewer than
  1/α − 1 calibration queries (9 at α = 0.1) gets full sets, which is the only valid
  choice. This affects about half of the Nations test queries (33 of 41 relations) and
  10% of the UMLS test queries (17 of 36 relations). Coverage then overshoots (0.96 and
  0.92), and sets double on Nations and grow sixfold on UMLS. Handling rare relations
  is what CondKGCP is for (not implemented yet).
- **Per-direction calibration** equalizes head and tail queries, which marginal
  calibration treats unequally: with Softmax, tail queries are covered more often than
  head queries (Nations 0.917 vs 0.888, UMLS 0.921 vs 0.880, Kinship 0.927 vs 0.875);
  per direction, both reach 0.90–0.905, with sets 1–3% larger. With APS, the imbalance
  is smaller and reversed on UMLS and Kinship.

### Calibrating on the validation split

As in earlier work, calibrate on the validation queries and evaluate on all test queries
(`--calibration valid`; known answers then come from train only). Range of marginal
coverage over the six scores:

| Dataset | α = 0.05 (target 0.95) | α = 0.1 (target 0.90) | α = 0.2 (target 0.80) |
|---|---|---|---|
| Nations | 0.909–0.955 | 0.857–0.903 | 0.736–0.784 |
| UMLS | 0.949–0.962 | 0.902–0.925 | 0.805–0.842 |
| Kinship | 0.949–0.961 | 0.888–0.905 | 0.786–0.824 |

Nations is under-covered for most scores at every α, by up to 0.064, and Kinship for
some. This is a single fixed split, so part of the deviation is chance (the sampling
error of one split is about ±0.02 on Nations). But validation and test queries are
not exchangeable to begin with: the validation split was used for early stopping and
model selection. Random halves of the test split avoid both problems and allow
averaging over many splits.

### Observations

- **Coverage holds** for every score, grouping, dataset and α when calibrating on test
  halves, as the theory guarantees.
- **Softmax gives the smallest sets** on Nations and UMLS (3.33 and 2.95 at α = 0.1);
  on Kinship, MinMax (5.06) and Softmax (5.17) are close. This matches the known result
  that 1 − softmax (LAC) minimizes the expected set size when the probabilities are
  accurate (Sadinle et al., 2019).
- **APS trades size for adaptivity.** Its sets are larger than Softmax's (+12% on
  Nations, +23% on Kinship, +97% on UMLS), but its per-relation coverage is more even
  on UMLS and Kinship (CovGap 0.061 vs 0.076, 0.042 vs 0.077). RAPS sits between the
  two.
- **Raw logits (Neg) are a poor score.** DistMult logits are not comparable across
  queries, so one global threshold gives large sets (22 of 135 entities on UMLS) and
  many empty sets (4–8%).
- **Rank (conformal top-k)** gives every query the same set size k (fewer only if
  fewer candidates remain after filtering). That is easy to interpret, but it wastes
  size on easy queries and never yields singletons.
- **Per-relation calibration** needs data: it works well on Kinship, but on Nations and
  UMLS rare relations fall back to full sets.

## Method

**Queries.** Every triple (s, p, o) gives a tail query (s, p, ?) with target o and a
head query (?, p, o) with target s. The candidates are all entities.

**Filtering.** Answers known from splits outside the calibration and evaluation data
(here train + valid; train only when calibrating on valid) are removed from the
candidates before scoring, for calibration and test queries alike. The target is never
removed. So the scores are filtered: softmax and APS normalize over the remaining
candidates, and Rank is the filtered rank. Other answers of the same query in the
calibration or evaluation data stay candidates; they are true facts, so including them
in a set is correct.

**Nonconformity scores** (lower = more plausible), computed from the model's logits x
over the candidates of a query, with p = softmax(x):

| Score | Nonconformity of candidate e | Reference |
|---|---|---|
| Neg | −x_e | |
| MinMax | (max x − x_e) / (max x − min x) | |
| Softmax | 1 − p_e | LAC; Sadinle et al., 2019 |
| APS | mass of the candidates ranked above e, plus u · p_e; u ~ U(0, 1) per query | Romano et al., 2020 |
| RAPS | APS + λ · max(0, rank(e) − k_reg), with λ = 0.01, k_reg = 5 | Angelopoulos et al., 2021 |
| Rank | rank of e among the candidates (1 = best): conformal top-k | |

**Calibration.** For n calibration scores of the targets, the threshold is the
⌈(n+1)(1−α)⌉-th smallest, or +∞ (the full set) if that exceeds n. A query's set holds
the candidates scoring at most the threshold. With a grouping, each group (relation,
or direction) gets its own threshold from its own calibration queries (Mondrian CP), so
coverage holds within every group.

**Calibration splits.** Test queries, not triples, are split at random. A triple's head
and tail queries are correlated (correlation 0.59 for Softmax on Nations). If both
always fell on the same side, the guarantee would weaken to ⌈(n+1)(1−α)⌉/(n+2): 0.896
instead of 0.901 on Nations at α = 0.1. Over 2,000 Nations splits, coverage was about
0.001 lower with triple splits than with query splits (Softmax and APS).

### Metrics

Computed on the evaluation queries, pooled over all splits (so a relation's coverage is
its average over the splits):

- **coverage**: fraction of sets containing the target; also per direction
  (`coverage_tail`, `coverage_head`)
- **size**: mean set size; also per direction; **singleton** and **empty**: fractions
  of sets of size 1 and 0
- per relation, over the relations with at least 10 distinct test queries:
  - **CovGap**: mean |coverage − (1 − α)|
  - **UCR** (under-coverage ratio): fraction of relations with coverage below 1 − α
  - **UCG** (under-coverage gap): mean shortfall of those relations (0 if none)
  - **cond_size**: mean set size, averaged over relations

## Protocol

| Setting | Value |
|---|---|
| Models | DistMult, the selected configuration of each dataset, 5 seeds (`runs/hpo/final/<dataset>-distmult/top<rank>/seed{1..5}`) |
| Calibration | random half of the test queries; evaluation on the other half |
| Splits | 100 per model, different for each model (500 per dataset) |
| Filtering | train + valid |
| α | 0.05, 0.1, 0.2 |
| Scores | Neg, MinMax, Softmax, APS, RAPS, Rank |
| Groupings | marginal, per relation, per direction |
| Relation metrics | relations with at least 10 test queries |
| Reported | metrics pooled over the splits; mean (± std) over the 5 models |

The full results, one row per (model, score, grouping, α), are in
[results/conformal/](../results/conformal/): `<dataset>-distmult-test.csv`, and
`<dataset>-distmult-valid.csv` for validation calibration. The whole grid runs in under
a minute on a laptop CPU.

## Differences from the earlier kgcp code

- **Filtering**: the earlier code computed the target's score without filtering and
  removed known answers only from the final sets. Here, known answers are removed before
  scoring, for calibration and test alike. Both are valid; they define different scores
  (e.g. raw vs filtered rank).
- **Quantile**: the earlier `calc_threshold` matches the exact quantile except for
  groups with fewer than 1/α − 1 calibration queries, where it returned the most extreme
  score instead of the full set. That breaks the guarantee for rare relations.
- **UCR**: a relation counts as under-covered only if its coverage is strictly below
  1 − α. The earlier code also counted coverage exactly at 1 − α.

## Reproduce

```bash
uv run scripts/conformal.py --datasets nations umls kinship
uv run scripts/conformal.py --datasets nations umls kinship --calibration valid
uv run pytest
```

The tests check the guarantee on synthetic data (expected coverage
⌈(n+1)(1−α)⌉/(n+1), per group too), filtering against a brute-force reference, the
scores, and that the LibKGE backend reproduces LibKGE's own filtered test MRR exactly.

## References

- Sadinle, Lei, Wasserman. Least ambiguous set-valued classifiers with bounded error
  levels. JASA, 2019.
- Romano, Sesia, Candès. Classification with valid and adaptive coverage. NeurIPS, 2020.
- Angelopoulos, Bates, Malik, Jordan. Uncertainty sets for image classifiers using
  conformal prediction. ICLR, 2021.
- Angelopoulos, Bates. A gentle introduction to conformal prediction and
  distribution-free uncertainty quantification. 2021.
