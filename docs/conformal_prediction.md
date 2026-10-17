# Conformal prediction

`kgcp` turns a KGE model's scores into answer sets with a coverage guarantee. For a
query (s, p, ?) or (?, p, o) and a miscoverage level α, it predicts a set of entities
that contains the true answer with probability at least 1 − α. This document describes
the method, the experimental protocol, and results for the tuned DistMult, ComplEx and
RESCAL models on Nations, UMLS and Kinship (see
[hyperparameter_tuning.md](hyperparameter_tuning.md)). For usage, see the
[README](../README.md#conformal-prediction).

## Results

Split conformal prediction with six nonconformity scores and three calibration
groupings, at α ∈ {0.05, 0.1, 0.2}. Each dataset and model has 5 trained models (seeds).
Each is calibrated on a random half of the test queries and evaluated on the other half,
over 100 random splits per seed; the three models share these splits. Metrics are pooled
over the splits, then averaged over the 5 seeds.

### Coverage

Marginal coverage is on target for every model. Range over the five continuous scores,
with Rank separately:

| Dataset | Model | α = 0.05 (target 0.95) | α = 0.1 (target 0.90) | α = 0.2 (target 0.80) |
|---|---|---|---|---|
| Nations | DistMult | 0.949–0.951 (Rank 0.961) | 0.901–0.903 (Rank 0.922) | 0.803–0.806 (Rank 0.853) |
| Nations | ComplEx | 0.949–0.950 (Rank 0.959) | 0.898–0.900 (Rank 0.917) | 0.799–0.800 (Rank 0.831) |
| Nations | RESCAL | 0.949–0.966 (Rank 0.958) | 0.899–0.901 (Rank 0.917) | 0.800–0.802 (Rank 0.834) |
| UMLS | DistMult | 0.949–0.950 (Rank 0.958) | 0.900–0.901 (Rank 0.922) | 0.800–0.801 (Rank 0.830) |
| UMLS | ComplEx | 0.949–0.951 (Rank 0.957) | 0.900–0.901 (Rank 0.918) | 0.800–0.802 (Rank 0.846) |
| UMLS | RESCAL | 0.949–0.950 (Rank 0.962) | 0.899–0.901 (Rank 0.916) | 0.801–0.802 (Rank 0.842) |
| Kinship | DistMult | 0.950–0.951 (Rank 0.953) | 0.900 (Rank 0.909) | 0.799–0.800 (Rank 0.819) |
| Kinship | ComplEx | 0.950–0.951 (Rank 0.957) | 0.900–0.901 (Rank 0.931) | 0.799–0.800 (Rank 0.875) |
| Kinship | RESCAL | 0.950–0.951 (Rank 0.956) | 0.899–0.900 (Rank 0.911) | 0.799–0.800 (Rank 0.863) |

The expected coverage of split CP with n calibration scores is ⌈(n+1)(1−α)⌉/(n+1)
(e.g. 0.901 on Nations at α = 0.1, n = 201), slightly above 1 − α. The Monte Carlo
error of these means (500 splits) is about ±0.0013 on Nations (the coverage of a single
split varies by ±0.03) and smaller on UMLS and Kinship. The lowest value, 0.898 for
ComplEx with Softmax on Nations, is 0.901 with 1,000 splits per seed. Two scores
over-cover because of ties:

- **Rank**: ranks are integers, so the calibrated k cannot be fractional, and the top-k
  sets overshoot, by up to 0.075 (Kinship ComplEx at α = 0.2).
- **MinMax on Nations RESCAL at α = 0.05** (0.966): the lowest-scored candidate always
  has MinMax score 1, and RESCAL scores 4.5% of the Nations targets lowest among their
  candidates (8.7 per query on average, after filtering). In splits where at least 10
  of the 201 calibration targets score 1, the threshold is 1 and every set is full.

### Set size

Mean set size at α = 0.05 / 0.1 / 0.2, marginal calibration (number of entities in
parentheses). In bold, the smallest at α = 0.1 for each model:

| Dataset | Score | DistMult | ComplEx | RESCAL |
|---|---|---|---|---|
| Nations (14) | Neg | 7.02 / 5.93 / 4.12 | 5.66 / 4.78 / 3.72 | 6.61 / 5.21 / 3.76 |
| | MinMax | 4.69 / 3.48 / 2.33 | 5.25 / 4.02 / 2.73 | 7.89 / 5.98 / 4.07 |
| | Softmax | 4.49 / **3.24** / 2.17 | 5.11 / **3.77** / 2.48 | 6.52 / **5.08** / 3.14 |
| | APS | 4.75 / 3.57 / 2.53 | 5.25 / 4.11 / 2.87 | 6.53 / 5.34 / 3.84 |
| | RAPS | 4.66 / 3.53 / 2.52 | 5.26 / 4.11 / 2.87 | 6.57 / 5.30 / 3.82 |
| | Rank | 5.54 / 4.25 / 2.99 | 6.11 / 4.77 / 3.19 | 7.11 / 5.61 / 4.00 |
| UMLS (135) | Neg | 29.73 / 17.16 / 10.12 | 4.85 / 3.64 / 2.79 | 3.98 / 3.49 / 2.82 |
| | MinMax | 4.58 / 3.22 / 2.42 | 3.69 / **2.94** / 2.29 | 3.32 / 2.89 / 2.38 |
| | Softmax | 3.84 / **3.12** / 2.29 | 3.75 / 3.10 / 2.44 | 3.37 / **2.88** / 2.20 |
| | APS | 8.75 / 6.32 / 4.04 | 16.42 / 10.36 / 5.50 | 10.51 / 7.20 / 4.17 |
| | RAPS | 6.38 / 5.09 / 3.64 | 10.99 / 7.54 / 4.56 | 7.58 / 5.71 / 3.69 |
| | Rank | 6.44 / 4.86 / 3.02 | 6.23 / 4.58 / 3.00 | 5.96 / 4.32 / 3.00 |
| Kinship (104) | Neg | 14.64 / 12.52 / 9.44 | 3.95 / 2.82 / 2.04 | 5.50 / 3.56 / 2.33 |
| | MinMax | 7.40 / 5.43 / 3.60 | 3.54 / **2.58** / 1.89 | 4.70 / **3.12** / 2.12 |
| | Softmax | 7.60 / **5.34** / 3.36 | 3.71 / 2.64 / 1.93 | 5.37 / 3.37 / 2.21 |
| | APS | 9.54 / 6.57 / 4.45 | 8.20 / 5.21 / 3.01 | 9.33 / 5.71 / 3.25 |
| | RAPS | 9.18 / 6.34 / 4.35 | 6.95 / 4.75 / 2.94 | 7.86 / 5.14 / 3.12 |
| | Rank | 11.68 / 7.40 / 4.38 | 5.17 / 3.92 / 2.85 | 6.65 / 4.09 / 3.00 |

Details at α = 0.1, marginal calibration, for the scores with the smallest sets and for
APS (size: mean ± std over the 5 seeds; the std of coverage is at most 0.004 here; all
scores are in [results/conformal/](../results/conformal/)):

| Dataset | Model | Score | Coverage | Size | Singletons | Empty | CovGap | UCR |
|---|---|---|---|---|---|---|---|---|
| Nations | DistMult | MinMax | 0.903 | 3.48 ± 0.20 | 0.166 | 0.000 | 0.081 | 0.440 |
| Nations | DistMult | Softmax | 0.902 | 3.24 ± 0.14 | 0.140 | 0.000 | 0.099 | 0.280 |
| Nations | DistMult | APS | 0.901 | 3.57 ± 0.14 | 0.104 | 0.002 | 0.091 | 0.333 |
| Nations | ComplEx | MinMax | 0.900 | 4.02 ± 0.05 | 0.108 | 0.000 | 0.065 | 0.360 |
| Nations | ComplEx | Softmax | 0.898 | 3.77 ± 0.17 | 0.035 | 0.000 | 0.094 | 0.293 |
| Nations | ComplEx | APS | 0.899 | 4.11 ± 0.22 | 0.034 | 0.001 | 0.081 | 0.280 |
| Nations | RESCAL | MinMax | 0.901 | 5.98 ± 0.28 | 0.025 | 0.000 | 0.079 | 0.373 |
| Nations | RESCAL | Softmax | 0.899 | 5.08 ± 0.12 | 0.007 | 0.000 | 0.105 | 0.200 |
| Nations | RESCAL | APS | 0.901 | 5.34 ± 0.17 | 0.004 | 0.000 | 0.083 | 0.280 |
| UMLS | DistMult | MinMax | 0.900 | 3.22 ± 0.19 | 0.217 | 0.000 | 0.076 | 0.304 |
| UMLS | DistMult | Softmax | 0.900 | 3.12 ± 0.15 | 0.158 | 0.003 | 0.082 | 0.289 |
| UMLS | DistMult | APS | 0.900 | 6.32 ± 0.67 | 0.151 | 0.015 | 0.071 | 0.437 |
| UMLS | ComplEx | MinMax | 0.900 | 2.94 ± 0.06 | 0.233 | 0.000 | 0.078 | 0.400 |
| UMLS | ComplEx | Softmax | 0.901 | 3.10 ± 0.04 | 0.213 | 0.033 | 0.097 | 0.363 |
| UMLS | ComplEx | APS | 0.900 | 10.36 ± 0.65 | 0.004 | 0.000 | 0.085 | 0.163 |
| UMLS | RESCAL | MinMax | 0.899 | 2.89 ± 0.02 | 0.263 | 0.000 | 0.070 | 0.422 |
| UMLS | RESCAL | Softmax | 0.901 | 2.88 ± 0.04 | 0.194 | 0.020 | 0.078 | 0.296 |
| UMLS | RESCAL | APS | 0.900 | 7.20 ± 0.50 | 0.054 | 0.004 | 0.059 | 0.385 |
| Kinship | DistMult | MinMax | 0.900 | 5.43 ± 0.14 | 0.098 | 0.000 | 0.057 | 0.248 |
| Kinship | DistMult | Softmax | 0.900 | 5.34 ± 0.11 | 0.031 | 0.000 | 0.070 | 0.238 |
| Kinship | DistMult | APS | 0.900 | 6.57 ± 0.25 | 0.044 | 0.001 | 0.041 | 0.305 |
| Kinship | ComplEx | MinMax | 0.900 | 2.58 ± 0.06 | 0.266 | 0.000 | 0.035 | 0.467 |
| Kinship | ComplEx | Softmax | 0.901 | 2.64 ± 0.06 | 0.200 | 0.018 | 0.062 | 0.419 |
| Kinship | ComplEx | APS | 0.900 | 5.21 ± 0.11 | 0.018 | 0.000 | 0.043 | 0.295 |
| Kinship | RESCAL | MinMax | 0.900 | 3.12 ± 0.05 | 0.207 | 0.000 | 0.036 | 0.467 |
| Kinship | RESCAL | Softmax | 0.900 | 3.37 ± 0.07 | 0.134 | 0.016 | 0.050 | 0.467 |
| Kinship | RESCAL | APS | 0.900 | 5.71 ± 0.12 | 0.008 | 0.000 | 0.032 | 0.419 |

Singletons and Empty are the fractions of sets with one and zero entities. CovGap and
UCR measure per-relation coverage (see [metrics](#metrics)); they use the relations with
at least 10 test queries: 15 of 41 test relations on Nations, 27 of 36 on UMLS, 21 of
23 on Kinship.

### Set size and model accuracy

Set size at α = 0.1 (marginal calibration) next to the filtered test metrics of the
models (mean over the 5 seeds; see [hyperparameter_tuning.md](hyperparameter_tuning.md)):

| Dataset | Model | MRR | Hits@1 | Hits@3 | Hits@10 | Softmax | MinMax | APS | Rank |
|---|---|---|---|---|---|---|---|---|---|
| Nations | DistMult | 0.813 | 0.706 | 0.899 | 0.997 | 3.24 | 3.48 | 3.57 | 4.25 |
| Nations | ComplEx | 0.780 | 0.667 | 0.866 | 0.994 | 3.77 | 4.02 | 4.11 | 4.77 |
| Nations | RESCAL | 0.661 | 0.468 | 0.813 | 0.986 | 5.08 | 5.98 | 5.34 | 5.61 |
| UMLS | DistMult | 0.896 | 0.825 | 0.962 | 0.992 | 3.12 | 3.22 | 6.32 | 4.86 |
| UMLS | ComplEx | 0.925 | 0.875 | 0.972 | 0.992 | 3.10 | 2.94 | 10.36 | 4.58 |
| UMLS | RESCAL | 0.898 | 0.807 | 0.990 | 0.997 | 2.88 | 2.89 | 7.20 | 4.32 |
| Kinship | DistMult | 0.704 | 0.572 | 0.799 | 0.949 | 5.34 | 5.43 | 6.57 | 7.40 |
| Kinship | ComplEx | 0.882 | 0.816 | 0.941 | 0.986 | 2.64 | 2.58 | 5.21 | 3.92 |
| Kinship | RESCAL | 0.847 | 0.766 | 0.915 | 0.977 | 3.37 | 3.12 | 5.71 | 4.09 |

- **More accurate models give smaller sets.** On Kinship, ComplEx (MRR 0.882) halves
  the sets of DistMult (0.704): 2.64 vs 5.34 entities with Softmax. On Nations, sets
  grow from DistMult to ComplEx to RESCAL as MRR drops.
- **But MRR and Hits@1 do not decide set size.** On UMLS, RESCAL has the lowest Hits@1
  (0.807, ComplEx 0.875) and the smallest Softmax, MinMax and Rank sets (Softmax 2.88 vs
  3.10, Rank 4.32 vs 4.58). At 90% coverage, the sets must also contain the targets that
  the model ranks worst, so the tail of the rank distribution matters: RESCAL has the
  highest Hits@3 and Hits@10 on UMLS. Models selected by validation MRR, as here, are
  not necessarily the ones with the smallest sets.
- **Hits@k overstates how small top-k sets can be.** LibKGE's Hits@k filter all other
  test answers of a query, as is standard. Conformal sets keep them as candidates,
  since they are true answers (see [filtering](#method)). UMLS test queries have 1.9
  other test answers on average (71% have at least one; Kinship 1.0 and 57%, Nations
  0.7 and 49%). For RESCAL on UMLS (seed 1), Hits@3 drops from 0.989 to 0.840 when the
  other test answers are not filtered, and the 90% quantile of the target's rank rises
  from 2 to 4; its conformal top-k sets hold 4.3 entities on average.

### Per-relation and per-direction calibration

Marginal calibration covers 90% on average, but not for every relation: with Softmax
and APS, 16–47% of the relations are under-covered (UCR), and per-relation coverage is
off target by 0.032–0.105 on average (CovGap). Calibrating per relation fixes this where
relations have enough calibration data, at the cost of larger sets. At α = 0.1:

| Dataset | Model | Groups | Softmax: coverage | size | CovGap | UCR | APS: coverage | size | CovGap | UCR |
|---|---|---|---|---|---|---|---|---|---|---|
| Nations | DistMult | marginal | 0.902 | 3.24 | 0.099 | 0.280 | 0.901 | 3.57 | 0.091 | 0.333 |
| | | direction | 0.905 | 3.34 | 0.095 | 0.280 | 0.903 | 3.66 | 0.089 | 0.333 |
| | | relation | 0.963 | 7.14 | 0.059 | 0.000 | 0.964 | 7.23 | 0.059 | 0.000 |
| | ComplEx | marginal | 0.898 | 3.77 | 0.094 | 0.293 | 0.899 | 4.11 | 0.081 | 0.280 |
| | | direction | 0.902 | 3.86 | 0.092 | 0.280 | 0.902 | 4.18 | 0.078 | 0.307 |
| | | relation | 0.962 | 7.16 | 0.058 | 0.000 | 0.962 | 7.30 | 0.057 | 0.013 |
| | RESCAL | marginal | 0.899 | 5.08 | 0.105 | 0.200 | 0.901 | 5.34 | 0.083 | 0.280 |
| | | direction | 0.902 | 5.20 | 0.103 | 0.200 | 0.904 | 5.44 | 0.081 | 0.267 |
| | | relation | 0.963 | 7.27 | 0.059 | 0.013 | 0.962 | 7.47 | 0.057 | 0.000 |
| UMLS | DistMult | marginal | 0.900 | 3.12 | 0.082 | 0.289 | 0.900 | 6.32 | 0.071 | 0.437 |
| | | direction | 0.901 | 3.16 | 0.083 | 0.296 | 0.901 | 6.30 | 0.070 | 0.444 |
| | | relation | 0.922 | 17.61 | 0.043 | 0.007 | 0.922 | 20.44 | 0.043 | 0.030 |
| | ComplEx | marginal | 0.901 | 3.10 | 0.097 | 0.363 | 0.900 | 10.36 | 0.085 | 0.163 |
| | | direction | 0.902 | 3.11 | 0.096 | 0.341 | 0.901 | 10.53 | 0.084 | 0.163 |
| | | relation | 0.923 | 17.50 | 0.044 | 0.037 | 0.922 | 23.56 | 0.042 | 0.037 |
| | RESCAL | marginal | 0.901 | 2.88 | 0.078 | 0.296 | 0.900 | 7.20 | 0.059 | 0.385 |
| | | direction | 0.901 | 2.91 | 0.077 | 0.319 | 0.901 | 7.28 | 0.059 | 0.378 |
| | | relation | 0.922 | 16.96 | 0.042 | 0.030 | 0.922 | 22.37 | 0.042 | 0.015 |
| Kinship | DistMult | marginal | 0.900 | 5.34 | 0.070 | 0.238 | 0.900 | 6.57 | 0.041 | 0.305 |
| | | direction | 0.900 | 5.38 | 0.070 | 0.238 | 0.901 | 6.59 | 0.041 | 0.314 |
| | | relation | 0.909 | 6.58 | 0.013 | 0.048 | 0.909 | 8.14 | 0.013 | 0.048 |
| | ComplEx | marginal | 0.901 | 2.64 | 0.062 | 0.419 | 0.900 | 5.21 | 0.043 | 0.295 |
| | | direction | 0.901 | 2.65 | 0.062 | 0.419 | 0.900 | 5.20 | 0.043 | 0.276 |
| | | relation | 0.909 | 3.63 | 0.013 | 0.076 | 0.909 | 6.15 | 0.012 | 0.095 |
| | RESCAL | marginal | 0.900 | 3.37 | 0.050 | 0.467 | 0.900 | 5.71 | 0.032 | 0.419 |
| | | direction | 0.901 | 3.40 | 0.050 | 0.467 | 0.900 | 5.69 | 0.031 | 0.419 |
| | | relation | 0.910 | 4.78 | 0.013 | 0.057 | 0.909 | 7.07 | 0.013 | 0.076 |

- **Kinship** has enough data per relation (about 47 calibration queries on average):
  per-relation calibration cuts CovGap to 0.013 for every model and both scores. With
  Softmax, sets grow by 23% (DistMult), 38% (ComplEx) and 42% (RESCAL); the more accurate
  models pay more in relative terms, as their marginal sets are smaller. The relations
  still counted as under-covered miss by at most 0.002 on average (UCG), which is within
  sampling noise.
- **Nations and UMLS** have many rare relations. A relation with fewer than
  1/α − 1 calibration queries (9 at α = 0.1) gets full sets, which is the only valid
  choice. This affects about half of the Nations test queries (33 of 41 relations) and
  10% of the UMLS test queries (17 of 36 relations), for every model. Coverage then
  overshoots (0.96 and 0.92); with Softmax, sets grow by 43–120% on Nations and
  5.6–5.9-fold on UMLS (to 17.0–17.6 entities). Handling rare relations is what CondKGCP is
  for (not implemented yet).
- **Per-direction calibration** equalizes head and tail queries, which marginal
  calibration treats unequally: with Softmax, the two directions differ by up to 0.07
  (UMLS DistMult: tail 0.936, head 0.864; Nations ComplEx 0.918 vs 0.878; UMLS RESCAL
  0.921 vs 0.881). Per direction, both reach 0.899–0.906, with sets at most 3.2% larger.

### Calibrating on the validation split

As in earlier work, calibrate on the validation queries and evaluate on all test queries
(`--calibration valid`; known answers then come from train only). Range of marginal
coverage over the six scores:

| Dataset | Model | α = 0.05 (target 0.95) | α = 0.1 (target 0.90) | α = 0.2 (target 0.80) |
|---|---|---|---|---|
| Nations | DistMult | 0.911–0.961 | 0.860–0.902 | 0.744–0.784 |
| Nations | ComplEx | 0.935–0.958 | 0.886–0.910 | 0.772–0.806 |
| Nations | RESCAL | 0.939–0.951 | 0.887–0.923 | 0.787–0.851 |
| UMLS | DistMult | 0.942–0.965 | 0.902–0.927 | 0.803–0.847 |
| UMLS | ComplEx | 0.940–0.956 | 0.883–0.919 | 0.782–0.828 |
| UMLS | RESCAL | 0.944–0.969 | 0.894–0.927 | 0.802–0.838 |
| Kinship | DistMult | 0.949–0.955 | 0.895–0.906 | 0.781–0.810 |
| Kinship | ComplEx | 0.948–0.954 | 0.891–0.923 | 0.780–0.803 |
| Kinship | RESCAL | 0.943–0.959 | 0.895–0.911 | 0.788–0.857 |

Every model is under-covered for some scores: on Nations by up to 0.056 (DistMult) and
0.028 (ComplEx), on UMLS and Kinship by up to 0.020. This is a single fixed split, so
part of the deviation is chance (the sampling error of one split is about ±0.02 on
Nations). But validation and test queries are not exchangeable to begin with: the
validation split was used for early stopping and model selection. Random halves of the
test split avoid both problems and allow averaging over many splits.

### Observations

- **Coverage holds** for every model, score, grouping, dataset and α when calibrating
  on test halves, as the theory guarantees.
- **Softmax or MinMax gives the smallest sets** at α = 0.1 for every model: Softmax for
  DistMult, for all models on Nations and for RESCAL on UMLS; MinMax for ComplEx on UMLS
  and Kinship and for RESCAL on Kinship. The two are within 8% of each other, except on
  Nations RESCAL (MinMax 18% larger). This matches the known result that 1 − softmax
  (LAC) minimizes the expected set size when the probabilities are accurate (Sadinle et
  al., 2019). Softmax can return empty sets, when no candidate is probable enough
  (1.6–3.3% of the sets for ComplEx and RESCAL on UMLS and Kinship); MinMax never does,
  as the top candidate always scores 0.
- **APS sets depend on how confident the model is.** They are larger than Softmax's
  for every model: by 5–10% on Nations, 1.2–2.0× on Kinship, 2.0–3.3× on UMLS. On UMLS
  and Kinship, the factor is largest for the three models trained with label smoothing
  (KvsAll, smoothing 0.28–0.30: ComplEx on both, RESCAL on Kinship), whose probabilities
  are flat: their mean top-1 probability over the filtered candidates is 0.12, against
  0.26–0.59 for the other models (seed 1; 0.42 for DistMult on UMLS, KvsAll without
  smoothing). APS adds up probability mass until the target
  is included, so flat distributions need many candidates, and these models' APS sets
  are rarely singletons (0.4–1.8%). Softmax and MinMax threshold each candidate on its
  own and are hardly affected. In return, APS covers relations more evenly than Softmax
  for all nine models (CovGap 0.032–0.091 vs 0.050–0.105). RAPS sits between the two.
- **Raw logits (Neg) need logits that are comparable across queries.** For the four
  models without reciprocal relations (ComplEx and RESCAL on UMLS and Kinship), a
  triple's head and tail query share the target's logit f(s, p, o). Neg then covers
  both directions equally (0.900–0.901), and its sets are only 6–21% larger than
  Softmax's. With reciprocal relations (all DistMult models, ComplEx and RESCAL on
  Nations), head queries are scored with a separate inverse relation, and the two
  directions' logits are on different scales: Neg covers them unequally (e.g. tail
  0.826, head 0.973 for ComplEx on Nations), and DistMult's Neg sets are 1.8–5.5×
  larger than Softmax's, with 3–7% empty sets. RESCAL on Nations, trained with BCE,
  is the exception (Neg 3% larger than Softmax).
- **Rank (conformal top-k)** gives every query the same set size k (fewer only if
  fewer candidates remain after filtering). That is easy to interpret, but it wastes
  size on easy queries, never yields singletons, and over-covers.
- **Per-relation calibration** needs data: it works well on Kinship for every model,
  but on Nations and UMLS rare relations fall back to full sets.

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
the candidates scoring at most the threshold (`SplitConformal`). With a grouping, each
group (relation, or direction) gets its own threshold from its own calibration queries
(Mondrian CP, `MondrianConformal`), so coverage holds within every group.

**CondKGCP** (`CondKGCP`; Zhu et al., 2025) calibrates per group of relations, in two
steps:

1. *Merging.* Relations with at least `min_size` calibration queries form a group each;
   every other relation joins the most similar of them, by the L1 distance of the
   relation embeddings (`MergedRelations`). With reciprocal relations, a relation's
   embeddings for tail and head queries are concatenated. Relations without
   calibration queries are merged too, so they get a threshold. If no relation has
   `min_size` queries, all form one group.
2. *Rank calibration* (as RC3P; Shi et al., 2024). In each group g, k(g) is the smallest
   k such that fewer than a fraction α of the calibration targets rank beyond k; they
   are a fraction ε(g). A set holds the candidates that rank within k(g) and score at
   most the conformal quantile at level α − γ·ε(g). A candidate's rank is the number of
   candidates scoring at most it (ties count against it), so the set is the
   candidates scoring below a per-query threshold, and set sizes need no extra pass.

Coverage within a group is at least 1 − α − (1 − γ)·ε(g) (Proposition 1 of the paper),
for ε(g) the group's true top-k(g) miscoverage: at least 1 − α for γ = 1, and a smaller γ
trades coverage for smaller sets. Here ε(g) is estimated on the group's calibration
queries, which is optimistic for small groups: with one group per relation, Nations
(about 3.6 calibration queries per relation) covers 0.87–0.88 at α = 0.1 even with
γ = 1, while merging (there into a single group) covers 0.91. The two
ablations of the paper are `CondKGCP(alpha, by_relation, gamma)` (without merging) and
`MondrianConformal(alpha, MergedRelations(...))` (without rank calibration). The paper
defines ranks by the logits, without filtering; here they are ranks by the
nonconformity score among the filtered candidates, the same order for every score up
to ties.

**Choosing CondKGCP's settings.** `scripts/conformal.py --select` picks `min_size` and
γ per score function and α on 10 random splits of the validation queries, each
calibrating on as many queries as a test half. The candidates are 8 values of
`min_size`, evenly spaced on a log scale from ⌊1/α⌋ + 1 (the smallest group that can
estimate a top-k miscoverage below α) to the expected calibration queries of the most
frequent relation, times γ ∈ {0, 0.25, 0.5, 0.75, 1}. Of the settings that cover at least
min(1 − α, marginal CP's coverage on the same splits), it keeps those within δ = 0.005 of
the lowest CovGap, and of these picks the one with the smallest mean set size per
relation. On the Pareto front of CovGap and set size, that is the front's smallest sets
within δ of its lowest CovGap. `scripts/condkgcp_search.py` runs every setting on
validation and on test and writes a page to review the search
([results/conformal/libkge/condkgcp-search.html](../results/conformal/libkge/condkgcp-search.html),
to open in a browser): per dataset, model, score function and α, the steps of the rule,
the Pareto front, the settings by `min_size` and γ, and the selected settings on test
against Mondrian and marginal CP. It holds LibKGE's six models (RESCAL, TransE, DistMult,
ComplEx, ConvE, RotatE) on WN18 and FB15k. Over the six scores and three α, the selected
settings' sets per relation are 0.18× (WN18) and 0.05× (FB15k) as large as Mondrian CP's
(geometric mean on test), for a mean CovGap of 0.023 and 0.038, against Mondrian's 0.016
and 0.030 and marginal CP's 0.076 and 0.101. Of the 216 cases, CondKGCP's CovGap is
lower than Mondrian's in 37 and at most 0.01 higher in 148: CondKGCP trades a little of
Mondrian's conditional coverage for much smaller sets. The gain is smallest for ComplEx
on WN18 (0.55× Mondrian's sets) and largest on FB15k, where Mondrian gives the full
entity set to the many relations with too few calibration queries for a finite
threshold (fewer than 1/α − 1).

**Calibration splits.** Test queries, not triples, are split at random. A triple's head
and tail queries are correlated (correlation 0.59 for Softmax with DistMult on Nations;
without reciprocal relations, their targets have the same Neg score). If both always fell
on the same side, the guarantee would weaken to ⌈(n+1)(1−α)⌉/(n+2): 0.896 instead of
0.901 on Nations at α = 0.1. Over 2,000 Nations splits, coverage was about 0.001 lower
with triple splits than with query splits (Softmax and APS).

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
| Models | DistMult, ComplEx and RESCAL: the selected configuration of each dataset, 5 seeds (`models/<dataset>-<model>/seed{1..5}.pt`, from [Hugging Face](https://huggingface.co/Yuqicheng/KGCP-KGE-Checkpoints), see [trained models](../README.md#trained-models)) |
| Calibration | random half of the test queries; evaluation on the other half |
| Splits | 100 per seed, different for each seed and the same for the three models (500 per dataset and model) |
| Filtering | train + valid |
| α | 0.05, 0.1, 0.2 |
| Scores | Neg, MinMax, Softmax, APS, RAPS, Rank |
| Groupings | marginal, per relation, per direction |
| Relation metrics | relations with at least 10 test queries |
| Reported | metrics pooled over the splits; mean (± std) over the 5 seeds |

The full results, one row per (seed, score, grouping, α), are in
[results/conformal/](../results/conformal/): `<dataset>-<model>-test.csv`, and
`<dataset>-<model>-valid.csv` for validation calibration. The whole grid (three
datasets and three models) runs in about 2 minutes on a laptop CPU.

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
- **CondKGCP**: the earlier code's sets used only the score threshold, without the
  rank cap k(g) of the paper's Eq. 7, and its calibration ranks compared the raw logits
  with the target's softmax probability, which is not the target's rank for Softmax or
  APS. Its k(g) also allowed a top-k miscoverage equal to α (the paper: below α).

## Reproduce

```bash
uv run scripts/conformal.py  # the tuned models in models/
uv run scripts/conformal.py --calibration valid
uv run scripts/condkgcp_search.py models/libkge/wn18-*.pt models/libkge/fb15k-*.pt  # or on Modal: cloud/modal/README.md
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
- Vovk, Gammerman, Shafer. Algorithmic learning in a random world. Springer, 2005.
- Shi, Ghosh, Belkhouja, Doppa, Yan. Conformal prediction for class-wise coverage via
  augmented label rank calibration. NeurIPS, 2024.
- Zhu, Hernández, He, Ding, Xiong, Kharlamov, Staab. Predicate-conditional conformalized
  answer sets for knowledge graph embeddings. Findings of ACL, 2025.
