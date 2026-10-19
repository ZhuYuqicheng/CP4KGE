# UnKGE: the models of UnKGCP in LibKGE

UnKGCP (Zhu, Wu et al., "Certainty in Uncertainty: Reasoning over Uncertain Knowledge
Graphs with Statistical Guarantees", EMNLP 2025, [arXiv:2510.24754](https://arxiv.org/abs/2510.24754))
predicts intervals for the confidence of triples of uncertain knowledge graphs, with
models of three uncertain KGE methods. This folder makes the authors' trained models
usable by `kgcp`: it converts them into LibKGE models, which `kgcp.LibKGE` loads like any
other.

| model | method | checkpoints |
|---|---|---|
| `ukge_logi` | UKGE (Chen et al., 2019), logistic output | TensorFlow 1 |
| `ukge_rect` | UKGE, bounded rectifier output (not in the paper's tables) | TensorFlow 1 |
| `passleaf` | PASSLEAF (Chen et al., 2021): UKGE trained with a pool of pseudo-labeled samples (`UKGE_logi_m2`) | TensorFlow 1 |
| `beurre` | BEUrRE (Chen et al., 2021): probabilistic box embeddings | pickled PyTorch modules |

Datasets: CN15k (ConceptNet), NL27k (NELL) and PPI5k (STRING), as released with UKGE.
The authors trained 10 models (seeds) per model and dataset, 120 in all. Only the first
five (seeds 0–4) are kept, as kgcp evaluates 5 models per model and dataset elsewhere.

**Status:** the 60 models of seeds 0–4 converted and verified (see
[Verification](#verification)), in `models/unkgcp/<dataset>-<model>/seed<N>.pt` (843 MB,
git-ignored).

## Layout

```
unkge/                   LibKGE models, found through the config option `modules`
  ukge.py, ukge.yaml     ukge: DistMult score w <s, p, o> + b; output logi or rect
  beurre.py, beurre.yaml beurre: Gumbel box scorer
download.py              datasets (UKGE's GitHub), the authors' models (Models.zip)
prepare_data.py          LibKGE datasets in third_party/libkge/data/<dataset>/
convert.py               the authors' models -> models/unkgcp/<dataset>-<model>/seed<N>.pt
verify.py                errors against the paper; BEUrRE against BEUrRE's code
convert_dataset.sh       download, convert and verify one dataset, then delete the downloads
tests/                   the models on LibKGE's toy dataset
```

The package is installed with the project (`uv sync`; it is a path dependency in the
top-level `pyproject.toml`), so the converted models load wherever `kgcp` is installed.

## The models

Both models score triples so that higher is more plausible (`score_spo`, and `score_sp`
and `score_po` for all candidate answers, as `kgcp.LibKGE.score` needs), and add
`confidence(s, p, o)`, the predicted confidence in [0, 1]:

- `ukge`: the score is `w <s, p, o> + b` (DistMult with an affine map); the confidence is
  its sigmoid (`ukge.output: logi`) or the score clamped to [0, 1] (`rect`, UKGE's bounded
  score; UnKGCP's code for rect models uses the unclamped score).
- `beurre`: the score is log P(head box | tail box); the confidence is its exponential.
  Entity embeddings hold [lower corner, log width] (size 2d), relation embeddings [head
  translation, head scale, tail translation, tail scale] (size 4d). The operations follow
  BEUrRE's code (`BiGumbelBox`) in the same order, so converted models give the same
  floats. Scoring all candidates of a query takes n × E × d floats (and twice that
  briefly): use small batches. As in BEUrRE's code, a confidence can exceed 1 by float32
  rounding: 1 + 9.5e-7 for one CN15k triple, the self-loop (11315, relatedto, 11315).

The models only score; there is no training code for them.

## Steps

```bash
uv run python third_party/unkge/download.py data      # ~40 MB, checked against git blob hashes
uv run python third_party/unkge/prepare_data.py
for d in ppi5k cn15k nl27k; do
  bash third_party/unkge/convert_dataset.sh $d Models.zip <UnKGCP checkout>
done
```

For one dataset, `convert_dataset.sh` runs these steps, and deletes the downloaded files
(models/unkgcp/raw/<dataset>-*) only if `verify.py` passes and every run was converted:

```bash
uv run python third_party/unkge/download.py models --zip Models.zip --datasets ppi5k
uv run --with tensorflow python third_party/unkge/convert.py --datasets ppi5k
uv run python third_party/unkge/verify.py --datasets ppi5k --beurre-code <UnKGCP>/beurre-main
```

`download.py` takes seeds 0–4 unless given `--seeds`. All three steps skip runs that are
done, so an interrupted run can be restarted. With the authors' OneDrive URL, all 120
runs (seeds 0–9) took about 25 minutes (download ~5 MB/s, conversion and verification
~1 minute per dataset) and at most 2.9 GB of downloads at a time (NL27k); seeds 0–4 are
about half of that.

`--zip` is the authors' `Models.zip` (4.47 GB), downloaded from their OneDrive folder
(link in the [UnKGCP README](https://github.com/0sidewalkenforcer0/UnKGCP)), or a direct
download URL of it: then only the selected runs are read, with HTTP range requests.
OneDrive's direct URLs come from its web page (the item's `@content.downloadUrl`) and
expire after about an hour; the share link itself does not serve the file (403).

`convert.py` needs TensorFlow 2 to read the TensorFlow 1 checkpoints, only for this step
(`uv run --with tensorflow`; it is not a project dependency). `verify.py --beurre-code`
runs BEUrRE's code from a checkout of the UnKGCP repository.

## Sources

- **Models:** `Models.zip` of the UnKGCP authors' OneDrive folder (4,466,147,436 bytes,
  uploaded 2025-08-25), 120 runs: `Models/UKGE/<dataset>/{logi,rect}_<date>_<seed>/`,
  `Models/PASSLEAF/<dataset>/UKGE_logi_m2_<date>_<seed>/` (TensorFlow checkpoints of the
  best epoch, with Adam's slots, which are dropped) and `Models/BeurRE/<dataset>/bigumbelbox-<id>.pt`.
  BEUrRE's CN15k and NL27k runs are named by wandb ids; they get seeds 0–9 in the order of
  their names (PPI5k's are named by seed). Each converted model records the zip members it
  came from, with their sizes and SHA-256 (`checkpoint["source"]`).
- **Data:** [UKGE](https://github.com/stasl0217/UKGE) at `5049b84`
  (`data/<dataset>/{train,val,test,test_with_neg,softlogic}.tsv`), and the names of CN15k's
  and NL27k's entities and relations from [BEUrRE](https://github.com/stasl0217/beurre)
  at `374b724`. The authors' `UnKGCP_Data.zip` cannot be used: each of its files stops at
  262,144 bytes (except CN15k's softlogic.tsv). Its partial files are byte-for-byte
  prefixes of UKGE's, except NL27k's test_with_neg.tsv, which names the entities and
  relations where UKGE's has BEUrRE's ids: mapped with BEUrRE's names, its 2,904 complete
  rows are identical.
- **Code:** [UnKGCP](https://github.com/0sidewalkenforcer0/UnKGCP) at `6cdf658`, read to
  write the models and the conversion. None of UnKGCP, UKGE, PASSLEAF and BEUrRE has a
  license, so no code of theirs is copied here: the models are written from the papers'
  formulas and the order of operations of BEUrRE's code.

## Ids

Entity and relation ids are those of the files: CN15k and PPI5k number them, NL27k's
files (except test_with_neg.tsv) name them, and their ids are BEUrRE's (`entity_id.csv`,
`relation_id.csv`). BEUrRE models use these ids. UKGE and PASSLEAF number entities and
relations in the order they first occur in train, val, test, test_with_neg and softlogic
(head before tail); `convert.py` reorders the rows of their embeddings. Entities or
relations that UKGE never saw would keep zero embeddings; there are none.

The LibKGE datasets have the splits train, valid, test (the 4th column, which LibKGE does
not read, is the confidence) and test_with_neg: the test triples, in the same order with
confidences rounded to 3 digits, followed by as many negatives (confidence 0) from UKGE.
softlogic.del (UKGE's soft-logic triples) is not a split; the conversion needs it for
UKGE's order.

| dataset | entities | relations | train | valid | test |
|---|---|---|---|---|---|
| CN15k | 15,000 | 36 | 204,984 | 16,881 | 19,293 |
| NL27k | 27,221 | 417 (405 in train) | 149,100 | 12,278 | 14,034 |
| PPI5k | 4,999 | 7 | 230,929 | 19,017 | 21,720 |

The split sizes are those of the paper; NL27k has 417 relation ids, as BEUrRE's
models, of which the paper counts 404.

Properties of the data to keep in mind for conformal prediction:

| dataset | repeated (h, r, t) in train / valid / test | valid ∩ train | test ∩ train | test ∩ valid | test negatives that are known positives |
|---|---|---|---|---|---|
| CN15k | 11,710 / 86 / 127 | 1,770 | 1,966 | 178 | 150 of 19,293 |
| NL27k | 0 / 0 / 0 | 0 | 0 | 0 | 201 of 14,034 |
| PPI5k | 16,268 / 130 / 154 | 2,564 | 2,940 | 250 | 1,772 of 21,720 |

## Verification

`convert.py` writes a UKGE or PASSLEAF model only if its confidences equal those of UKGE's
formula on the original arrays with UKGE's ids, for all test_with_neg triples (up to 1e-5;
float32). `verify.py` computes the errors of the converted models and, with
`--beurre-code`, compares converted BEUrRE models with the originals, scored by BEUrRE's
code.

All 120 models (seeds 0–9) passed when converted: every UKGE and PASSLEAF model
reproduces UKGE's formula (the differences printed for PPI5k and CN15k were 1.3e-7 to
2.4e-7), and every BEUrRE model gives exactly the confidences of BEUrRE's code. Seeds 5–9
were then deleted. Errors on test of the kept models, mean ± std over seeds 0–4
(`models/unkgcp/verify-<dataset>.md`; the paper's Table 2 reports means over 10 seeds,
rounded to 2 digits):

| dataset | model | test MSE | test MAE | paper MSE / MAE | test_with_neg MSE | test_with_neg MAE |
|---|---|---|---|---|---|---|
| CN15k | ukge_logi | 0.2433 ± 0.0021 | 0.4114 ± 0.0025 | 0.24 / 0.41 | 0.1273 ± 0.0011 | 0.2270 ± 0.0017 |
| CN15k | ukge_rect | 0.1381 ± 0.0013 | 0.2936 ± 0.0018 | – | 0.0823 ± 0.0006 | 0.1945 ± 0.0007 |
| CN15k | passleaf | 0.2386 ± 0.0009 | 0.4064 ± 0.0009 | 0.24 / 0.41 | 0.1242 ± 0.0003 | 0.2322 ± 0.0004 |
| CN15k | beurre | 0.1174 ± 0.0021 | 0.2835 ± 0.0018 | 0.12 / 0.28 | 0.0872 ± 0.0009 | 0.2363 ± 0.0020 |
| NL27k | ukge_logi | 0.0500 ± 0.0009 | 0.1111 ± 0.0019 | 0.05 / 0.11 | 0.0335 ± 0.0005 | 0.0765 ± 0.0013 |
| NL27k | ukge_rect | 0.0259 ± 0.0002 | 0.0775 ± 0.0011 | – | 0.0243 ± 0.0002 | 0.0710 ± 0.0012 |
| NL27k | passleaf | 0.0552 ± 0.0008 | 0.1113 ± 0.0010 | 0.06 / 0.11 | 0.0358 ± 0.0004 | 0.0815 ± 0.0008 |
| NL27k | beurre | 0.0338 ± 0.0004 | 0.1184 ± 0.0004 | 0.03 / 0.12 | 0.0315 ± 0.0002 | 0.0910 ± 0.0006 |
| PPI5k | ukge_logi | 0.0050 ± 0.0001 | 0.0360 ± 0.0003 | 0.01 / 0.04 | 0.0097 ± 0.0001 | 0.0394 ± 0.0005 |
| PPI5k | ukge_rect | 0.0031 ± 0.0000 | 0.0318 ± 0.0004 | – | 0.0094 ± 0.0000 | 0.0400 ± 0.0002 |
| PPI5k | passleaf | 0.0051 ± 0.0001 | 0.0323 ± 0.0001 | 0.01 / 0.03 | 0.0100 ± 0.0000 | 0.0409 ± 0.0003 |
| PPI5k | beurre | 0.0072 ± 0.0004 | 0.0569 ± 0.0015 | 0.01 / 0.06 | 0.0150 ± 0.0004 | 0.0572 ± 0.0008 |

17 of the paper's 18 values agree to the rounding. The other is just below a rounding
boundary: PPI5k UKGE's MSE (0.004999, the paper 0.01). The models load with
`kgcp.LibKGE`, and their scores of all candidates agree with `score_spo`.
