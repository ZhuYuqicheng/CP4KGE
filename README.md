# KGE Conformal

Conformal prediction for knowledge graph embeddings. `kgcp` turns the scores of a
trained KGE model into answer sets that contain the true answer with a chosen
probability (e.g. 90%). Models are trained with [LibKGE](https://github.com/uma-pi1/kge),
patched for recent PyTorch and Apple Silicon; other pipelines can plug in their scores.

## Layout

```
src/kgcp/             conformal prediction library
configs/              training configurations (tuned ones: see docs/)
docs/                 conformal prediction and hyperparameter tuning: protocols, results
results/              conformal prediction and tuning results
scripts/              dataset download, hyperparameter tuning, conformal experiments
tests/                tests of kgcp
third_party/libkge/   LibKGE (upstream 9bca31f, with patches below)
  data/<dataset>/     preprocessed datasets
runs/                 training outputs (git-ignored)
```

## Setup

Requires [uv](https://docs.astral.sh/uv/). This creates `.venv` with the locked
dependencies, and `kgcp` and LibKGE installed in editable mode:

```bash
uv sync
```

On Linux x86_64, the locked PyTorch wheel targets CUDA 13. If your GPU driver is
older, point uv at a matching [PyTorch index](https://docs.astral.sh/uv/guides/integration/pytorch/).

## Conformal prediction

```python
import kgcp

model = kgcp.LibKGE("runs/hpo/final/umls-distmult/top3/seed1")  # device: cuda > mps > cpu
kg = model.kg  # splits and ids of the model's dataset

# nonconformity of every candidate answer of the test queries (s, p, ?) and (?, p, o);
# answers known from train and valid are filtered out
queries = kgcp.Queries.from_triples(kg.split("test"))
scores = kgcp.nonconformity(model, queries, kgcp.APS(), known=kg.known("train", "valid"))

# calibrate on half of the test queries, predict answer sets for the other half
cal, test = scores.split(0.5)
cp = kgcp.SplitConformal(alpha=0.1, groups=kgcp.by_relation).calibrate(cal)
sets = cp.predict(test)  # sets.mask: [queries, entities] bool
kgcp.evaluate(sets, alpha=0.1)  # coverage, set size, per-relation coverage gaps, ...
```

Scores: `Neg`, `MinMax`, `Softmax`, `APS`, `RAPS`, `Rank` (top-k). Groupings:
`marginal` (default), `by_relation`, `by_direction`. Another KGE pipeline only needs to
provide `[queries, entities]` logits: pass them to `nonconformity` instead of a model.

At α = 0.1, the tuned DistMult models give the following answer sets, calibrated on
random halves of the test queries (5 models × 100 splits; marginal calibration):

| Dataset | Entities | Coverage | Set size (Softmax) | Set size (APS) |
|---|---|---|---|---|
| Nations | 14 | 0.90 | 3.33 | 3.74 |
| UMLS | 135 | 0.90 | 2.95 | 5.81 |
| Kinship | 104 | 0.90 | 5.17 | 6.36 |

See [docs/conformal_prediction.md](docs/conformal_prediction.md) for the method, the
protocol and all results (per-relation coverage, other scores and α), reproduced with

```bash
uv run scripts/conformal.py --datasets nations umls kinship
```

Run the tests with `uv run pytest` (the LibKGE test needs the tuned Nations models).

## Training KGE models

Download and preprocess datasets (`nations`, `umls`, `kinship`, `fb15k-237`, `wnrr`,
`yago3-10`, `codex-{s,m,l}`, ...; see `scripts/download_data.py`):

```bash
uv run scripts/download_data.py nations
```

Train, then evaluate the best checkpoint on the test split:

```bash
uv run kge start configs/nations-transe.yaml --folder runs/nations-transe
uv run kge test runs/nations-transe
```

Any config key can be overridden on the command line, e.g. a quick debug run on a
small training subset:

```bash
uv run kge start configs/nations-transe.yaml --folder runs/debug \
    --train.split train_sample --train.max_epochs 2 --valid.every 1
```

See the [LibKGE README](third_party/libkge/README.md) for all commands and options.

### Hyperparameter tuning

`scripts/tune.py` tunes a model on a dataset with the protocol of LibKGE's authors
(Ruffinelli et al., ICLR 2020), adapted to small graphs. It runs a Sobol search, then
Bayesian optimization, then reruns the top 3 configurations with 5 seeds. See
[docs/hyperparameter_tuning.md](docs/hyperparameter_tuning.md) for the protocol and
results.

```bash
uv run scripts/tune.py sobol --datasets nations umls kinship --models distmult
```

### Devices

`job.device` defaults to `auto`, which picks `cuda`, then `mps` (Apple Silicon), then
`cpu`. Set it explicitly with `--job.device cpu|cuda|cuda:1|mps`. Checkpoints can be
loaded on any device, e.g. `kge test runs/nations-transe --job.device cpu`.

For tiny datasets such as Nations, the CPU is about as fast as the GPU.

## KGE results

Filtered test metrics of the tuned configurations, mean ± std over 5 seeds (see
[docs/hyperparameter_tuning.md](docs/hyperparameter_tuning.md)):

| Dataset | Model | Config | MRR | Hits@1 | Hits@10 |
|---|---|---|---|---|---|
| Nations | DistMult | `configs/nations-distmult.yaml` | 0.806 ± 0.009 | 0.694 ± 0.016 | 0.998 ± 0.002 |
| UMLS | DistMult | `configs/umls-distmult.yaml` | 0.918 ± 0.010 | 0.862 ± 0.018 | 0.997 ± 0.001 |
| Kinship | DistMult | `configs/kinship-distmult.yaml` | 0.725 ± 0.007 | 0.603 ± 0.011 | 0.955 ± 0.002 |

`configs/nations-transe.yaml` comes from an earlier, small manual sweep, not from this
protocol. It reaches test MRR 0.49 (MPS) and 0.50 (CPU) on Nations, in a single run.

## Changes to LibKGE

All changes are visible with `git -C third_party/libkge diff`.

- **Devices**: new default `job.device: auto` (cuda > mps > cpu). Hits@k is accumulated
  on the CPU because MPS has no float64.
- **Optional dependencies**: Ax/GraSH hyperparameter search dependencies moved to the
  `libkge[search]` extra and are imported lazily, so training needs only
  torch, numpy, pandas, numba, pyyaml and path. Unused `torchviz`, `pytest-shutil` and
  `argparse` dependencies were removed.
- **Recent PyTorch (2.6+)**: checkpoints are loaded with `weights_only=False`, legacy
  sparse tensor constructors replaced by `torch.sparse_coo_tensor`, deprecated
  `torch.jit.script` dropped from RotatE helpers, and the torch version is stored in
  traces as a string so `trace.yaml` stays readable with `yaml.safe_load`.
- **Python 3.12+**: regex literals are raw strings (no invalid escape warnings).

Not yet verified: training on CUDA (no NVIDIA GPU was available) and the
hyperparameter search extras with current Ax releases.
