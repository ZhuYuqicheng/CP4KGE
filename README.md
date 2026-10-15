# KGCP: conformal prediction for knowledge graph embeddings

`kgcp` turns the scores of a trained knowledge graph embedding (KGE) model into answer sets that contain the true answer with a chosen
probability (e.g. 90%). Models are trained with [LibKGE](https://github.com/uma-pi1/kge),
patched for recent PyTorch and Apple Silicon; other pipelines can plug in their scores.

## Layout

```
src/kgcp/             conformal prediction library
cloud/                hyperparameter tuning on cloud GPUs (Modal)
configs/              training configurations (tuned ones: see docs/)
models/               trained models of the tuned configurations (downloaded, git-ignored)
docs/                 conformal prediction and hyperparameter tuning: protocols, results
notebooks/            Jupyter notebooks: a guide to kgcp and investigations
results/              conformal prediction and tuning results
scripts/              dataset download, hyperparameter tuning, conformal experiments
tests/                tests of kgcp
third_party/libkge/   LibKGE (upstream 9bca31f, with patches below)
  data/<dataset>/     preprocessed datasets (downloaded, git-ignored)
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

## Trained models

The 5 models (seeds 1–5) of each tuned configuration in [KGE results](#kge-results),
for DistMult, ComplEx and RESCAL on Nations, UMLS and Kinship, are on Hugging Face:
[Yuqicheng/KGCP-KGE-Checkpoints](https://huggingface.co/Yuqicheng/KGCP-KGE-Checkpoints)
(31 MB). Download them into `models/`, which is git-ignored:

```bash
uvx --from huggingface_hub hf download Yuqicheng/KGCP-KGE-Checkpoints \
    --include "models/*" --local-dir .
```

Keep `--include`: without it, the model card of the Hugging Face repository overwrites
this README. `--include "models/umls-*"` downloads the models of one dataset. One
folder per pair:

```
models/<dataset>-<model>/
  config.yaml           LibKGE configuration; seed<N>.pt was trained with
                        kge start config.yaml --random_seed.default <N>
  seed1.pt … seed5.pt   LibKGE packaged models: weights, configuration, entity/relation ids
```

The models need their dataset, since they hold the entity and relation ids but not the
triples (`uv run scripts/download_data.py nations umls kinship`). Load one with
`kgcp.LibKGE("models/umls-distmult/seed1.pt")`, or with LibKGE alone:
`KgeModel.create_from(load_checkpoint("models/umls-distmult/seed1.pt"))`.
`uv run pytest` checks that every model reproduces its recorded test metrics.

DistMult and ComplEx were trained on Apple MPS, RESCAL on an NVIDIA T4; retraining
from `config.yaml` on other hardware gives similar, not identical models. To rebuild
`models/` from the tuning runs in `runs/hpo/final/`, using the selection recorded in
`results/hpo/`:

```bash
uv run scripts/tune.py export --models distmult complex rescal
```

## Conformal prediction

```python
import kgcp

model = kgcp.LibKGE("models/umls-distmult/seed1.pt")  # device: cuda > mps > cpu
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

At α = 0.1, the tuned models give the following mean set sizes with the Softmax score,
calibrated on random halves of the test queries (5 seeds × 100 splits; marginal
calibration). Coverage is 0.90 for every model:

| Dataset | Entities | DistMult | ComplEx | RESCAL |
|---|---|---|---|---|
| Nations | 14 | 3.24 | 3.77 | 5.08 |
| UMLS | 135 | 3.12 | 3.10 | 2.88 |
| Kinship | 104 | 5.34 | 2.64 | 3.37 |

See [docs/conformal_prediction.md](docs/conformal_prediction.md) for the method, the
protocol and all results (per-relation coverage, other scores and α, set size against
model accuracy), reproduced with

```bash
uv run scripts/conformal.py  # the tuned models in models/
```

Run the tests with `uv run pytest` (the LibKGE tests need the [trained models](#trained-models)).

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

`--device` (default `cpu`) runs the trials on `cuda` or `mps` instead. To tune on
cloud GPUs, see [cloud/](cloud/).

### Devices

`job.device` defaults to `auto`, which picks `cuda`, then `mps` (Apple Silicon), then
`cpu`. Set it explicitly with `--job.device cpu|cuda|cuda:1|mps`. Checkpoints can be
loaded on any device, e.g. `kge test runs/nations-transe --job.device cpu`.

For tiny datasets such as Nations, the CPU is about as fast as the GPU.

## KGE results

Filtered test metrics of the tuned configurations, mean ± std over 5 seeds (see
[docs/hyperparameter_tuning.md](docs/hyperparameter_tuning.md)). DistMult and ComplEx
were tuned on the Mac GPU (`--device mps`), RESCAL on a Modal T4 GPU:

| Dataset | Model | Config | MRR | Hits@1 | Hits@10 |
|---|---|---|---|---|---|
| Nations | DistMult | `configs/nations-distmult.yaml` | 0.813 ± 0.013 | 0.706 ± 0.018 | 0.997 ± 0.003 |
| UMLS | DistMult | `configs/umls-distmult.yaml` | 0.896 ± 0.016 | 0.825 ± 0.024 | 0.992 ± 0.004 |
| Kinship | DistMult | `configs/kinship-distmult.yaml` | 0.704 ± 0.007 | 0.572 ± 0.011 | 0.949 ± 0.002 |
| Nations | ComplEx | `configs/nations-complex.yaml` | 0.780 ± 0.006 | 0.667 ± 0.013 | 0.994 ± 0.002 |
| UMLS | ComplEx | `configs/umls-complex.yaml` | 0.925 ± 0.014 | 0.875 ± 0.027 | 0.992 ± 0.001 |
| Kinship | ComplEx | `configs/kinship-complex.yaml` | 0.882 ± 0.006 | 0.816 ± 0.010 | 0.986 ± 0.001 |
| Nations | RESCAL | `configs/nations-rescal.yaml` | 0.661 ± 0.007 | 0.468 ± 0.015 | 0.986 ± 0.004 |
| UMLS | RESCAL | `configs/umls-rescal.yaml` | 0.898 ± 0.006 | 0.807 ± 0.012 | 0.997 ± 0.001 |
| Kinship | RESCAL | `configs/kinship-rescal.yaml` | 0.847 ± 0.002 | 0.766 ± 0.003 | 0.977 ± 0.003 |

`configs/nations-transe.yaml` comes from an earlier, small manual sweep, not from this
protocol. It reaches test MRR 0.49 (MPS) and 0.50 (CPU) on Nations, in a single run.

## Changes to LibKGE

LibKGE is vendored without its git history. All changes against upstream 9bca31f are in
[third_party/libkge.patch](third_party/libkge.patch).

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

Training was tested on CPU, Apple MPS and CUDA (an NVIDIA T4 on Modal, see
[cloud/](cloud/)). Not yet verified: the hyperparameter search extras with current Ax
releases.
