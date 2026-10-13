# Tuning on Modal

[app.py](app.py) runs the stages of `scripts/tune.py` (`sobol`, `refine`, `final`,
`report`) on [Modal](https://modal.com) GPUs. Every (dataset, model) pair gets its own
GPU container, and all pairs run in parallel. Within a container, `--workers` trials
share the GPU, each with an equal part of its memory; a training that needs more splits
its batches into sub-batches, which gives the same results more slowly.

- **Environment**: built from `uv.lock`, so the container has the same package versions
  as `uv sync` (PyTorch with CUDA 13; Modal's driver supports it). The image is built
  once and cached; it is rebuilt when `uv.lock` changes.
- **Code and data**: the working tree is uploaded on every run, including uncommitted
  changes and the datasets prepared locally in `third_party/libkge/data/`. Each run
  folder's `history.txt` records the commit, the GPU and the number of workers.
- **Results**: a pair's run folder is kept in the container and copied to the Modal
  volume `kge-runs` every 5 minutes and when the stages end. It has the same layout as
  `runs/hpo/` locally (journal, final runs, results, configs), plus `log.txt` with the
  output of `tune.py`.

## Setup (once)

1. Log in; this opens the browser and stores a token in `~/.modal.toml`:

   ```bash
   uv run --group modal modal token new
   ```

2. Set a spending limit in the Modal dashboard (workspace settings, usage and billing).
   The free Starter plan includes $30 of compute per month; usage beyond that is billed
   if a payment method is added.
3. Prepare the datasets locally, as for local training:

   ```bash
   uv run scripts/download_data.py nations umls kinship
   ```

## Running

First measure the speed of one training on the GPU (one model per embedding size,
10 epochs each; a few cents), to estimate the cost of a search before running it:

```bash
uv run --group modal modal run cloud/modal/app.py --datasets nations --models rescal --stages time --workers 1 --run timing
```

A quick test of all stages on Nations with 1 trial per search stage (7 Sobol trials,
1 Bayesian trial, then the 15 final runs), with the trainings capped at a few epochs:

```bash
uv run --group modal modal run cloud/modal/app.py --datasets nations --models rescal --trials 1 --max-epochs 5 --run smoke
```

The full protocol for a pair. `--detach` keeps it running when the terminal is closed;
start one command per pair, since Modal keeps only the last started call of a detached
command alive once the terminal disconnects:

```bash
uv run --group modal modal run --detach cloud/modal/app.py --datasets umls --models distmult
```

RESCAL uses its own script, [tune_rescal.py](tune_rescal.py): the protocol of
`scripts/tune.py` with embedding sizes 32, 64 and 128. RESCAL learns a d × d matrix per
relation; on Nations, sizes 256 and 512 took 70% of the compute without reaching the best
validation MRR.

```bash
uv run --group modal modal run --detach cloud/modal/app.py --datasets umls --models rescal --script cloud/modal/tune_rescal.py --run rescal
```

| Option | Default | |
|---|---|---|
| `--datasets`, `--models` | required | comma-separated; every combination is one pair |
| `--run` | `hpo` | folder on the volume: `<run>/<dataset>-<model>/` |
| `--stages` | `sobol,refine,final,report` | comma-separated, run in this order; `time` measures speed |
| `--trials` | 30 | trials per search stage (for tests) |
| `--max-epochs` | 400 | epochs per training (for tests) |
| `--gpu` | `T4` | Modal GPU type, e.g. `L4`, `A10`, `A100` |
| `--workers` | 4 | parallel trials per GPU; also the CPU cores requested |
| `--script` | `scripts/tune.py` | tuning script, e.g. `cloud/modal/tune_rescal.py` |

**Resuming.** Stages resume where they stopped (see the protocol), so after an
interruption (a stopped app, a crash, Modal's 24-hour limit per call) run the same
command again. Trials that were running, and at most the last 5 minutes of finished
trials, are rerun.

**Watching and stopping.** Apps, logs and costs are on the
[Modal dashboard](https://modal.com/apps). From the terminal:

```bash
uv run --group modal modal app logs kge-tune                 # live output of all pairs
uv run --group modal modal volume get kge-runs hpo/nations-rescal/log.txt -   # saved output
uv run --group modal modal app stop kge-tune                 # stops all pairs
```

Billing is per second while a container runs, so stop apps that are no longer needed.
On the Starter plan, logs are kept for one day; `log.txt` on the volume keeps them.

## Results

Download a pair's run folder:

```bash
uv run --group modal modal volume get kge-runs hpo/nations-rescal runs/modal/
```

The selected configuration is in `configs/`, the results (search ranks, validation MRR
per seed, test metrics) in `results/`, and the 5 trained models in `final/`. Copy the
configuration and results into `configs/` and `results/hpo/` of the repository, as for
local runs.

GPU and CPU runs are not bit-identical, so each pair is tuned on one device from start
to finish (`history.txt` shows the device).

## Costs

Modal bills GPU, CPU cores and memory per second; region pinning and non-preemptible
containers cost extra and are not used. On the Starter plan, GPU prices are about
$0.59/h (T4), $0.80/h (L4) and $1.10/h (A10). Each container requests `--workers` CPU
cores and 2 GiB of memory per worker.
