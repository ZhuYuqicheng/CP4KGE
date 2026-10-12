# Hyperparameter tuning

How the configurations in `configs/` were tuned, and what they achieve. Covered so
far: **DistMult on Nations, UMLS and Kinship**. A search for TransE was started on
Kinship and paused to save time; its trials are kept in `runs/hpo/journal.log`, and
running the same commands with `--models transe` resumes it.

## Results

Test metrics of the selected configurations: mean ± std over 5 seeds. These use the
standard filtered protocol, which removes all known triples (train, valid and test)
before ranking.

| Dataset | Model | Config | Test MRR | Hits@1 | Hits@3 | Hits@10 |
|---|---|---|---|---|---|---|
| Nations | DistMult | [nations-distmult.yaml](../configs/nations-distmult.yaml) | **0.806 ± 0.009** | 0.694 ± 0.016 | 0.904 ± 0.009 | 0.998 ± 0.002 |
| UMLS | DistMult | [umls-distmult.yaml](../configs/umls-distmult.yaml) | **0.918 ± 0.010** | 0.862 ± 0.018 | 0.971 ± 0.005 | 0.997 ± 0.001 |
| Kinship | DistMult | [kinship-distmult.yaml](../configs/kinship-distmult.yaml) | **0.725 ± 0.007** | 0.603 ± 0.011 | 0.811 ± 0.007 | 0.955 ± 0.002 |

All three selected configurations use reciprocal relations and 1vsAll training with
cross-entropy. Nations and UMLS share most other settings (batch size 64, Adagrad,
embedding size 256, uniform initialization, weighted L3 regularization, no dropout);
Kinship prefers Adam, embedding size 64 and relation dropout. The 5 trained models per
dataset are kept in `runs/hpo/final/<dataset>-distmult/top<rank>/seed{1..5}/`, as
`checkpoint_best.pt`, where rank is the selected search rank (3 for Nations and UMLS,
1 for Kinship). The full results are in [results/hpo/](../results/hpo/): one JSON per
dataset, plus `trials.csv` with all 720 search trials.

As a sanity check: Lin et al. (EMNLP 2018, Table 2) report DistMult at MRR 0.868 on
UMLS and 0.614 on Kinship on the same splits, and ComplEx and ConvE at 0.87–0.88 on
Kinship. Their protocol differs (object prediction only, fixed hyperparameters), so the
numbers are not directly comparable, but ours are in the expected range: DistMult is
strong on UMLS and clearly weaker on Kinship.

**Why validation MRR is lower than test MRR.** Model selection ranks validation
answers after removing train and valid triples only, so the test split stays unused
(see [deviations](#deviations-from-the-original-protocol)). A test triple ranked above
the validation answer of the same query then counts as an error, even though it is a
true fact. UMLS is dense: many validation queries also have answers in the test set.
With the standard filter, validation and test MRR agree, so the models do not overfit
the validation set:

| Validation MRR (5 seeds) | Nations | UMLS | Kinship |
|---|---|---|---|
| filtered with train + valid (used for selection) | 0.757 ± 0.006 | 0.678 ± 0.005 | 0.641 ± 0.005 |
| filtered with train + valid + test (standard) | 0.845 ± 0.002 | 0.926 ± 0.005 | 0.746 ± 0.005 |

Both filterings order the three finalists identically on all datasets (next table),
so the stricter metric did not change which configuration was selected.

### Search details

Best validation MRR (train + valid filtering) per training type and loss in the Sobol
stage, each the best of 30 trials. The best combination is in bold. The last row is
the Bayesian stage, run on that combination.

| Training + loss | Nations | UMLS | Kinship |
|---|---|---|---|
| NegSamp + CE | 0.752 | 0.655 | 0.558 |
| NegSamp + BCE | 0.728 | 0.633 | 0.523 |
| NegSamp + MR | 0.727 | 0.641 | 0.555 |
| 1vsAll + CE | **0.752** | **0.667** | **0.623** |
| 1vsAll + BCE | 0.739 | 0.650 | 0.551 |
| KvsAll + CE | 0.750 | 0.661 | 0.621 |
| KvsAll + BCE | 0.717 | 0.619 | 0.492 |
| *Bayesian stage, 1vsAll + CE* | 0.764 | 0.685 | 0.647 |

Final stage: the 3 best search trials (all 1vsAll + CE), each retrained with 5 seeds.
Validation MRR, filtered with train + valid / with train + valid + test:

| Dataset | Search rank 1 | Search rank 2 | Search rank 3 | Selected |
|---|---|---|---|---|
| Nations | 0.750 / 0.839 | 0.754 / 0.844 | **0.757 / 0.845** | rank 3 |
| UMLS | 0.666 / 0.915 | 0.677 / 0.920 | **0.678 / 0.926** | rank 3 |
| Kinship | **0.641 / 0.746** | 0.639 / 0.741 | 0.632 / 0.738 | rank 1 |

Selected hyperparameters (all others are in the config files):

| | Nations | UMLS | Kinship |
|---|---|---|---|
| Reciprocal relations | yes | yes | yes |
| Training, loss | 1vsAll, CE | 1vsAll, CE | 1vsAll, CE |
| Batch size | 64 | 64 | 128 |
| Optimizer, learning rate | Adagrad, 0.124 | Adagrad, 0.561 | Adam, 0.0026 |
| LR scheduler patience | 2 | 3 | 2 |
| Embedding size | 256 | 256 | 64 |
| Initialization | uniform, [−0.50, 0.50] | uniform, [−0.34, 0.34] | Xavier normal |
| Regularization | L3, weighted; entity 2.8·10⁻⁴, relation 8.9·10⁻⁸ | L3, weighted; entity 5.6·10⁻⁸, relation 2.3·10⁻¹⁴ | L1; entity 5.8·10⁻¹³, relation 1.2·10⁻¹⁸ |
| Dropout (entity, relation) | 0, 0 | 0, 0 | 0, 0.39 |

**Compute.** The 720 search trials used 3.0 CPU-hours (Nations 0.4, UMLS 1.0, Kinship
1.6); the median trial took 9 s. None diverged and none failed; the 5 Kinship trials
cut off when the search was stopped were rerun with their sampled parameters. Nations
and UMLS (480 trials, 30 final runs) took 15 minutes of wall-clock time on 8 CPU
workers. Kinship was run in two parts: about three quarters of its Sobol trials ran
alongside the paused TransE search, and the rest, plus its Bayesian and final stages,
took another 12 minutes.

### Observations

- **Training type and loss matter more than the other hyperparameters.** Cross-entropy
  beat BCE and margin ranking for every training type on all three datasets, as in
  Ruffinelli et al. 1vsAll and KvsAll with CE are close in the Sobol stage, and on
  Nations negative sampling with CE is too. After refinement and reruns, all three
  finalists on every dataset use 1vsAll + CE.
- **The Bayesian stage helped.** It improved the best single-run validation MRR by
  +0.012 (Nations), +0.018 (UMLS) and +0.024 (Kinship) over 210 Sobol trials, with
  only 30 more trials.
- **The seed matters about as much as the finalists' differences.** Seed-to-seed
  standard deviations (0.003–0.009 validation, ~0.01 test) are about as large as the
  gaps between the three finalists. On Nations and UMLS, the search's best single run
  was *not* the best configuration on average; on Kinship it was, by 0.002. This
  winner's curse is why the final stage reruns several candidates.
- **Little regularization is needed.** The selected weight penalties are negligible,
  and only Kinship uses dropout (on relations). On these small graphs, early stopping
  plus the learning-rate schedule is enough.
- **The epoch budget is sufficient.** Median best epochs are 55–85 across trials. The
  selected Kinship runs train longest, peaking at epochs 215–320, and still stop early
  before the 400-epoch limit.

## Protocol

We follow the protocol LibKGE's authors used to tune the models in LibKGE's model zoo
(Ruffinelli, Broscheit & Gemulla, *You CAN Teach an Old Dog New Tricks! On Training
Knowledge Graph Embeddings*, ICLR 2020). Their search configs are public in
[uma-pi1/kge-iclr20](https://github.com/uma-pi1/kge-iclr20/tree/master/config_files).
We adapt the protocol to small graphs and add a seed-robust selection step. Every
deviation is listed [below](#deviations-from-the-original-protocol).

The search is run separately for every (dataset, model) pair, in three stages:

1. **Sobol stage (broad search).** Each of the 7 combinations of training type and
   loss gets its own study of 30 trials. The combinations are negative sampling with
   CE, BCE or margin ranking (MR); 1vsAll with CE or BCE; and KvsAll with CE or BCE.
   Trials are drawn from a scrambled Sobol sequence (quasi-random, covers the space
   more evenly than random search) over the full search space below. This gives 210
   trials.
2. **Bayesian stage (refinement).** The combination with the best validation MRR gets
   30 more trials from a Bayesian optimizer, multivariate TPE. The optimizer is
   warm-started with that combination's 30 Sobol trials. Trials run in 3 parallel
   chunks of 10, and Optuna's *constant liar* keeps running trials from being proposed
   twice. This gives 240 search trials per pair.
3. **Final stage (seed-robust selection and test).** The 3 best configurations of all
   240 trials are retrained with 5 new seeds (1–5). The configuration with the highest
   **mean** validation MRR is selected. Its 5 models are evaluated on the test split,
   and we report mean ± standard deviation. Validation MRR of a single run fluctuates
   with the seed (about ±0.01 on Nations). Taking the best of 240 noisy runs is
   therefore optimistically biased, and the reruns correct for that. The test split is
   used **only** here, and only for the selected configuration.

### Training and selection settings (fixed for all trials)

| Setting | Value | Original |
|---|---|---|
| Max. epochs | 400 | 400 |
| Validation | every 5 epochs, on `valid` | same |
| Selection metric | filtered MRR, filtering with train + valid triples (`mean_reciprocal_rank_filtered`) | filtered with test triples too |
| Early stopping | after 10 validations without improvement | same |
| Early abort | stop if MRR is not better than a random ranking after 50 epochs (H_N/N for N entities: Nations 0.232, UMLS 0.040, Kinship 0.050) | MRR < 0.05 after 50 epochs |
| LR schedule | ReduceLROnPlateau on validation MRR, factor 0.95, threshold 1e-4 | same |
| Invalid values | libkge's `job.auto_correct`: label smoothing in (0, 1/N] is raised to 1/N | same |
| Divergence | a trial whose loss becomes NaN scores MRR 0 | — |
| Failed or interrupted trials | retried with the same sampled parameters, so every study gets its full 30 points | — |
| Seed | 0 in the search stages; 1–5 in the final stage | not fixed |
| Hardware | CPU, 8 parallel workers with 1 thread each | GPUs |

### Search space

Every parameter is sampled in every trial; parameters that do not apply (e.g.
`init_std` with Xavier initialization) are ignored. "M" is `transe` or `distmult`.

| Hyperparameter | libkge option | Values | Original |
|---|---|---|---|
| Reciprocal relations | `model` | {M, `reciprocal_relations_model`} | same |
| Batch size | `train.batch_size` | {64, 128, 256, 512} | {128, 256, 512, 1024} |
| Optimizer | `train.optimizer.default.type` | {Adam, Adagrad} | same |
| Learning rate | `train.optimizer.default.args.lr` | [3·10⁻⁴, 1], log | same |
| LR scheduler patience | `train.lr_scheduler_args.patience` | {0, …, 10} | same |
| Embedding size | `lookup_embedder.dim` | {32, 64, 128, 256, 512} | {128, 256, 512} |
| Initialization | `lookup_embedder.initialize` | {Xavier normal, Xavier uniform, normal, uniform} | same |
| Normal init. std | `…initialize_args.normal_.std` | [10⁻⁵, 1], log | same |
| Uniform init. interval | `…initialize_args.uniform_.{a,b}` | [a, −a] with a ∈ [−1, −10⁻⁵] | [a, 1] (see below) |
| Regularization | `lookup_embedder.regularize`, `regularize_args.p` | {none, L1, L2, L3} | same |
| Frequency weighting | `lookup_embedder.regularize_args.weighted` | {true, false} | same |
| Entity / relation reg. weight | `M.{entity,relation}_embedder.regularize_weight` | [10⁻²⁰, 10⁻¹], log, each | same |
| Entity / relation dropout | `M.{entity,relation}_embedder.dropout` | [−0.5, 0.5], each; < 0 means off | same |
| Negatives per slot (NegSamp) | `negative_sampling.num_samples.{s,o}` | [1, #entities], log, each | [1, 1000] |
| Margin (MR loss) | `train.loss_arg` | [0, 10] | same |
| Label smoothing (KvsAll) | `KvsAll.label_smoothing` | [−0.3, 0.3]; < 0 means off | same |
| Distance norm (TransE) | `transe.l_norm` | {L1, L2} | same |
| Entity / relation normalization (TransE) | `transe.{entity,relation}_embedder.normalize.p` | {off, L2}, each | same |

### Deviations from the original protocol

- **Smaller batches and embeddings.** The training sets have only 1.6k–8.5k triples. A
  batch of 1024 would give just 2–9 updates per epoch, so batch sizes go from 64 to
  512. Graphs with 14–135 entities may also prefer small embeddings, so sizes 32 and
  64 were added.
- **At most #entities negatives.** With 14 entities (Nations), more negatives than
  entities only repeat samples and cost time.
- **Symmetric uniform initialization.** The original configs set only the lower bound
  `a`. libkge then keeps PyTorch's default upper bound of 1, so it actually samples
  U[a, 1]. We set U[a, −a], which matches the intent.
- **No test triples in model selection.** The original selected on MRR filtered with
  test triples (libkge's default). We filter with train and valid only, so the test
  split stays untouched until the final evaluation. The test split will later serve
  conformal prediction.
- **Random-guessing threshold.** A fixed MRR threshold of 0.05 is meaningless on
  Nations, where a random ranking already scores 0.23. We use the dataset's
  random-ranking MRR instead.
- **TransE with all training types.** The original searched TransE only with negative
  sampling, because pairwise scoring against all entities is expensive on large
  graphs. On these small graphs, 1vsAll and KvsAll are affordable, so both models get
  all 7 combinations.
- **TPE instead of Gaussian-process BO, plus seed-robust final selection.** We use
  Optuna's TPE sampler instead of Ax's Gaussian-process Bayesian optimization. libkge's
  Ax search job needs the optional `libkge[search]` dependencies, which are untested
  with current library versions. TPE is a lightweight Bayesian optimizer. The final
  rerun stage is our addition.

## Reproducing

```bash
for stage in sobol refine final report; do
    uv run scripts/tune.py $stage --datasets nations umls kinship --models distmult
done
```

The stages are `sobol` (stage 1), `refine` (stage 2), `final` (stage 3: reruns,
selection and test evaluation) and `report` (all trials as CSV). A progress bar shows
the finished trials (final runs in stage 3) and the estimated remaining time. `--workers`
sets the number of parallel processes (default 8). `--dir` and `--trials` allow a quick debug
run in a separate directory, e.g. `--dir /tmp/hpo-debug --trials 2`.

Everything is written to `runs/hpo/`: `journal.log` (all trials; every stage resumes
where it stopped), `final/` (the 5-seed models of the top-3 configurations),
`results/` and `configs/`. The selected configurations and results are copied into
`configs/` and `results/hpo/` of this repository.

Train a selected configuration directly with libkge:

```bash
uv run kge start configs/nations-distmult.yaml --random_seed.default 1
```
