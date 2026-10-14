# Hyperparameter tuning

How the configurations in `configs/` were tuned, and what they achieve. Covered so
far: **DistMult and ComplEx on Nations, UMLS and Kinship**, tuned on a laptop GPU
(Apple M2 Pro, MPS), and **RESCAL on Nations, UMLS and Kinship**, tuned on a cloud GPU
(see [below](#rescal)).

## Results

Test metrics of the selected configurations: mean ± std over 5 seeds. These use the
standard filtered protocol, which removes all known triples (train, valid and test)
before ranking.

| Dataset | Model | Config | Test MRR | Hits@1 | Hits@3 | Hits@10 |
|---|---|---|---|---|---|---|
| Nations | DistMult | [nations-distmult.yaml](../configs/nations-distmult.yaml) | **0.813 ± 0.013** | 0.706 ± 0.018 | 0.899 ± 0.016 | 0.997 ± 0.003 |
| UMLS | DistMult | [umls-distmult.yaml](../configs/umls-distmult.yaml) | **0.896 ± 0.016** | 0.825 ± 0.024 | 0.962 ± 0.009 | 0.992 ± 0.004 |
| Kinship | DistMult | [kinship-distmult.yaml](../configs/kinship-distmult.yaml) | **0.704 ± 0.007** | 0.572 ± 0.011 | 0.799 ± 0.006 | 0.949 ± 0.002 |
| Nations | ComplEx | [nations-complex.yaml](../configs/nations-complex.yaml) | **0.780 ± 0.006** | 0.667 ± 0.013 | 0.866 ± 0.013 | 0.994 ± 0.002 |
| UMLS | ComplEx | [umls-complex.yaml](../configs/umls-complex.yaml) | **0.925 ± 0.014** | 0.875 ± 0.027 | 0.972 ± 0.003 | 0.992 ± 0.001 |
| Kinship | ComplEx | [kinship-complex.yaml](../configs/kinship-complex.yaml) | **0.882 ± 0.006** | 0.816 ± 0.010 | 0.941 ± 0.002 | 0.986 ± 0.001 |
| Nations | RESCAL | [nations-rescal.yaml](../configs/nations-rescal.yaml) | **0.661 ± 0.007** | 0.468 ± 0.015 | 0.813 ± 0.016 | 0.986 ± 0.004 |
| UMLS | RESCAL | [umls-rescal.yaml](../configs/umls-rescal.yaml) | **0.898 ± 0.006** | 0.807 ± 0.012 | 0.990 ± 0.004 | 0.997 ± 0.001 |
| Kinship | RESCAL | [kinship-rescal.yaml](../configs/kinship-rescal.yaml) | **0.847 ± 0.002** | 0.766 ± 0.003 | 0.915 ± 0.004 | 0.977 ± 0.003 |

The selected configurations differ between models and datasets (see
[below](#search-details)). DistMult uses reciprocal relations and cross-entropy
everywhere, with 1vsAll training on Nations and Kinship and KvsAll on UMLS. ComplEx
uses KvsAll + CE without reciprocal relations on UMLS and Kinship, and negative
sampling with BCE on Nations. The 5 trained models per pair are in
`models/<dataset>-<model>/seed{1..5}.pt`, with the configuration as `config.yaml` (see
[trained models](../README.md#trained-models)). `scripts/tune.py export` packages them
from the final runs, `runs/hpo/final/<dataset>-<model>/top<rank>/seed{1..5}/`, where
rank is the selected search rank (`selected` in the result JSON). The full results are
in [results/hpo/](../results/hpo/): one JSON per dataset and model, plus `trials.csv`
with all 2,160 search trials (240 per pair).

As a sanity check: Lin et al. (EMNLP 2018, Table 2) report DistMult at MRR 0.868 on
UMLS and 0.614 on Kinship on the same splits, and ComplEx and ConvE at 0.87–0.88 on
Kinship. Their protocol differs (object prediction only, fixed hyperparameters), so the
numbers are not directly comparable, but ours are in the expected range: DistMult is
strong on UMLS and clearly weaker on Kinship, and ComplEx on Kinship (0.882) matches
their result.

**Why validation MRR is lower than test MRR.** Model selection ranks validation
answers after removing train and valid triples only, so the test split stays unused
(see [deviations](#deviations-from-the-original-protocol)). A test triple ranked above
the validation answer of the same query then counts as an error, even though it is a
true fact. UMLS is dense: many validation queries also have answers in the test set.
With the standard filter, validation and test MRR are close, so the models do not
overfit the validation set:

| Validation MRR (5 seeds) | Nations | UMLS | Kinship |
|---|---|---|---|
| DistMult, filtered with train + valid (used for selection) | 0.756 ± 0.005 | 0.673 ± 0.006 | 0.625 ± 0.006 |
| DistMult, filtered with train + valid + test (standard) | 0.843 ± 0.004 | 0.903 ± 0.008 | 0.724 ± 0.006 |
| ComplEx, filtered with train + valid (used for selection) | 0.718 ± 0.007 | 0.688 ± 0.010 | 0.741 ± 0.003 |
| ComplEx, filtered with train + valid + test (standard) | 0.801 ± 0.009 | 0.930 ± 0.012 | 0.885 ± 0.002 |

Both filterings order the three finalists identically for five of the six pairs (see
the final stage below). For ComplEx on UMLS, ranks 1 and 3 swap places, but they differ
by 0.001 under either filter, so the stricter metric did not change the selection in
any meaningful way.

### Search details

Best validation MRR (train + valid filtering) per training type and loss in the Sobol
stage, each the best of 30 trials. The best combination of each pair is in bold. The
last row is the Bayesian stage, run on that combination.

| Training + loss | DistMult Nations | DistMult UMLS | DistMult Kinship | ComplEx Nations | ComplEx UMLS | ComplEx Kinship |
|---|---|---|---|---|---|---|
| NegSamp + CE | 0.752 | 0.658 | 0.558 | 0.705 | 0.685 | 0.713 |
| NegSamp + BCE | 0.723 | 0.628 | 0.532 | 0.718 | 0.683 | 0.712 |
| NegSamp + MR | 0.724 | 0.641 | 0.555 | 0.703 | 0.678 | 0.712 |
| 1vsAll + CE | **0.752** | 0.665 | **0.618** | **0.720** | 0.694 | 0.725 |
| 1vsAll + BCE | 0.741 | 0.645 | 0.553 | 0.705 | 0.689 | 0.681 |
| KvsAll + CE | 0.750 | **0.677** | 0.617 | 0.716 | **0.699** | **0.737** |
| KvsAll + BCE | 0.718 | 0.618 | 0.502 | 0.693 | 0.681 | 0.724 |
| *Bayesian stage* | 0.756 | 0.682 | 0.634 | 0.720 | 0.703 | 0.749 |

Final stage: the 3 best search trials of each pair, each retrained with 5 seeds.
Validation MRR, filtered with train + valid / with train + valid + test:

| Pair | Search rank 1 | Search rank 2 | Search rank 3 | Selected |
|---|---|---|---|---|
| DistMult Nations | **0.756 / 0.843** | 0.754 / 0.843 | 0.749 / 0.837 | rank 1 |
| DistMult UMLS | 0.672 / 0.898 | 0.668 / 0.896 | **0.673 / 0.903** | rank 3 |
| DistMult Kinship | **0.625 / 0.724** | 0.618 / 0.714 | 0.623 / 0.720 | rank 1 |
| ComplEx Nations | 0.686 / 0.772 | 0.698 / 0.781 | **0.718 / 0.801** | rank 3 |
| ComplEx UMLS | 0.687 / 0.931 | 0.681 / 0.927 | **0.688 / 0.930** | rank 3 |
| ComplEx Kinship | **0.741 / 0.885** | 0.741 / 0.883 | 0.741 / 0.880 | rank 1 |

All finalists of a pair share the training type and loss of the selected one, except
on ComplEx Nations: ranks 1 and 2 use 1vsAll + CE (embedding size 128), rank 3
negative sampling with BCE (size 512). ComplEx Kinship's three finalists tie at 0.741;
rank 1 beat rank 3 by 10⁻⁵.

Selected hyperparameters (all others are in the config files):

| | DistMult Nations | DistMult UMLS | DistMult Kinship |
|---|---|---|---|
| Reciprocal relations | yes | yes | yes |
| Training, loss | 1vsAll, CE | KvsAll, CE (no label smoothing) | 1vsAll, CE |
| Batch size | 64 | 512 | 256 |
| Optimizer, learning rate | Adagrad, 0.158 | Adam, 0.075 | Adagrad, 0.157 |
| LR scheduler patience | 3 | 0 | 10 |
| Embedding size | 256 | 128 | 32 |
| Initialization | uniform, [−0.51, 0.51] | normal, std 9.8·10⁻⁴ | Xavier uniform |
| Regularization | L3, weighted; entity 1.3·10⁻⁴, relation 4.8·10⁻¹³ | L2, weighted; entity 1.8·10⁻¹⁰, relation 2.0·10⁻⁹ | none |
| Dropout (entity, relation) | 0, 0 | 0, 0.48 | 0.01, 0.08 |

| | ComplEx Nations | ComplEx UMLS | ComplEx Kinship |
|---|---|---|---|
| Reciprocal relations | yes | no | no |
| Training, loss | NegSamp (1 head, 6 tail negatives), BCE | KvsAll, CE (label smoothing 0.30) | KvsAll, CE (label smoothing 0.28) |
| Batch size | 64 | 256 | 256 |
| Optimizer, learning rate | Adagrad, 0.229 | Adam, 0.0047 | Adagrad, 0.170 |
| LR scheduler patience | 1 | 10 | 8 |
| Embedding size | 512 | 256 | 512 |
| Initialization | uniform, [−0.29, 0.29] | Xavier uniform | uniform, [−0.53, 0.53] |
| Regularization | L3, weighted; entity 1.1·10⁻³, relation 4.2·10⁻¹² | none | L2, weighted; entity 1.4·10⁻⁶, relation 4.5·10⁻⁴ |
| Dropout (entity, relation) | 0, 0 | 0, 0.33 | 0.38, 0 |

**Compute.** Four trials shared the GPU at a time. Added up, the trial durations were
7.6 hours for DistMult (Nations 1.0, UMLS 2.2, Kinship 4.4) and 8.9 hours for ComplEx
(Nations 1.0, UMLS 2.8, Kinship 5.1); median trials took 11–52 s. A pair took from 20
minutes (Nations) to about 1.5 hours (Kinship) of wall-clock time, including its 15
final runs. None diverged. The runs were interrupted several times (stopped by hand,
and once by a worker process that died without an error message); the 16 trials cut
off were rerun with their sampled parameters, so every study has its full 30 trials.

### Observations

- **Training type and loss matter, and the best combination depends on the model.**
  With 1vsAll and KvsAll, cross-entropy beat BCE on every pair, as in Ruffinelli et
  al. DistMult does best with 1vsAll + CE or KvsAll + CE, which are close on every
  dataset. ComplEx prefers KvsAll + CE on UMLS and Kinship; on Nations, its selected
  configuration uses negative sampling with BCE, which beat the best Sobol combination
  (1vsAll + CE) on average over seeds. With negative sampling, the three losses are
  within 0.015 of each other for ComplEx.
- **ComplEx is much stronger than DistMult on Kinship** (test MRR 0.882 vs 0.704) and
  better on UMLS (0.925 vs 0.896); DistMult is better on Nations (0.813 vs 0.780).
  DistMult scores (s, p, o) and (o, p, s) identically, so it cannot fit asymmetric
  relations such as Kinship's kinship terms; ComplEx can.
- **The Bayesian stage helped most on Kinship.** It improved the best single-run
  validation MRR by +0.016 (DistMult) and +0.012 (ComplEx) on Kinship, by 0.004–0.005
  on DistMult Nations and UMLS and ComplEx UMLS, and not at all on ComplEx Nations.
- **The seed matters about as much as the finalists' differences.** Seed-to-seed
  standard deviations (0.003–0.018 validation, 0.006–0.016 test) are about as large as
  the gaps between the three finalists. For three of the six pairs, the search's best
  single run was *not* the best configuration on average. This winner's curse is why
  the final stage reruns several candidates.
- **Little weight regularization is needed, but some dropout and label smoothing.**
  The selected weight penalties are mostly negligible. Three configurations use
  dropout (relation dropout 0.48 and 0.33 on UMLS, entity dropout 0.38 for ComplEx on
  Kinship), and both ComplEx KvsAll configurations use label smoothing close to the
  upper end of the search range (0.28 and 0.30, of at most 0.3).
- **The epoch budget is mostly sufficient.** Median best epochs are 78–118 across
  trials. DistMult on Kinship trains longest: its selected runs peak at epochs
  155–400, two of the five at epochs 390 and 400, so it might gain from more epochs.
- **Results vary between devices.** An earlier run of this protocol for DistMult on
  the CPU (8 workers) reached test MRR 0.806 (Nations), 0.918 (UMLS) and 0.725
  (Kinship). Both runs try the same Sobol configurations, but floating-point results
  differ between CPU and GPU, so the trials score slightly differently. On UMLS this
  made KvsAll + CE the best combination instead of 1vsAll + CE, so the Bayesian stage
  refined a different combination; on Kinship, the final stage selected a different
  configuration (embedding size 32 with Adagrad instead of 64 with Adam). Test MRR
  differed by up to 0.022, more than the seed-to-seed deviation. The CPU run was
  replaced by this one, so that all models are tuned on the same device.

### RESCAL

Tuned with the same protocol on T4 GPUs on Modal ([cloud/modal/](../cloud/modal/)), one
container per dataset. RESCAL learns a d × d matrix per relation, so its cost grows with
d². Nations was searched first, over the full search space with 4 trials sharing the
GPU: its trials of embedding size 256 and 512 took 70% of the compute without reaching
the best validation MRR (size 32 won). The original RESCAL paper (Nickel et al., ICML
2011) used ranks 10–40 on these graphs. UMLS and Kinship were therefore searched with
embedding sizes 32, 64 and 128 only, with
[cloud/modal/tune_rescal.py](../cloud/modal/tune_rescal.py) (`scripts/tune.py` with the
smaller sizes) and 7 trials sharing the GPU.

Best validation MRR per training type and loss in the Sobol stage, each the best of 30
trials, and the Bayesian stage on the best combination:

| Training + loss | Nations (sizes 32–512) | UMLS (sizes 32–128) | Kinship (sizes 32–128) |
|---|---|---|---|
| NegSamp + CE | 0.594 | 0.677 | 0.709 |
| NegSamp + BCE | 0.579 | 0.678 | 0.700 |
| NegSamp + MR | 0.598 | 0.669 | 0.704 |
| 1vsAll + CE | 0.588 | **0.683** | 0.709 |
| 1vsAll + BCE | **0.605** | 0.678 | 0.709 |
| KvsAll + CE | 0.570 | 0.678 | **0.712** |
| KvsAll + BCE | 0.593 | 0.677 | 0.703 |
| *Bayesian stage* | 0.598 | 0.678 | 0.714 |

Final stage, validation MRR (train + valid filtering), mean ± std over 5 seeds:

| Pair | Search rank 1 | Search rank 2 | Search rank 3 | Selected |
|---|---|---|---|---|
| RESCAL Nations | 0.591 ± 0.013 | **0.597 ± 0.007** | 0.588 ± 0.005 | rank 2 |
| RESCAL UMLS | **0.675 ± 0.004** | 0.667 ± 0.004 | 0.670 ± 0.004 | rank 1 |
| RESCAL Kinship | 0.708 ± 0.003 | 0.709 ± 0.001 | **0.711 ± 0.002** | rank 3 |

On Nations, ranks 1 and 2 use 1vsAll + BCE (size 32) and rank 3 NegSamp + MR (size 64);
on UMLS, 1vsAll + CE (size 128), KvsAll + CE (size 32) and 1vsAll + BCE (size 128); on
Kinship, all three use KvsAll + CE (sizes 64, 32 and 128). Selected hyperparameters:

| | RESCAL Nations | RESCAL UMLS | RESCAL Kinship |
|---|---|---|---|
| Reciprocal relations | yes | no | no |
| Training, loss | 1vsAll, BCE | 1vsAll, CE | KvsAll, CE (label smoothing 0.29) |
| Batch size | 128 | 256 | 256 |
| Optimizer, learning rate | Adagrad, 0.253 | Adagrad, 0.035 | Adagrad, 0.041 |
| LR scheduler patience | 7 | 7 | 5 |
| Embedding size | 32 | 128 | 128 |
| Initialization | uniform, [−0.64, 0.64] | normal, std 0.088 | uniform, [−0.57, 0.57] |
| Regularization | L1, weighted; entity 3.1·10⁻¹⁴, relation 2.6·10⁻¹³ | L2, weighted; entity 1.3·10⁻⁸, relation 1.1·10⁻² | L2, weighted; entity 3.1·10⁻⁷, relation 1.3·10⁻² |
| Dropout (entity, relation) | 0.50, 0 | 0.43, 0 | 0.50, 0 |

- **RESCAL is close to DistMult on UMLS and between DistMult and ComplEx on Kinship**
  (test MRR 0.898 vs 0.896 and 0.925 on UMLS; 0.847 vs 0.704 and 0.882 on Kinship),
  but clearly the weakest model on Nations (0.661 vs 0.813 and 0.780). Like ComplEx,
  its relation matrices can model the asymmetric relations of Kinship, which DistMult
  cannot.
- **Training type and loss matter little on UMLS and Kinship**: the best trials of the
  7 combinations are within 0.014 of each other. On Nations, BCE beat CE with 1vsAll,
  unlike for DistMult and ComplEx.
- **Embedding size.** On Nations, all of the top 5 trials use size 32 or 64. On UMLS
  and Kinship, the 20 best trials spread over all three sizes, and both selected
  configurations use 128, the largest size searched there, so larger sizes might gain
  a little. All three selected configurations use strong entity dropout (0.43–0.50);
  on UMLS and Kinship also the largest relation penalties of all selected
  configurations (about 10⁻²).
- **The Bayesian stage did not help** on Nations and UMLS, and added 0.002 on Kinship.
  Seed-to-seed deviations (0.001–0.013) are again as large as the gaps between the
  finalists.
- **Compute.** Added up, the trial durations were 6.0 hours on Nations, 5.5 on UMLS and
  11.7 on Kinship (median trials 44, 52 and 109 s). A run took about 2 hours of T4
  time on Nations (4 workers, including restarts), 1.1 hours on UMLS and 2.4 hours on
  Kinship (7 workers: with sizes up to 128 a training needs little GPU memory, so each
  Sobol study gets its own worker). None diverged. On Nations, 19 trials failed, by
  running out of GPU memory before the memory limits described in
  [cloud/modal/](../cloud/modal/) were added, or by being cut off by restarts; the
  Kinship run was stopped once by hand, cutting off 7 trials. All were rerun with their
  sampled parameters. One of them (Nations, 1vsAll + BCE, size 512, below the top 5)
  finished only after the Bayesian stage, which therefore started from 29 of the 30
  Sobol trials of its combination.

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
| Hardware | one GPU shared by parallel workers: Apple M2 Pro (MPS) with 4 workers for DistMult and ComplEx; a T4 on Modal for RESCAL, with 4 workers (Nations) or 7 (UMLS, Kinship) | GPUs |

### Search space

Every parameter is sampled in every trial; parameters that do not apply (e.g.
`init_std` with Xavier initialization) are ignored. "M" is the model, e.g. `distmult`.

| Hyperparameter | libkge option | Values | Original |
|---|---|---|---|
| Reciprocal relations | `model` | {M, `reciprocal_relations_model`} | same |
| Batch size | `train.batch_size` | {64, 128, 256, 512} | {128, 256, 512, 1024} |
| Optimizer | `train.optimizer.default.type` | {Adam, Adagrad} | same |
| Learning rate | `train.optimizer.default.args.lr` | [3·10⁻⁴, 1], log | same |
| LR scheduler patience | `train.lr_scheduler_args.patience` | {0, …, 10} | same |
| Embedding size | `lookup_embedder.dim` | {32, 64, 128, 256, 512}; RESCAL on UMLS and Kinship: {32, 64, 128} | {128, 256, 512} |
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
  64 were added. RESCAL on UMLS and Kinship searches only sizes up to 128 (see
  [RESCAL](#rescal)).
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
    uv run scripts/tune.py $stage --datasets nations umls kinship --models distmult complex \
        --device mps --workers 4
done
```

The stages are `sobol` (stage 1), `refine` (stage 2), `final` (stage 3: reruns,
selection and test evaluation) and `report` (all trials as CSV). Once the results are
copied into `results/hpo/` (see below), `export` packages the 5 models of each selected
configuration into `models/`. A progress bar shows the finished trials (final runs in
stage 3) and the estimated remaining time. `--workers` sets the number of parallel
processes (default 8), `--device` where they train (default `cpu`; e.g. `cuda`, `mps`).
`--dir` and `--trials` allow a quick debug run in a separate directory, e.g.
`--dir /tmp/hpo-debug --trials 2`.

Everything is written to `runs/hpo/`: `journal.log` (all trials; every stage resumes
where it stopped), `final/` (the 5-seed models of the top-3 configurations),
`results/` and `configs/`. The selected configurations and results are copied into
`configs/` and `results/hpo/` of this repository. [cloud/modal/](../cloud/modal/) runs
the same stages on Modal GPUs, one container per (dataset, model) pair; RESCAL on UMLS
and Kinship ran [cloud/modal/tune_rescal.py](../cloud/modal/tune_rescal.py) instead of
`scripts/tune.py`. RESCAL's trials are in its downloaded Modal run folders,
`runs/modal/<dataset>-rescal/`, and its final models were copied to
`runs/hpo/final/<dataset>-rescal/` (or export them with `--dir runs/modal/<pair>`).

Train a selected configuration directly with libkge:

```bash
uv run kge start configs/nations-distmult.yaml --random_seed.default 1
```
