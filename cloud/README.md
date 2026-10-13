# Cloud GPUs

Running `scripts/tune.py` on rented GPUs. The tuning protocol, its resumable trial
journal and the selected configurations are the same as on a local machine (see
[docs/hyperparameter_tuning.md](../docs/hyperparameter_tuning.md)); only the place where
trials run changes.

| Provider | Folder | Status |
|---|---|---|
| [Modal](https://modal.com) | [modal/](modal/) | in use |

When a GPU pays off: the small graphs (Nations, UMLS, Kinship) with DistMult or ComplEx
train in seconds per trial on a laptop CPU, where a GPU does not help. GPUs pay off for
models with large parameter updates, such as RESCAL (a d × d matrix per relation), and
for larger graphs (FB15k-237, WN18RR, CoDEx, YAGO3-10).

Measured with `tune.py time` (RESCAL on Nations, 1vsAll, batch size 64, one model at a
time):

| dim | laptop CPU, 1 thread (M2 Pro) | Modal T4 |
|---|---|---|
| 32–128 | 0.02–0.13 s/epoch | 0.08 s/epoch |
| 256 | 0.44 s/epoch | 0.14 s/epoch |
| 512 | 1.89 s/epoch | 0.42 s/epoch |
