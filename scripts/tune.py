"""Hyperparameter search for libkge models on small knowledge graphs.

Implements the protocol of docs/hyperparameter_tuning.md, an adaptation of LibKGE's
ICLR 2020 study (Ruffinelli et al., "You CAN Teach an Old Dog New Tricks!"):

    sobol   30 scrambled-Sobol trials for every (training type, loss) combination
    refine  30 Bayesian-optimization (TPE) trials on the best combination
    final   rerun the top-3 configurations with 5 seeds, select by mean validation
            MRR, evaluate the selected configuration on test
    report  export all trials to runs/hpo/results/trials.csv

    python scripts/tune.py sobol --datasets nations umls kinship --models transe distmult

Everything is written to runs/hpo/ (see --dir): the trials (journal.log, which makes
every stage resumable), the final runs, and the selected configs and results.
"""

import argparse
import json
import multiprocessing
import shutil
import statistics
import tempfile
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from functools import cache
from pathlib import Path

import optuna
import pandas as pd
import torch
import yaml
from optuna.samplers import QMCSampler, TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState
from tqdm import tqdm

from kge import Config, Dataset
from kge.job import Job
from kge.misc import kge_base_dir
from kge.util.io import load_checkpoint
from kge.util.seed import seed_from_config

RUNS = Path(__file__).resolve().parents[1] / "runs" / "hpo"

COMBINATIONS = [  # (train.type, train.loss), as in Ruffinelli et al. (2020)
    ("negative_sampling", "kl"),
    ("negative_sampling", "bce"),
    ("negative_sampling", "margin_ranking"),
    ("1vsAll", "kl"),
    ("1vsAll", "bce"),
    ("KvsAll", "kl"),
    ("KvsAll", "bce"),
]
TRIALS = {"sobol": 30, "refine": 30}
REFINE_CHUNK = 10  # refine trials of one study run in parallel chunks of this size
TOP_K, SEEDS = 3, (1, 2, 3, 4, 5)
SEARCH_SEED = 0
# validation metric for model selection; test triples are not used for filtering
METRIC = "mean_reciprocal_rank_filtered"


# --- search space -------------------------------------------------------------------


@cache
def dataset_meta(dataset: str) -> dict:
    with open(Path(kge_base_dir()) / "data" / dataset / "dataset.yaml") as f:
        return yaml.safe_load(f)["dataset"]


def fixed_options(dataset: str, model: str) -> dict:
    """Training and validation settings shared by all trials."""
    n = dataset_meta(dataset)["num_entities"]
    random_mrr = sum(1 / k for k in range(1, n + 1)) / n  # uniformly random ranking
    return {
        "job.type": "train",
        "job.device": "cpu",
        "job.auto_correct": True,  # as in the original configs, e.g. for label smoothing
        "console.quiet": True,
        "dataset.name": dataset,
        "dataset.pickle": False,  # parallel workers must not race on the cache
        "import": [model, "reciprocal_relations_model"],
        "reciprocal_relations_model.base_model.type": model,
        "train.max_epochs": 400,
        "train.lr_scheduler": "ReduceLROnPlateau",
        "train.lr_scheduler_args.mode": "max",
        "train.lr_scheduler_args.factor": 0.95,
        "train.lr_scheduler_args.threshold": 1e-4,
        "train.checkpoint.every": 0,
        "train.checkpoint.keep_init": False,
        "valid.every": 5,
        "valid.metric": METRIC,
        "valid.early_stopping.patience": 10,
        # stop trials that are no better than random guessing after 50 epochs
        "valid.early_stopping.threshold.epochs": 50,
        "valid.early_stopping.threshold.metric_value": round(random_mrr, 4),
        "entity_ranking.filter_with_test": False,
        "eval.batch_size": 256,
        "random_seed.default": 0,
    }


def suggest(trial: optuna.Trial, dataset: str, model: str, train_type: str, loss: str):
    """Samples libkge options from the (adapted) search space of Ruffinelli et al.

    As in the original Sobol search, every parameter is sampled in every trial, so the
    space of a (training type, loss) combination is unconditional. Negative dropout and
    label smoothing values mean "off".
    """
    f, i, c = trial.suggest_float, trial.suggest_int, trial.suggest_categorical
    regularize = c("regularize", ["", "l1", "l2", "l3"])
    uniform_a = f("init_a", -1.0, -1e-5)
    options = {
        "model": c("model", [model, "reciprocal_relations_model"]),
        "train.type": train_type,
        "train.loss": loss,
        "train.batch_size": c("batch_size", [64, 128, 256, 512]),
        "train.optimizer.default.type": c("optimizer", ["Adam", "Adagrad"]),
        "train.optimizer.default.args.lr": f("lr", 3e-4, 1.0, log=True),
        "train.lr_scheduler_args.patience": i("lr_patience", 0, 10),
        "lookup_embedder.dim": c("dim", [32, 64, 128, 256, 512]),
        "lookup_embedder.initialize": c(
            "initialize", ["xavier_normal_", "xavier_uniform_", "normal_", "uniform_"]
        ),
        # every initializer needs its own args entry, else libkge passes all of them
        "lookup_embedder.initialize_args.normal_.mean": 0.0,
        "lookup_embedder.initialize_args.normal_.std": f("init_std", 1e-5, 1.0, log=True),
        "lookup_embedder.initialize_args.uniform_.a": uniform_a,
        "lookup_embedder.initialize_args.uniform_.b": -uniform_a,
        "lookup_embedder.initialize_args.xavier_normal_.gain": 1.0,
        "lookup_embedder.initialize_args.xavier_uniform_.gain": 1.0,
        "lookup_embedder.regularize": "lp" if regularize else "",
        "lookup_embedder.regularize_args.p": int(regularize[1:]) if regularize else 2,
        "lookup_embedder.regularize_args.weighted": c("reg_weighted", [True, False]),
        f"{model}.entity_embedder.regularize_weight": f("reg_entity", 1e-20, 0.1, log=True),
        f"{model}.relation_embedder.regularize_weight": f(
            "reg_relation", 1e-20, 0.1, log=True
        ),
        f"{model}.entity_embedder.dropout": max(0.0, f("dropout_entity", -0.5, 0.5)),
        f"{model}.relation_embedder.dropout": max(0.0, f("dropout_relation", -0.5, 0.5)),
    }
    if train_type == "negative_sampling":
        n = dataset_meta(dataset)["num_entities"]  # more negatives than entities add nothing
        options["negative_sampling.num_samples.s"] = i("negatives_s", 1, n, log=True)
        options["negative_sampling.num_samples.o"] = i("negatives_o", 1, n, log=True)
    if train_type == "KvsAll":
        options["KvsAll.label_smoothing"] = max(0.0, f("label_smoothing", -0.3, 0.3))
    if loss == "margin_ranking":
        options["train.loss_arg"] = f("margin", 0.0, 10.0)
    if model == "transe":
        options["transe.l_norm"] = c("l_norm", [1.0, 2.0])
        options["transe.entity_embedder.normalize.p"] = c("normalize_entity", [-1.0, 2.0])
        options["transe.relation_embedder.normalize.p"] = c(
            "normalize_relation", [-1.0, 2.0]
        )
    return options


# --- training and evaluation with libkge -----------------------------------------------


def train(options: dict, folder: Path) -> dict:
    """Trains a model; returns the validation trace entry of its best epoch."""
    config = Config(folder=str(folder))
    config.load_options(options)
    config.init_folder()
    seed_from_config(config)
    job = Job.create(config, Dataset.create(config))
    job.run()
    return max(job.valid_trace, key=lambda entry: entry[METRIC])


def evaluate(folder: Path, split: str = "test") -> dict:
    """Evaluates the best checkpoint of a training run (like `kge test`)."""
    config = Config(folder=str(folder))
    config.load(str(folder / "config.yaml"))
    config.set("job.type", "eval")
    config.set("eval.split", split)
    checkpoint = load_checkpoint(str(folder / "checkpoint_best.pt"), device="cpu")
    return Job.create_from(checkpoint, new_config=config).run()


# --- search stages -----------------------------------------------------------------


def storage() -> JournalStorage:
    return JournalStorage(JournalFileBackend(str(RUNS / "journal.log")))


def study_name(dataset, model, train_type, loss) -> str:
    return f"{dataset}-{model}-{train_type}-{loss}"


def by_cost(datasets, models):
    """(dataset, model) pairs, most expensive first, so that workers stay busy."""
    pairs = [(d, m) for d in datasets for m in models]
    # TransE scores with pairwise distances and is several times slower than DistMult
    return sorted(
        pairs, key=lambda p: (dataset_meta(p[0])["files.train.size"], p[1] == "transe"),
        reverse=True,
    )


def objective(trial, dataset, model, train_type, loss, phase) -> float:
    options = fixed_options(dataset, model) | suggest(trial, dataset, model, train_type, loss)
    trial.set_user_attr("phase", phase)
    trial.set_user_attr("options", options)
    start = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            best = train(options, Path(tmp) / "run")
        except FloatingPointError:  # libkge aborts when the loss becomes NaN
            trial.set_user_attr("diverged", True)
            return 0.0
        except Exception as e:  # fails the trial; it is retried on resume
            trial.set_user_attr("error", repr(e))
            raise
        finally:
            trial.set_user_attr("seconds", round(time.time() - start, 1))
    trial.set_user_attr("epoch", best["epoch"])
    for key in ("hits_at_1_filtered", "hits_at_3_filtered", "hits_at_10_filtered"):
        trial.set_user_attr(key, best[key])
    return best[METRIC]


def prepare(dataset, model, train_type, loss, phase) -> int:
    """Returns how many trials of a stage are still to be run.

    Trials that failed or were interrupted are enqueued again with their sampled
    parameters, so that no point of the Sobol sequence is lost. Must not be called
    while workers are running.
    """
    try:
        study = optuna.load_study(
            study_name=study_name(dataset, model, train_type, loss), storage=storage()
        )
    except KeyError:
        return TRIALS[phase]
    trials = [t for t in study.trials if t.user_attrs.get("phase") == phase]
    retried = {t.user_attrs.get("retry_of") for t in study.trials}
    for t in trials:
        if t.state in (TrialState.FAIL, TrialState.RUNNING) and t.number not in retried:
            study.enqueue_trial(t.params, user_attrs={"retry_of": t.number})
    done = sum(t.state == TrialState.COMPLETE for t in trials)
    return max(0, TRIALS[phase] - done)


def optimize(task) -> str:
    """Runs `n_trials` trials of one stage of one study."""
    dataset, model, train_type, loss, phase, n_trials, seed = task
    sampler = (
        QMCSampler(qmc_type="sobol", scramble=True, seed=seed)
        if phase == "sobol"
        else TPESampler(multivariate=True, constant_liar=True, seed=seed)
    )
    study = optuna.create_study(
        study_name=study_name(dataset, model, train_type, loss),
        storage=storage(),
        sampler=sampler,
        direction="maximize",
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(trial, dataset, model, train_type, loss, phase),
        n_trials=n_trials,
        catch=(Exception,),
        callbacks=[count_finished],
    )
    return f"{study.study_name} [{phase}]: best {study.best_value:.3f}"


def completed_trials(dataset, model):
    """All completed trials of a dataset/model pair, best first."""
    trials = []
    for train_type, loss in COMBINATIONS:
        name = study_name(dataset, model, train_type, loss)
        try:
            study = optuna.load_study(study_name=name, storage=storage())
        except KeyError:
            continue
        trials += [(name, t) for t in study.trials if t.state == TrialState.COMPLETE]
    return sorted(trials, key=lambda item: item[1].value, reverse=True)


def best_combination(dataset, model):
    _, best = completed_trials(dataset, model)[0]
    options = best.user_attrs["options"]
    return options["train.type"], options["train.loss"]


def final_run(task) -> tuple:
    """Trains one (candidate, seed) of the final stage in a persistent folder."""
    dataset, model, rank, options, seed = task
    folder = RUNS / "final" / f"{dataset}-{model}" / f"top{rank}" / f"seed{seed}"
    if not (folder / ".done").exists():
        shutil.rmtree(folder, ignore_errors=True)  # incomplete run from an interruption
        train(options | {"random_seed.default": seed}, folder)
        (folder / ".done").touch()
    trace = [yaml.safe_load(line) for line in (folder / "trace.yaml").read_text().splitlines()]
    valid = [e for e in trace if e.get("event") == "eval_completed" and e["split"] == "valid"]
    count_finished()
    return task, max(valid, key=lambda entry: entry[METRIC]), folder


def init_worker(runs: Path, trials: dict, finished):
    global RUNS, TRIALS, FINISHED
    RUNS, TRIALS, FINISHED = runs, trials, finished
    torch.set_num_threads(1)  # tiny models: one thread per worker is fastest
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


def count_finished(*_):
    """Counts a finished trial or final run for the progress bar of the main process."""
    with FINISHED.get_lock():
        FINISHED.value += 1


def run_parallel(fn, tasks, workers, total, unit="trial"):
    """Runs tasks in submission order on worker processes; yields results as they finish.

    A progress bar shows the `total` trials (or final runs) that the workers count as
    finished. Print with `tqdm.write` while it is shown.
    """
    finished = multiprocessing.Value("i", 0)
    initargs = (RUNS, TRIALS, finished)
    with (
        ProcessPoolExecutor(workers, initializer=init_worker, initargs=initargs) as pool,
        tqdm(total=total, unit=unit, smoothing=0) as bar,
    ):
        pending = {pool.submit(fn, task) for task in tasks}
        while pending:
            done, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            bar.update(finished.value - bar.n)
            for future in done:
                yield future.result()


# --- commands ----------------------------------------------------------------------


def cmd_sobol(args):
    tasks = [
        (d, m, t, l, "sobol", n, SEARCH_SEED)
        for d, m in by_cost(args.datasets, args.models)
        for t, l in COMBINATIONS
        if (n := prepare(d, m, t, l, "sobol"))
    ]
    run_studies(tasks, args.workers)


def cmd_refine(args):
    tasks = []
    for d, m in by_cost(args.datasets, args.models):
        combination = best_combination(d, m)
        n = prepare(d, m, *combination, "refine")
        tasks += [
            (d, m, *combination, "refine", min(REFINE_CHUNK, n - k), SEARCH_SEED + 1 + k)
            for k in range(0, n, REFINE_CHUNK)
        ]
    run_studies(tasks, args.workers)


def run_studies(tasks, workers):
    """Runs `optimize` tasks; prints the best value of each study when its trials are done."""
    total = sum(n_trials for *_, n_trials, _seed in tasks)
    for line in run_parallel(optimize, tasks, workers, total):
        tqdm.write(line)


def cmd_final(args):
    tasks = []
    for d, m in by_cost(args.datasets, args.models):
        for rank, (_, trial) in enumerate(completed_trials(d, m)[:TOP_K], start=1):
            tasks += [(d, m, rank, trial.user_attrs["options"], seed) for seed in SEEDS]
    runs = {}
    results = run_parallel(final_run, tasks, args.workers, len(tasks), unit="run")
    for (d, m, rank, options, seed), best, folder in results:
        runs.setdefault((d, m), {}).setdefault(rank, []).append((options, seed, best, folder))
        tqdm.write(f"{d}-{m} top{rank} seed{seed}: valid MRR {best[METRIC]:.3f}")

    (RUNS / "results").mkdir(exist_ok=True)
    for (d, m), candidates in runs.items():
        valid_mrr = {
            r: [best[METRIC] for _, _, best, _ in sorted(c, key=lambda x: x[1])]
            for r, c in candidates.items()
        }
        selected = max(valid_mrr, key=lambda r: statistics.mean(valid_mrr[r]))
        tests = {
            seed: evaluate(folder) for _, seed, _, folder in sorted(candidates[selected], key=lambda x: x[1])
        }
        result = {
            "dataset": d,
            "model": m,
            "selected": selected,
            "candidates": {
                r: {"valid_mrr": valid_mrr[r], "options": candidates[r][0][0]}
                for r in sorted(candidates)
            },
            "test": {seed: {k: v for k, v in t.items() if "filtered" in k} for seed, t in tests.items()},
        }
        with open(RUNS / "results" / f"{d}-{m}.json", "w") as f:
            json.dump(result, f, indent=2)
        write_config(d, m, candidates[selected][0][0])
        test_mrr = [t[METRIC] for t in tests.values()]
        print(
            f"{d}-{m}: selected top{selected}, test MRR "
            f"{statistics.mean(test_mrr):.3f} ± {statistics.stdev(test_mrr):.3f}"
        )


def write_config(dataset, model, options):
    """Writes the selected configuration as a plain libkge config file."""
    runtime = {"job.device", "console.quiet", "dataset.pickle", "random_seed.default"}
    runtime |= {k for k in options if k.startswith("train.checkpoint")}
    config = {k: v for k, v in options.items() if k not in runtime}
    header = (
        f"# {model} on {dataset}, tuned with scripts/tune.py "
        "(see docs/hyperparameter_tuning.md).\n"
        f"#   kge start configs/{dataset}-{model}.yaml --random_seed.default 1\n"
    )
    (RUNS / "configs").mkdir(exist_ok=True)
    with open(RUNS / "configs" / f"{dataset}-{model}.yaml", "w") as f:
        f.write(header + yaml.safe_dump(config, sort_keys=False))


def cmd_report(args):
    rows = [
        {
            "study": name,
            "dataset": d,
            "model": m,
            "train_type": t.user_attrs["options"]["train.type"],
            "loss": t.user_attrs["options"]["train.loss"],
            "phase": t.user_attrs["phase"],
            "trial": t.number,
            "valid_mrr": round(t.value, 4),
            **{k: t.user_attrs.get(k) for k in ("epoch", "seconds", "diverged")},
            **{f"hp.{k}": v for k, v in t.params.items()},
        }
        for d in args.datasets
        for m in args.models
        for name, t in completed_trials(d, m)
    ]
    (RUNS / "results").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RUNS / "results" / "trials.csv", index=False)
    print(f"wrote {len(rows)} trials to {RUNS / 'results' / 'trials.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["sobol", "refine", "final", "report"])
    parser.add_argument("--datasets", nargs="+", default=["nations", "umls", "kinship"])
    parser.add_argument("--models", nargs="+", default=["transe", "distmult"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dir", type=Path, default=RUNS, help="where trials are stored")
    parser.add_argument("--trials", type=int, help="trials per stage (default: 30)")
    args = parser.parse_args()
    RUNS = args.dir
    TRIALS = {stage: args.trials or n for stage, n in TRIALS.items()}
    RUNS.mkdir(parents=True, exist_ok=True)
    globals()[f"cmd_{args.command}"](args)
