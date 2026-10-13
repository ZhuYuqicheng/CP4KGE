"""Hyperparameter tuning with scripts/tune.py on Modal GPUs.

Each (dataset, model) pair runs the tuning stages in its own GPU container, and all pairs
run in parallel. A pair's run folder is kept in the container and mirrored to the Modal
volume `kge-runs` every few minutes, so running the same command again resumes an
interrupted pair. See cloud/modal/README.md.

    uv run --group modal modal run cloud/modal/app.py --datasets nations --models rescal \\
        --stages time --workers 1 --run timing
"""

import shutil
import subprocess
import threading
import time
from pathlib import Path

import modal

CODE = Path("/root/code")  # the repository in the container
ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else CODE
VOLUME = Path("/vol")  # where the volume is mounted in the container
WORK = Path("/tmp/hpo")  # run folders in the container
SAVE_EVERY = 300  # seconds between two copies of a run folder to the volume

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("rsync")
    # the locked dependencies; libkge and kgcp are imported from the code below
    .uv_sync(str(ROOT), extra_options="--no-install-package libkge")
    .env(
        {
            "PYTHONPATH": f"{CODE}/src:{CODE}/third_party/libkge",
            "PYTHONUNBUFFERED": "1",
            "TQDM_MININTERVAL": "60",  # one progress bar line per minute in the logs
        }
    )
    # the working tree with the locally prepared datasets, uploaded on every run
    .add_local_dir(
        ROOT,
        str(CODE),
        ignore=[".git", ".venv", "runs", "**/__pycache__", "**/*.pckl", "**/.DS_Store"],
    )
)
app = modal.App("kge-tune", image=image)
volume = modal.Volume.from_name("kge-runs", create_if_missing=True)


@app.local_entrypoint()
def main(
    datasets: str,
    models: str,
    run: str = "hpo",
    stages: str = "sobol,refine,final,report",
    trials: int = 0,
    max_epochs: int = 0,
    gpu: str = "T4",
    workers: int = 4,
):
    """Tunes every pair of the comma-separated `datasets` and `models` in parallel.

    Run folders are stored on the volume as <run>/<dataset>-<model>. `trials` and
    `max_epochs` override the trials per search stage and the epochs per training (for
    tests); `workers` is the number of trials that run in parallel on a pair's GPU.
    """
    args = ["--device", "cuda", "--workers", str(workers)]
    args += ["--trials", str(trials)] if trials else []
    args += ["--max-epochs", str(max_epochs)] if max_epochs else []
    setup = f"{git_version()}, {gpu} GPU, {' '.join(args)}"
    tune_on = tune.with_options(gpu=gpu, cpu=workers, memory=2048 * workers)
    calls = {
        (dataset, model): tune_on.spawn(dataset, model, run, stages.split(","), args, setup)
        for dataset in datasets.split(",")
        for model in models.split(",")
    }
    for (dataset, model), call in calls.items():
        try:
            print("\n".join(call.get()))
        except Exception as error:  # report it, but wait for the other pairs
            print(f"{dataset}-{model} failed: {error!r}")


@app.function(volumes={VOLUME: volume}, gpu="T4", timeout=24 * 3600)
def tune(dataset, model, run, stages, args, setup) -> list[str]:
    """Runs tuning stages for one pair; returns the last line that each stage printed.

    `args` are passed on to scripts/tune.py; `setup` is recorded in history.txt.
    """
    pair = f"{dataset}-{model}"
    work, saved = WORK / run / pair, VOLUME / run / pair
    restore(saved, work)
    with open(work / "history.txt", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M')} {','.join(stages)}: {setup}\n")

    stop = threading.Event()
    saver = threading.Thread(target=save_every, args=(work, saved, stop))
    saver.start()
    summary = []
    try:
        for stage in stages:
            command = ["python", "scripts/tune.py", stage, "--datasets", dataset]
            command += ["--models", model, "--dir", str(work), *args]
            summary.append(f"{pair} {stage}: {tee(command, work / 'log.txt', f'[{pair}]')}")
    finally:
        stop.set()
        saver.join()
        save(work, saved)
    return summary


def tee(command, log: Path, prefix: str) -> str:
    """Runs a command in the repository, printing its output and appending it to `log`.

    Returns the last line of the output.
    """
    line = ""
    with (
        subprocess.Popen(
            command, cwd=CODE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        ) as process,
        open(log, "a") as f,
    ):
        for line in process.stdout:
            print(prefix, line, end="")
            f.write(line)
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)
    return line.strip()


# --- run folders on the volume ------------------------------------------------------


def restore(saved: Path, work: Path):
    """Copies a run folder from the volume into the container."""
    shutil.rmtree(work, ignore_errors=True)  # left by an earlier call to this container
    work.mkdir(parents=True)
    if saved.exists():
        mirror(saved, work)
    # a copy made while a trial was being logged can end in a partial line, which
    # Optuna would reject once more lines are appended
    journal = work / "journal.log"
    if journal.exists():
        log = journal.read_bytes()
        journal.write_bytes(log[: log.rfind(b"\n") + 1])


def save(work: Path, saved: Path):
    saved.mkdir(parents=True, exist_ok=True)
    mirror(work, saved)
    volume.commit()


def save_every(work: Path, saved: Path, stop: threading.Event):
    while not stop.wait(SAVE_EVERY):
        save(work, saved)


def mirror(source: Path, target: Path):
    """Makes `target` a copy of `source`, copying only changed files."""
    command = ["rsync", "-a", "--delete", f"{source}/", f"{target}/"]
    returncode = subprocess.run(command).returncode
    if returncode not in (0, 24):  # 24: files vanished during the copy, which is fine
        raise subprocess.CalledProcessError(returncode, command)


def git_version() -> str:
    """The commit of the uploaded code, and whether the code had uncommitted changes."""

    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout

    changes = " with uncommitted changes" if git("status", "--porcelain").strip() else ""
    return git("rev-parse", "--short", "HEAD").strip() + changes
