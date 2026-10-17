"""CondKGCP's search space (scripts/condkgcp_search.py) for LibKGE's published models on
Modal GPUs.

Each model runs in its own GPU container, and all models run in parallel. The checkpoints
are read from the volume `kge-runs` (libkge-models/<dataset>-<model>.pt), and each
model's CSV is written to condkgcp-search/ on it. Models whose CSV is there already are
skipped, so running the same command again runs only the missing ones. Each container
prints its model's progress bar once a minute, with the time left.

    uv run --group modal modal run --detach cloud/modal/condkgcp.py --models wn18-transe
    uv run --group modal modal volume get kge-runs condkgcp-search <dir>
    uv run scripts/condkgcp_search.py --out <dir>/condkgcp-search  # the page

See cloud/modal/README.md.
"""

import time
from pathlib import Path

import modal

CODE = Path("/root/code")  # the repository in the container
ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else CODE
VOLUME = Path("/vol")  # where the volume is mounted in the container
CHECKPOINTS = "libkge-models"  # folders on the volume
OUT = "condkgcp-search"
DATASETS = ["wn18", "fb15k"]
MODELS = ["rescal", "transe", "distmult", "complex", "conve", "rotate"]
# queries per model call, where the default is too many for a T4's 16 GB: RotatE scores
# all entities with a [queries, entities, dim] tensor
BATCH_SIZES = {"rotate": 64}
CPUS = 8  # the settings are evaluated on the CPU
TIMEOUT = 6 * 3600

image = (
    # the same first steps as cloud/modal/app.py, so its environment is reused
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("rsync")
    .uv_sync(str(ROOT), extra_options="--no-install-package libkge")
    .env(
        {
            "PYTHONPATH": f"{CODE}/src:{CODE}/third_party/libkge:{CODE}/scripts",
            "PYTHONUNBUFFERED": "1",
            "TQDM_MININTERVAL": "60",  # one progress bar line per minute in the logs
        }
    )
    # the code and the locally prepared datasets, uploaded on every run
    .add_local_dir(ROOT / "src", f"{CODE}/src", ignore=["**/__pycache__"])
    .add_local_dir(ROOT / "scripts", f"{CODE}/scripts", ignore=["**/__pycache__"])
    .add_local_dir(
        ROOT / "third_party/libkge/kge", f"{CODE}/third_party/libkge/kge", ignore=["**/__pycache__"]
    )
)
for dataset in DATASETS:
    image = image.add_local_dir(
        ROOT / "third_party/libkge/data" / dataset,
        f"{CODE}/third_party/libkge/data/{dataset}",
        ignore=["*.pckl", "*.txt"],
    )
app = modal.App("kgcp-search", image=image)
volume = modal.Volume.from_name("kge-runs")


@app.local_entrypoint()
def main(models: str = ""):
    """Runs the comma-separated `models` (<dataset>-<model>; default: every model of
    DATASETS and MODELS) whose CSV is not on the volume yet."""
    names = models.split(",") if models else [f"{d}-{m}" for d in DATASETS for m in MODELS]
    try:
        done = {Path(entry.path).name for entry in volume.listdir(OUT)}
    except modal.exception.NotFoundError:
        done = set()
    todo = [name for name in names if f"{name}-search.csv" not in done]
    if len(todo) < len(names):
        print(f"skipped, in {OUT}/ already:", *sorted(set(names) - set(todo)))
    if todo:
        run.remote(todo)


@app.function(timeout=TIMEOUT)
def run(names: list[str]):
    """Runs the models in parallel and reports each. The command starts only this call,
    since `modal run --detach` keeps only the last call of a command running once the
    terminal disconnects."""
    calls = {name: search.spawn(name) for name in names}
    for name, call in calls.items():
        try:
            print(call.get())
        except Exception as error:  # report it, but wait for the other models
            print(f"{name} failed: {error!r}")


@app.function(volumes={VOLUME: volume}, gpu="T4", cpu=CPUS, memory=24 * 1024, timeout=TIMEOUT)
def search(name: str) -> str:
    """Runs scripts/condkgcp_search.py on a model and writes its CSV to the volume."""
    import condkgcp_search
    import conformal
    import torch

    torch.set_num_threads(CPUS)  # not the host's cores
    model = name.rsplit("-", 1)[1]
    options = [str(VOLUME / CHECKPOINTS / f"{name}.pt"), "--device", "cuda"]
    if model in BATCH_SIZES:
        options += ["--batch-size", str(BATCH_SIZES[model])]
    args = condkgcp_search.parser().parse_args(options)
    start = time.time()
    rows = condkgcp_search.search(args.models[0], args)
    out = VOLUME / OUT / f"{name}-search.csv"
    out.parent.mkdir(exist_ok=True)
    conformal.write(rows, out)
    volume.commit()
    minutes = (time.time() - start) / 60
    return f"{name}: {minutes:.0f} min on {torch.cuda.get_device_name()}, {OUT}/{out.name}"
