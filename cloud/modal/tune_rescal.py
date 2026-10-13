"""scripts/tune.py for RESCAL: the same protocol with embedding sizes 32, 64 and 128.

RESCAL learns a d x d matrix per relation, so its cost grows with d squared. On Nations,
the trials of sizes 256 and 512 took 70% of the compute without reaching the best
validation MRR (32 was best), and the original RESCAL paper (Nickel et al., 2011) used
ranks 10 to 40 on Nations, UMLS and Kinship. Takes the arguments of scripts/tune.py:

    python cloud/modal/tune_rescal.py sobol --datasets umls kinship --models rescal
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import tune  # noqa: E402

# set on import, so that worker processes that import this script (spawn) get it too
tune.DIMS = [32, 64, 128]

if __name__ == "__main__":
    # the command line of scripts/tune.py, which defines it in its __main__ block
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["sobol", "refine", "final", "report", "time"])
    parser.add_argument("--datasets", nargs="+", default=["nations", "umls", "kinship"])
    parser.add_argument("--models", nargs="+", default=["rescal"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="where models train: cpu, cuda, mps")
    parser.add_argument("--dir", type=Path, default=tune.RUNS, help="where trials are stored")
    parser.add_argument("--trials", type=int, help="trials per stage (default: 30)")
    parser.add_argument("--max-epochs", type=int, help="epochs per training (default: 400)")
    args = parser.parse_args()
    tune.RUNS = args.dir
    tune.RUNTIME["job.device"] = args.device
    if args.max_epochs:
        tune.RUNTIME["train.max_epochs"] = args.max_epochs
    tune.TRIALS = {stage: args.trials or n for stage, n in tune.TRIALS.items()}
    tune.RUNS.mkdir(parents=True, exist_ok=True)
    getattr(tune, f"cmd_{args.command}")(args)
