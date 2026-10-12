from kge.job.trace import Trace
from kge.job.job import Job
from kge.job.job import TrainingOrEvaluationJob
from kge.job.train import TrainingJob
from kge.job.train_1vsAll import TrainingJob1vsAll
from kge.job.train_KvsAll import TrainingJobKvsAll
from kge.job.train_negative_sampling import TrainingJobNegativeSampling
from kge.job.eval import EvaluationJob
from kge.job.eval_training_loss import TrainingLossEvaluationJob
from kge.job.eval_entity_ranking import EntityRankingJob
from kge.job.eval_entity_pair_ranking import EntityPairRankingJob
from kge.job.search import SearchJob
from kge.job.search_grid import GridSearchJob
from kge.job.search_manual import ManualSearchJob
from kge.job.search_auto import AutoSearchJob

# Search jobs with heavy optional dependencies (`pip install libkge[search]`) are
# imported lazily on first access, so that training works without them.
_LAZY_JOBS = {
    "AxSearchJob": "kge.job.search_ax",
    "GraSHSearchJob": "kge.job.search_grash",
}


def __getattr__(name):
    if name in _LAZY_JOBS:
        import importlib

        return getattr(importlib.import_module(_LAZY_JOBS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
