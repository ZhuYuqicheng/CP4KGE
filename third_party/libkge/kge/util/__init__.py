from kge.util.loss import KgeLoss
from kge.util.optimizer import KgeOptimizer
from kge.util.optimizer import KgeLRScheduler
from kge.util.sampler import KgeSampler
from kge.util.io import load_checkpoint

# Utilities with heavy optional dependencies (igraph, ConfigSpace) are imported lazily.
_LAZY_UTILS = {
    "KCoreManager": "kge.util.subgraph",
    "get_configspace": "kge.util.configspace_converter",
}


def __getattr__(name):
    if name in _LAZY_UTILS:
        import importlib

        return getattr(importlib.import_module(_LAZY_UTILS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
