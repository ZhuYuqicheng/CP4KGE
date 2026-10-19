"""LibKGE models for uncertain knowledge graphs, to load the models of UnKGCP (Zhu et
al., 2025): UKGE (and PASSLEAF's UKGE models) and BEUrRE.

LibKGE finds them when a config lists this package in `modules`; the converted
checkpoints do.
"""

from unkge.beurre import BEUrRE
from unkge.ukge import UKGE

MODULES = ["kge.job", "kge.model", "kge.model.embedder", "unkge"]

__all__ = ["BEUrRE", "MODULES", "UKGE"]
