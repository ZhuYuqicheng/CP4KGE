"""UKGE (Chen et al., 2019): DistMult with a logistic or bounded-rectifier output.

PASSLEAF's models (Chen et al., 2021) of the UnKGCP paper, UKGE_logi_m2, differ only
in how they were trained and are UKGE models with the logistic output.
"""

import torch
from kge import Config, Dataset
from kge.model.distmult import DistMultScorer
from kge.model.kge_model import KgeModel
from torch import Tensor


class UKGEScorer(DistMultScorer):
    """The DistMult score mapped affinely, w <s, p, o> + b: the logit of the confidence
    (logi) or the unbounded confidence (rect)."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        return self.weight * super().score_emb(s_emb, p_emb, o_emb, combine) + self.bias


class UKGE(KgeModel):
    """UKGE: scores are logits (`ukge.output` logi) or unbounded confidences (rect), so
    higher is more plausible; `confidence` maps them to [0, 1]."""

    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key=None,
        init_for_load_only=False,
    ):
        super().__init__(
            config=config,
            dataset=dataset,
            scorer=UKGEScorer,
            configuration_key=configuration_key,
            init_for_load_only=init_for_load_only,
        )
        self.output = self.get_option("output")
        if self.output not in ("logi", "rect"):
            raise ValueError(f"ukge.output must be logi or rect, not {self.output}")

    def confidence(self, s: Tensor, p: Tensor, o: Tensor) -> Tensor:
        """[n] confidence of the triples: the sigmoid of the score (logi), or the score
        clamped to [0, 1] (rect, UKGE's bounded score)."""
        score = self.score_spo(s, p, o)
        return score.sigmoid() if self.output == "logi" else score.clamp(0, 1)
