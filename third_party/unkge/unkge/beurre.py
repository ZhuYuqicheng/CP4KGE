"""BEUrRE (Chen et al., 2021): probabilistic box embeddings with Gumbel boxes.

An entity is a box, its lower corner and the log of its width. A relation maps the
head's and the tail's box affinely (translation, and scaling by the ReLU of a
factor). The confidence of a triple is P(head box | tail box), the volume of their
intersection over the tail box's volume, with the Gumbel intersection and expected
volume of Dasgupta et al. (2020). The operations follow the order of the BEUrRE code
used by UnKGCP (`BiGumbelBox`), so that converted models score as the original.
"""

import torch
import torch.nn.functional as F
from kge import Config, Dataset
from kge.model.kge_model import KgeModel, RelationalScorer
from torch import Tensor

EULER_GAMMA = 0.57721566490153286060


class BEUrREScorer(RelationalScorer):
    """log P(head box | tail box). Entity embeddings hold [lower corner, log width],
    relation embeddings [head translation, head scale, tail translation, tail scale]."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)
        self.beta = self.get_option("gumbel_beta")

    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        if combine == "sp_":  # each query against all objects: [n, E]
            s_emb, p_emb, o_emb = s_emb[:, None], p_emb[:, None], o_emb[None]
        elif combine == "_po":
            s_emb, p_emb, o_emb = s_emb[None], p_emb[:, None], o_emb[:, None]
        elif combine != "spo":
            return super().score_emb(s_emb, p_emb, o_emb, combine)
        head_translation, head_scale, tail_translation, tail_scale = p_emb.chunk(4, dim=-1)
        head = _box(s_emb, head_translation, head_scale)
        tail = _box(o_emb, tail_translation, tail_scale)
        return self._log_volume(self._intersection(head, tail)) - self._log_volume(tail)

    def _intersection(self, box1, box2):
        (low1, high1, _), (low2, high2, _) = box1, box2
        low = self.beta * _logsumexp(low1 / self.beta, low2 / self.beta)
        low = torch.max(low, torch.max(low1, low2))
        high = -self.beta * _logsumexp(-high1 / self.beta, -high2 / self.beta)
        high = torch.min(high, torch.min(high1, high2))
        return low, high, high - low

    def _log_volume(self, box) -> Tensor:
        width = box[2]
        side = F.softplus(width - 2 * EULER_GAMMA * self.beta)
        return side.clamp_min(torch.finfo(width.dtype).tiny).log().sum(-1)


def _logsumexp(x: Tensor, y: Tensor) -> Tensor:
    """log(exp(x) + exp(y)), computed as BEUrRE does (torch.logaddexp differs in the last
    bits)."""
    return torch.logsumexp(torch.stack(torch.broadcast_tensors(x, y)), 0)


def _box(entity: Tensor, translation: Tensor, scale: Tensor):
    """(lower corner, upper corner, width) of the mapped box of entity embeddings."""
    low, log_width = entity.chunk(2, dim=-1)
    width = (low + log_width.exp()) - low  # as BEUrRE, which stores the upper corner
    low = low + translation
    width = width * scale.relu()
    return low, low + width, width


class BEUrRE(KgeModel):
    """BEUrRE: scores are log confidences; `confidence` is their exponential."""

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
            scorer=BEUrREScorer,
            configuration_key=configuration_key,
            init_for_load_only=init_for_load_only,
        )
        entity_dim, relation_dim = self.get_s_embedder().dim, self.get_p_embedder().dim
        if entity_dim % 2 or relation_dim != 2 * entity_dim:
            raise ValueError(
                f"beurre needs entity embeddings of size 2d and relation embeddings of "
                f"size 4d, not {entity_dim} and {relation_dim}"
            )

    def confidence(self, s: Tensor, p: Tensor, o: Tensor) -> Tensor:
        """[n] confidence of the triples, P(head box | tail box)."""
        return self.score_spo(s, p, o).exp()
