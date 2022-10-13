import torch
from typing_extensions import overload

from athena.core.registry import DiscriminatedUnion
from athena.metrics.base import MetricBase
from athena.metrics.hit import Hit
from athena.metrics.mrr import MRR
from athena.metrics.ndcg import NDCG


@MetricBase.register()
class MetricRoster(DiscriminatedUnion):
    @classmethod
    def default_mrr(cls) -> "MetricRoster":
        return cls(MRR=MRR(topk=50))

    @classmethod
    def default_hit(cls) -> "MetricRoster":
        return cls(Hit=Hit(topk=50, hit_if_empty=False))

    @classmethod
    def default_ndcg(cls) -> "MetricRoster":
        return cls(NDCG=NDCG(topk=50))

    @overload
    def __call__(self, y_true: torch.Tensor, y_score: torch.Tensor) -> torch.Tensor:
        ...

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.value(*args, **kwargs)
