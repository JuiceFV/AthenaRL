import torch

import athena.metrics.functional as MF
from athena.core.dataclasses import dataclass
from athena.metrics.base import MetricBase


@dataclass
class NDCG(MetricBase):
    topk: int

    def __post_init_post_parse__(self):
        if self.topk < 1:
            raise ValueError("Top k is less than one, specify appropriate value.")

    def compute(self, y_true: torch.Tensor, y_score: torch.Tensor) -> torch.Tensor:
        return MF.ndcg(y_true, y_score, self.topk)
