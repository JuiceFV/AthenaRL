import abc

import torch

from athena.core.parameters import NormalizationData
from athena.models.base import BaseModel


class SlateRankingNetBuilder:
    @abc.abstractmethod
    def build_slate_ranking_network(
        self, state_dim: int, candidate_dim: int, num_of_candidates: int, slate_size: int
    ) -> torch.nn.Module:
        pass

    @abc.abstractmethod
    def build_serving_module(
        self,
        network: BaseModel,
        state_normalization_data: NormalizationData,
        candidate_normalization_data: NormalizationData
    ) -> torch.nn.Module:
        pass
