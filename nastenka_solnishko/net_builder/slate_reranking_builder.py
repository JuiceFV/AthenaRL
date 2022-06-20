import abc
import torch

class SlateRerankingNetBuilder:
    @abc.abstractmethod
    def build_slate_reranking_network(
        self, 
        latent_state_dim: int, 
        candidate_dim: int,
        num_of_candidates: int,
        slate_size: int 
    ) -> torch.nn.Module:
        pass