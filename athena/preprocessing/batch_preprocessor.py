from typing import Dict

import torch
import torch.nn as nn

import athena.core.dtypes as adt
from athena.preprocessing.preprocessor import Preprocessor


class BatchPreprocessor(nn.Module):
    pass


def batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


class Seq2SlateBatchPreprocessor(BatchPreprocessor):
    def __init__(
        self,
        num_of_candidates: int,
        candidate_dim: int,
        state_preprocessor: Preprocessor,
        candidate_preprocessor: Preprocessor,
        on_policy: bool = True,
        use_gpu: bool = False
    ) -> None:
        super().__init__()
        self.num_of_candidates = num_of_candidates
        self.candidate_dim = candidate_dim
        self.state_preprocessor = state_preprocessor
        self.candidate_preprocessor = candidate_preprocessor
        self.device = torch.device("cuda") if use_gpu else torch.device("cpu")
        self.on_policy = on_policy

    def forward(self, batch: Dict[str, torch.Tensor]) -> adt.PreprocessedRankingInput:
        batch = batch_to_device(batch, self.device)
        batch_dict = {"device": self.device}
        batch_dict["state"] = self.state_preprocessor(batch["state_features"], batch["state_features_presence"])

        batch_size, max_seq_len = batch["actions"].shape
        batch_dict["candidates"] = self.candidate_preprocessor(
            batch["state_sequence_features"].view(-1, self.candidate_dim),
            batch["state_sequence_features_presence"].view(-1, self.candidate_dim)
        ).view(batch_size, max_seq_len, -1)[:, :self.num_of_candidates, :]

        batch_dict["actions"] = batch["actions"].long()[:, :self.num_of_candidates]

        if not self.on_policy:
            batch_dict["logged_propensities"] = batch["actions_probability"].unsqueeze(1)
            batch_dict["slate_reward"] = -batch["slate_reward"]
        else:
            batch_dict["position_reward"] = batch["item_reward"][:, :self.num_of_candidates]

        return adt.PreprocessedRankingInput.from_input(**batch_dict)
