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

        batch_size, max_seq_len = batch["actions"].shape

        if batch_size > 1:
            raise RuntimeError(
                """
                This temporal exception appears to be solved as soon as possible.
                I will introduce you to the cause of it. The reason why Seq2Slate
                requires 1 sample per batch is its architecture. The final (pointwise)
                decoder layer samples items from a vocabulary of variable size, so the
                paddings must be fully masked out (i.e., the mask has to be symmetric),
                meaning the probability of drawing a padding symbol is 0. A lower triangle
                mask is also applied to the vocabulary to erase already chosen items at each
                time step. The problem appears when on-policy learning is applied. In this
                method, the model re-arranges a slate in an autoregressive way, so if a slate
                is padded up to the longest in a batch, at some time step, the probability
                distribution won't be summed up to one. Even more, it always will be zero
                because all the non-padding symbols are chosen. The easiest way to solve this
                is to reduce the number of samples in a batch down to one.
                """
            )
        # TODO: remove once issue with padding is solved
        num_of_candidates = min(self.num_of_candidates, batch["actions_presence"].sum(-1).item())

        batch_dict["candidates"] = self.candidate_preprocessor(
            batch["state_sequence_features"].view(-1, self.candidate_dim),
            batch["state_sequence_features_presence"].view(-1, self.candidate_dim)
        ).view(batch_size, max_seq_len, -1)[:, :num_of_candidates, :]
        batch_dict["state"] = self.state_preprocessor(batch["state_features"], batch["state_features_presence"])

        batch_dict["actions"] = batch["actions"].long()[:, :num_of_candidates]

        if not self.on_policy:
            batch_dict["logged_propensities"] = torch.clamp(
                batch["actions_probability"].unsqueeze(1), min=1e-40
            )
            batch_dict["slate_reward"] = -batch["slate_reward"]
        else:
            batch_dict["position_reward"] = batch["item_reward"][:, :num_of_candidates]

        return adt.PreprocessedRankingInput.from_input(**batch_dict)
