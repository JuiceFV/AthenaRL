from typing import Dict

import torch
import torch.nn as nn

import athena.core.dtypes as adt
import athena.preprocessing.transforms.tensor as ttr
from athena.preprocessing.preprocessor import Preprocessor


class BatchPreprocessor(nn.Module):
    pass


def batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


class RankingBatchPreprocessor(BatchPreprocessor):
    def __init__(
        self,
        num_of_candidates: int,
        candidate_dim: int,
        state_preprocessor: Preprocessor,
        candidate_preprocessor: Preprocessor,
        use_gpu: bool = False
    ) -> None:
        super().__init__()
        self.num_of_candidates = num_of_candidates
        self.candidate_dim = candidate_dim
        self.state_preprocessor = state_preprocessor
        self.candidate_preprocessor = candidate_preprocessor
        self.device = torch.device("cuda") if use_gpu else torch.device("cpu")
        self.flatten_view = ttr.FlattenSlateView(
            [
                "state_sequence_features",
                "state_sequence_features_presence"
            ],
            candidate_dim=self.candidate_dim
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> adt.PreprocessedRankingInput:
        batch = batch_to_device(batch, self.device)
        preprocessed_state = self.state_preprocessor(
            batch["state_features"], batch["state_features_presence"]
        )

        batch_size = list(batch.values())[0].shape[0]
        max_seq_len = batch["state_sequence_features"].shape[1] // self.candidate_dim
        batch = self.flatten_view(batch)
        preprocessed_candidate = self.candidate_preprocessor(
            batch["state_sequence_features"],
            batch["state_sequence_features_presence"]
        ).view(batch_size, max_seq_len, -1)[:, :self.num_of_candidates, :]

        actions = batch["actions"].long()[:, :self.num_of_candidates]

        return adt.PreprocessedRankingInput.from_input(
            state=preprocessed_state,
            candidates=preprocessed_candidate,
            actions=actions,
            device=self.device,
        )


class Seq2SlateBatchPreprocessor(RankingBatchPreprocessor):
    def __init__(
        self,
        num_of_candidates: int,
        candidate_dim: int,
        state_preprocessor: Preprocessor,
        candidate_preprocessor: Preprocessor,
        use_gpu: bool = False,
        on_policy: bool = True
    ) -> None:
        super().__init__(
            num_of_candidates,
            candidate_dim,
            state_preprocessor,
            candidate_preprocessor,
            use_gpu
        )
        self.on_policy = on_policy

    def forward(self, batch: Dict[str, torch.Tensor]) -> adt.PreprocessedRankingInput:
        base_ranking_input = super().forward(batch)
        batch_dict = {
            "state": base_ranking_input.state.dense_features,
            "candidates": base_ranking_input.source_seq.dense_features,
            "actions": base_ranking_input.target_output_indcs - 2,
            "device": base_ranking_input.state.dense_features.device
        }
        if not self.on_policy:
            batch_dict["logged_propensities"] = batch["actions_probability"].unsqueeze(1)
            batch_dict["slate_reward"] = -batch["slate_reward"]
        else:
            batch_dict["position_reward"] = batch["item_reward"][:, :self.num_of_candidates]

        return adt.PreprocessedRankingInput.from_input(**batch_dict)
