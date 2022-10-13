from dataclasses import dataclass, fields

import numpy as np
import torch
import torch.nn as nn

import athena.core.dtypes as adt
from athena.models import Seq2SlateTransformerNetwork


@dataclass
class EvaluationOnBatch(adt.TensorDataClass):
    model_propensities: torch.Tensor
    model_rewards: torch.Tensor
    actions_mask: torch.Tensor
    logged_rewards: torch.Tensor
    logged_action_rewards: torch.Tensor
    logged_propensities: torch.Tensor

    @classmethod
    @torch.no_grad()
    def from_seq2slate(
        cls,
        seq2slate_network: Seq2SlateTransformerNetwork,
        propensity_network: nn.Module,
        training_input: adt.PreprocessedRankingInput,
        greedy_eval: bool
    ):
        if (
            training_input.slate_reward is None
            or training_input.target_output_probas is None
            or training_input.target_output_indcs is None
            or training_input.target_output_seq is None
        ):
            raise ValueError(f"Expected that {seq2slate_network} is trained.")

        batch_size, target_seq_len, candidate_dim = training_input.target_output_seq.dense_features.shape
        device = training_input.state.dense_features.device

        rank_output: adt.RankingOutput = seq2slate_network(training_input, adt.Seq2SlateMode.RANK_MODE, greedy=True)

        if rank_output.ordered_target_out_indcs is None:
            raise ValueError("Missed result slate. ")

        if greedy_eval:
            model_propensities = torch.ones(batch_size, 1, device=device)
            actions_mask = torch.all(
                (training_input.target_output_indcs - 2) == (rank_output.ordered_target_out_indcs - 2),
                dim=1, keepdim=True
            ).float()
        else:
            model_propensities = torch.exp(
                seq2slate_network(training_input, adt.Seq2SlateMode.PER_SEQ_LOG_PROB_MODE).log_probas
            ).float()
            actions_mask = torch.ones(batch_size, 1, device=device).float()

        logged_action_rewards = propensity_network(
            training_input.state.dense_features,
            training_input.source_seq.dense_features,
            training_input.target_output_seq.dense_features,
            training_input.source2source_mask,
            training_input.target_output_indcs,
        ).reshape(-1, 1)

        seq_arangement = torch.arange(batch_size, device=device).repeat_interleave(target_seq_len)
        items_arangement = rank_output.ordered_target_out_indcs.flatten() - 2
        ordered_target_output_seq = training_input.source_seq.dense_features[seq_arangement, items_arangement]
        ordered_target_output_seq = ordered_target_output_seq.reshape(batch_size, target_seq_len, candidate_dim)

        model_rewards = propensity_network(
            training_input.state.dense_features,
            training_input.source_seq.dense_features,
            ordered_target_output_seq,
            training_input.source2source_mask,
            rank_output.ordered_target_out_indcs
        ).reshape(-1, 1)

        logged_rewards = training_input.slate_reward.reshape(-1, 1)
        logged_propensities = training_input.target_output_probas.reshape(-1, 1)
        return cls(
            model_propensities=model_propensities,
            model_rewards=model_rewards,
            actions_mask=actions_mask,
            logged_rewards=logged_rewards,
            logged_action_rewards=logged_action_rewards,
            logged_propensities=logged_propensities
        )

    def append(self, eob: "EvaluationOnBatch") -> "EvaluationOnBatch":
        new_eob = {}

        for field in fields(EvaluationOnBatch):
            original_tensor = getattr(self, field.name)
            addable_tensor = getattr(eob, field.name)
            if int(original_tensor is None) + int(addable_tensor is None) == 1:
                raise AttributeError(
                    f"Tried to append when a tensor existed in one training page but not the other: {field.name}"
                )
            if addable_tensor is not None:
                if isinstance(original_tensor, torch.Tensor):
                    new_eob[field.name] = torch.cat((original_tensor, addable_tensor), dim=0)
                elif isinstance(original_tensor, np.ndarray):
                    new_eob[field.name] = np.concatenate((original_tensor, addable_tensor), axis=0)
                else:
                    raise TypeError("Invalid type in batch data.")
            else:
                new_eob[field.name] = None
        return EvaluationOnBatch(**new_eob)
