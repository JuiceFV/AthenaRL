from dataclasses import dataclass
from typing import Optional

import torch
from athena.core.dtypes import TensorDataClass, Feature


@dataclass
class DocSeq(TensorDataClass):
    # Documents feature-wise represented
    # torch.Size([batch_size, num_candidates, num_doc_features])
    repr: torch.Tensor
    mask: Optional[torch.Tensor] = None
    gain: Optional[torch.Tensor] = None

    def __post_init__(self):
        if len(self.repr.shape) != 3:
            raise ValueError(
                f"Unexpected shape: {self.repr.shape}"
            )
        if self.mask is None:
            self.mask = self.repr.new_ones(
                self.repr.shape[:2], dtype=torch.bool
            )
        if self.gain is None:
            self.gain = self.repr.new_ones(
                self.repr.shape[:2]
            )

@dataclass
class PreprocessedRankingInput(TensorDataClass):
    # latent memory state, i.e. the transduction
    # {x_i}^n -> {e_i}^n, where e_i = Encoder(x_i).
    # In ranking seq2slate problem latent state
    # depicts the items placed at 0 ... t-1 timestamp.
    latent_state: Feature

    # Sequence feature-wise representation {x_i}^n.
    source_seq: Feature

    # Mask for source sequences' items, especially
    # required for the reward (i.e. RL mode) models.
    # It defines the items set to which an item
    # should pay attention to in purpose to increase
    # reward metrics.
    source2source_mask: Optional[torch.Tensor] = None

    # Target sequence passed to the decoder.
    # Used in weak supervised and teacher forcing learning.
    target_input_seq: Optional[Feature] = None

    # Target sequence after passing throughout the decoder
    # and stacked fully-connected layers.
    target_output_seq: Optional[Feature] = None

    # Mask for target sequences' items, s.t.
    # each item of a sequence has its own set of
    # items to pay attention to.
    target2target_mask: Optional[torch.Tensor] = None
    slate_reward: Optional[torch.Tensor] = None
    position_reward: Optional[torch.Tensor] = None

    # all indices will be shifted onto 2 due to padding items
    # start/end symbol.
    source_input_indcs: Optional[torch.Tensor] = None
    target_input_indcs: Optional[torch.Tensor] = None
    target_output_indcs: Optional[torch.Tensor] = None
    target_output_probas: Optional[torch.Tensor] = None

    # Ground-truth target sequence (for teacher forcing)
    gt_target_input_indcs: Optional[torch.Tensor] = None
    gt_target_output_indcs: Optional[torch.Tensor] = None
    gt_target_input_seq: Optional[Feature] = None
    gt_target_output_seq: Optional[Feature] = None

    @property
    def batch_size(self) -> int:
        return self.latent_state.repr.size()[0]

    def __len__(self) -> int:
        return self.batch_size


@dataclass
class RankingOutput(TensorDataClass):
    ordered_target_out_idcs: Optional[torch.Tensor] = None
    ordered_per_item_probas: Optional[torch.Tensor] = None
    ordered_per_seq_probas: Optional[torch.Tensor] = None
    log_probas: Optional[torch.Tensor] = None
    encoder_scores: Optional[torch.Tensor] = None