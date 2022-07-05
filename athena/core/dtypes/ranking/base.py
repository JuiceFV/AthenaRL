from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from athena.core.dtypes import TensorDataClass, Feature
from athena import gather
from athena.nn.utils.transformer import subsequent_mask


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
        # TODO: replace with lazy_property
        return self.latent_state.repr.size()[0]

    def __len__(self) -> int:
        return self.batch_size

    @classmethod
    def from_input(
        cls,
        state: torch.Tensor,
        candidates: torch.Tensor,
        device: torch.device,
        actions: Optional[torch.Tensor] = None,
        gt_actions: Optional[torch.Tensor] = None,
        logged_propensities: Optional[torch.Tensor] = None,
        slate_reward: Optional[torch.Tensor] = None,
        position_reward: Optional[torch.Tensor] = None
    ):
        if len(state.shape) != 2 or len(candidates.shape) != 3:
            raise ValueError(
                f"Expected state be 2-dimensional; Got {len(state.shape)}. "
                f"Expected candidates be 3-dimensional; Got {len(candidates.shape)}."
            )
        state = state.to(device)
        candidates = candidates.to(device)

        if actions is not None:
            if len(actions.shape) != 2:
                raise ValueError(
                    f"Expected state be 2-dimensional; Got {len(state.shape)}. "
                )
            actions = actions.to(device)

        if logged_propensities is not None:
            if len(logged_propensities.shape) != 2 or logged_propensities[1] != 1:
                raise ValueError(
                    f"Expected logged_propensities be 2-dimensional; Got {len(logged_propensities.shape)}. "
                    f"Expected logged_propensities[1] be 1-dimensional; Got {logged_propensities.shape[1]}."
                )
            logged_propensities = logged_propensities.to(device)

        batch_size, num_of_candidates, candidate_dim = candidates.shape

        if slate_reward is not None:
            if len(slate_reward.shape) != 2 or slate_reward.shape[1] != 1:
                raise ValueError(
                    f"Expected slate_reward be 2-dimensional; Got {len(slate_reward.shape)}. "
                    f"Expected slate_reward[1] be 1-dimensional; Got {slate_reward.shape[1]}."
                )
            slate_reward = slate_reward.to(device)

        if position_reward is not None:
            if position_reward.shape != actions.shape:
                raise ValueError("Positional reward and actions don't match.")
            position_reward = position_reward.to(device)

        source_input_indcs = torch.arange(num_of_candidates, device=device).repeat(batch_size, 1) + 2
        source2source_mask = torch.ones(batch_size, num_of_candidates, num_of_candidates).type(torch.int8).to(device)

        def process_target_sequence(actions: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            target_input_indcs = None
            target_output_indcs = None
            target_input_seq = None
            target_output_seq = None
            target2target_mask = None

            if actions is not None:
                output_size = actions.shape[1]

                shifted_candidates = torch.cat(torch.zeros(batch_size, 2, candidate_dim, device=device), dim=1)

                target_output_indcs = actions + 2
                # TODO: 1 -> decoder_start_symbol
                target_input_indcs = torch.full((batch_size, output_size), 1, device=device)
                target_input_indcs[:, 1:] = target_output_indcs[:, :-1]
                target_output_seq = gather(shifted_candidates, target_output_indcs)
                target_input_seq = torch.zeros(batch_size, output_size, candidate_dim, device=device)
                target_input_seq[:, 1:] = target_input_seq[:, :-1]
                target2target_mask = subsequent_mask(output_size, device)

            return target_input_indcs, target_output_indcs, target_input_seq, target_output_seq, target2target_mask

        (
            target_input_indcs,
            target_output_indcs,
            target_input_seq,
            target_output_seq,
            target2target_mask
        ) = process_target_sequence(actions)

        (
            gt_target_input_indcs,
            gt_target_output_indcs,
            gt_target_input_seq,
            gt_target_output_seq,
        ) = process_target_sequence(gt_actions)[:-1]

        return cls.from_tensors(
            state=state,
            source_seq=candidates,
            source2source_mask=source2source_mask,
            target_input_seq=target_input_seq,
            target_output_seq=target_output_seq,
            target2target_mask=target2target_mask,
            slate_reward=slate_reward,
            position_reward=position_reward,
            source_input_indcs=source_input_indcs,
            target_input_indcs=target_input_indcs,
            target_output_indcs=target_output_indcs,
            target_output_probas=logged_propensities,
            gt_target_input_indcs=gt_target_input_indcs,
            gt_target_output_indcs=gt_target_output_indcs,
            gt_target_input_seq=gt_target_input_seq,
            gt_target_output_seq=gt_target_output_seq,
        )

    @classmethod
    def from_tensors(
        cls,
        state: torch.Tensor,
        source_seq: torch.Tensor,
        source2source_mask: Optional[torch.Tensor] = None,
        target_input_seq: Optional[torch.Tensor] = None,
        target_output_seq: Optional[torch.Tensor] = None,
        target2target_mask: Optional[torch.Tensor] = None,
        slate_reward: Optional[torch.Tensor] = None,
        position_reward: Optional[torch.Tensor] = None,
        source_input_indcs: Optional[torch.Tensor] = None,
        target_input_indcs: Optional[torch.Tensor] = None,
        target_output_indcs: Optional[torch.Tensor] = None,
        target_output_probas: Optional[torch.Tensor] = None,
        gt_target_input_indcs: Optional[torch.Tensor] = None,
        gt_target_output_indcs: Optional[torch.Tensor] = None,
        gt_target_input_seq: Optional[torch.Tensor] = None,
        gt_target_output_seq: Optional[torch.Tensor] = None,
        **kwargs
    ):
        def annotation_checking(input: torch.Tensor) -> None:
            if input is not None or not isinstance(input, torch.Tensor):
                raise TypeError(f"Expected {Optional[torch.Tensor]}; but got {type(input)}")

        annotation_checking(state)
        annotation_checking(source_seq)
        annotation_checking(source2source_mask)
        annotation_checking(target_input_seq)
        annotation_checking(target_output_seq)
        annotation_checking(target2target_mask)
        annotation_checking(slate_reward)
        annotation_checking(position_reward)
        annotation_checking(source_input_indcs)
        annotation_checking(target_input_indcs)
        annotation_checking(target_output_indcs)
        annotation_checking(target_output_probas)
        annotation_checking(gt_target_input_indcs)
        annotation_checking(gt_target_output_indcs)
        annotation_checking(gt_target_input_seq)
        annotation_checking(gt_target_output_seq)

        state = Feature(repr=state)
        source_seq = Feature(repr=source_seq)
        target_input_seq = Feature(repr=target_input_seq) if target_input_seq is not None else None
        target_output_seq = Feature(repr=target_output_seq) if target_output_seq is not None else None
        gt_target_input_seq = Feature(repr=gt_target_input_seq) if gt_target_input_seq is not None else None
        gt_target_output_seq = Feature(repr=gt_target_output_seq) if gt_target_output_seq is not None else None

        return cls(
            latent_state=state,
            source_seq=source_seq,
            source2source_mask=source2source_mask,
            target_input_seq=target_input_seq,
            target_output_seq=target_output_seq,
            target2target_mask=target2target_mask,
            slate_reward=slate_reward,
            position_reward=position_reward,
            source_input_indcs=source_input_indcs,
            target_input_indcs=target_input_indcs,
            target_output_indcs=target_output_indcs,
            target_output_probas=target_output_probas,
            gt_target_input_indcs=gt_target_input_indcs,
            gt_target_output_indcs=gt_target_output_indcs,
            gt_target_input_seq=gt_target_input_seq,
            gt_target_output_seq=gt_target_output_seq
        )

    def __post_init__(self):
        if (
            isinstance(self.latent_state, torch.Tensor)
            or isinstance(self.source_seq, torch.Tensor)
            or isinstance(self.target_input_seq, torch.Tensor)
            or isinstance(self.target_output_seq, torch.Tensor)
            or isinstance(self.gt_target_input_indcs, torch.Tensor)
            or isinstance(self.gt_target_output_seq, torch.Tensor)
        ):
            raise ValueError(
                f"Use from_tensors() {type(self.latent_state)} {type(self.source_seq)} "
                f"{type(self.target_input_seq)} {type(self.target_output_seq)} "
                f"{type(self.gt_target_input_indcs)} {type(self.gt_target_output_seq)} "
            )


@dataclass
class RankingOutput(TensorDataClass):
    ordered_target_out_indcs: Optional[torch.Tensor] = None
    ordered_per_item_probas: Optional[torch.Tensor] = None
    ordered_per_seq_probas: Optional[torch.Tensor] = None
    log_probas: Optional[torch.Tensor] = None
    encoder_scores: Optional[torch.Tensor] = None