from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch

from athena import gather
from athena.core.dtypes import ExtraData, Feature, TensorDataClass
from athena.nn.utils.transformer import (DECODER_START_SYMBOL, PADDING_SYMBOL,
                                         encoder_mask,
                                         subsequent_and_padding_mask)


@dataclass
class PreprocessedRankingInput(TensorDataClass):
    """The data format dedicated as input to a ranking model. Tentatiely, the data must be preprocessed.

    .. note::

        Due to ranking algorithms are so diverse there are only two mandatory fields, while others are Optional.

        1. State representation. A state is typically used in RL to represent action-independent observable space
           relative to the agent. There is no such space regarding the ranking problem, but we can reformulate the
           problem as a user-specific problem by adding a user vector.

        2. Featurewise sequence. Obviously, we need an input sequence that should be ranked or re-ranked.

    .. warning::

        Note that in the ``target_*`` indices are shifted by two, due to the padding and start symbol.
    """

    #: Action-independent state of an environment.
    state: Feature

    #: Sequence featurewise representation.
    source_seq: Feature

    #: Mask for source sequences' items.
    source2source_mask: Optional[torch.Tensor] = None

    #: Featurewise target sequence passed to a decoder.
    target_input_seq: Optional[Feature] = None

    #: Featurewise target sequence that is sampled by a decoder.
    target_output_seq: Optional[Feature] = None

    #: Mask for target sequences' items.
    target2target_mask: Optional[torch.Tensor] = None

    #: Reward calculated for a permutation.
    slate_reward: Optional[torch.Tensor] = None

    #: Reward calculated for an item being in a given position.
    position_reward: Optional[torch.Tensor] = None

    #: Source sequence arrangement indices.
    source_input_indcs: Optional[torch.Tensor] = None

    #: Target sequence indices passed to a decoder.
    target_input_indcs: Optional[torch.Tensor] = None

    #: Re-aranged target sequence once a decoder proceeds.
    target_output_indcs: Optional[torch.Tensor] = None

    #: The probabilities of each item in the output sequence to be placed.
    target_output_probas: Optional[torch.Tensor] = None

    #: Ground-truth target input indices of sequence.
    gt_target_input_indcs: Optional[torch.Tensor] = None

    #: Ground-truth target output indices of sequence.
    gt_target_output_indcs: Optional[torch.Tensor] = None

    #: Ground-truth target input sequence representaation.
    gt_target_input_seq: Optional[Feature] = None

    #: Ground-truth target input sequence representaation.
    gt_target_output_seq: Optional[Feature] = None

    #: Extra infromation is more data understending tool.
    extras: Optional[ExtraData] = field(default_factory=ExtraData)

    def batch_size(self) -> int:
        return self.state.dense_features.size()[0]

    def __len__(self) -> int:
        return self.batch_size()

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
        position_reward: Optional[torch.Tensor] = None,
        extras: Optional[ExtraData] = None
    ):
        """Transform the preprocessed data from raw input, s.t. it may be used in the ranking problem.

        Args:
            state (torch.Tensor): Additional action independent representation. For example, user vector.
            candidates (torch.Tensor): Candidates for the next item to choose.
            device (torch.device): Device where computations occur.
            actions (Optional[torch.Tensor], optional): Target arangment "actions". Defaults to None.
            gt_actions (Optional[torch.Tensor], optional): Ground truth actions. Defaults to None.
            logged_propensities (Optional[torch.Tensor], optional): Propensities predicted by base model.
            slate_reward (Optional[torch.Tensor], optional): Total reward calculated for a permutation.
            position_reward (Optional[torch.Tensor], optional): Item-at-position reward. Defaults to None.
            extras: (Optional[ExtraData], optional): Additional batch information. Defaults to None.

        Raises:
            ValueError: Wrong dimensionality of either ``state`` or ``candidates``.
            ValueError: Wrong dimensionality of ``actions``.
            ValueError: Wrong dimensionality of ``logged_propensities``.
            ValueError: Wrong dimensionality of ``slate_reward``.
            ValueError: If ``position_reward`` and ``actions`` dimensionalities don't match.

        Shape:
            - state: :math:`(B, E)`
            - candidates: :math:`(B, N, C)`
            - actions: :math:`(B, S)`
            - actions_mask: :math:`(B, S)`
            - gt_actions: :math:`(B, S)`
            - gt_actions_mask: :math:`(B, S)`
            - logged_propensities: :math:`(B, 1)`
            - slate_reward: :math:`(B, 1)`
            - position_reward: :math:`(B, S)`

        Notations:
            - :math:`B` - batch size.
            - :math:`E` - state vector dimensionality.
            - :math:`N` - number of candidates for a position.
            - :math:`C` - a candidate dimensionality.
            - :math:`S` - source sequence length.


        Returns:
            PreprocessedRankingInput: Input processed s.t. it could be used in the ranking models.
        """
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
                    f"Expected state be 2-dimensional; Got {len(actions.shape)}. "
                )
            actions = actions.to(device)

        if logged_propensities is not None:
            if len(logged_propensities.shape) != 2 or logged_propensities.shape[1] != 1:
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
                raise ValueError("Positional reward and actions shapes don't match.")
            position_reward = position_reward.to(device)

        # Shift original sequence in purpose to take starting/padding into account
        source_input_indcs = torch.arange(num_of_candidates, device=device).repeat(batch_size, 1) + 2
        # Mask out padding symbols for an encoder layer
        source2source_mask = encoder_mask(source_input_indcs, 1, PADDING_SYMBOL)

        def process_target_sequence(
            actions: Optional[torch.Tensor]
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            target_input_indcs = None
            target_output_indcs = None
            target_input_seq = None
            target_output_seq = None
            target2target_mask = None

            if actions is not None:
                output_size = actions.shape[1]

                shifted_candidates = torch.cat(
                    (torch.zeros(batch_size, 2, candidate_dim, device=device), candidates), dim=1
                )

                # Shift decoder input/output indices for 2 positions to incorporate start/padding symbols
                target_output_indcs = actions + 2
                target_input_indcs = torch.full((batch_size, output_size), DECODER_START_SYMBOL, device=device)
                # Input sequence starts with DECODER_START_SYMBOL
                target_input_indcs[:, 1:] = target_output_indcs[:, :-1]
                # Collect featurewise sequences
                target_output_seq = gather(shifted_candidates, target_output_indcs)
                target_input_seq = torch.zeros(batch_size, output_size, candidate_dim, device=device)
                target_input_seq[:, 1:] = target_output_seq[:, :-1]
                target2target_mask = subsequent_and_padding_mask(target_input_indcs, PADDING_SYMBOL)

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
            extras=extras
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
        extras: Optional[ExtraData] = None,
        **kwargs
    ):
        def annotation_checking(input: torch.Tensor) -> None:
            if input is not None and not isinstance(input, torch.Tensor):
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

        state = Feature(dense_features=state)
        source_seq = Feature(dense_features=source_seq)
        target_input_seq = Feature(dense_features=target_input_seq) if target_input_seq is not None else None
        target_output_seq = Feature(dense_features=target_output_seq) if target_output_seq is not None else None
        gt_target_input_seq = Feature(dense_features=gt_target_input_seq) if gt_target_input_seq is not None else None
        gt_target_output_seq = Feature(
            dense_features=gt_target_output_seq) if gt_target_output_seq is not None else None

        return cls(
            state=state,
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
            gt_target_output_seq=gt_target_output_seq,
            extras=extras
        )

    def __post_init__(self):
        if (
            isinstance(self.state, torch.Tensor)
            or isinstance(self.source_seq, torch.Tensor)
            or isinstance(self.target_input_seq, torch.Tensor)
            or isinstance(self.target_output_seq, torch.Tensor)
            or isinstance(self.gt_target_input_indcs, torch.Tensor)
            or isinstance(self.gt_target_output_seq, torch.Tensor)
        ):
            raise ValueError(
                f"Use from_tensors() {type(self.state)} {type(self.source_seq)} "
                f"{type(self.target_input_seq)} {type(self.target_output_seq)} "
                f"{type(self.gt_target_input_indcs)} {type(self.gt_target_output_seq)} "
            )


@dataclass
class RankingOutput(TensorDataClass):
    r"""
    Generalized output of any ranking models.
    """
    ordered_target_out_indcs: Optional[torch.Tensor] = None
    ordered_per_item_probas: Optional[torch.Tensor] = None
    ordered_per_seq_probas: Optional[torch.Tensor] = None
    log_probas: Optional[torch.Tensor] = None
    encoder_scores: Optional[torch.Tensor] = None
