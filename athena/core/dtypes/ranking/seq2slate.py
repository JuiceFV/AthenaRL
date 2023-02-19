from enum import Enum
from typing import NamedTuple, Optional

import torch

from athena.core.enum_meta import AthenaEnumMeta


class Seq2SlateMode(Enum, metaclass=AthenaEnumMeta):
    r"""
    The mode in which :class:`~athena.models.ranking.seq2slate.Seq2SlateTransformerModel`
    performs.
    """
    #: Returns ranked items and their generative probabilities.
    RANK_MODE = "rank"
    #: Returns generative log probabilities of the given target sequence (used for REINFORCE training).
    PER_SEQ_LOG_PROB_MODE = "per_sequence_log_prob"
    #: Returns generative log probabilties of each item in the given sequences (used in TEACHER FORCING training).
    PER_ITEM_LOG_PROB_DIST_MODE = "per_item_log_prob_dist"
    #: One-shot decoding that uses encoder compact representation.
    DECODE_ONE_STEP_MODE = "decode_one_step"
    #: Produces only encoder scores.
    ENCODER_SCORE_MODE = "encoder_score_mode"


class Seq2SlateOutputArch(Enum, metaclass=AthenaEnumMeta):
    r"""
    The variation of Seq2Slate model.
    """
    #: Only output encoder scores.
    ENCODER_SCORE = "encoder_score"

    #: A decoder outputs a sequence in an autoregressive way.
    AUTOREGRESSIVE = "autoregressive"

    #: Using encoder scores, a decoder outputs a sequence using frechet sort.
    FRECHET_SORT = "frechet_sort"


class Seq2SlateTransformerOutput(NamedTuple):
    r"""
    Every possible output of :class:`~athena.models.ranking.seq2slate.Seq2SlateTransformerModel`
    """
    #: Item's probabilities bieng placed. Item probability is :math:`p(\pi_i | \pi_{<i})`
    ordered_per_item_probas: Optional[torch.Tensor]
    #: Generative probabilities of each permutation.
    ordered_per_seq_probas: Optional[torch.Tensor]
    #: Rearranged sequence.
    ordered_target_out_indcs: Optional[torch.Tensor]
    #: Log probabilities of item to be in place.
    per_item_log_probas: Optional[torch.Tensor]
    #: Log of probability of sequence.
    per_seq_log_probas: Optional[torch.Tensor]
    #: Encoder weights that are used in encoder only method.
    encoder_scores: Optional[torch.Tensor]


class Seq2SlateVersion(Enum, metaclass=AthenaEnumMeta):
    r"""
    Version of the :class:`~athena.models.ranking.seq2slate.Seq2SlateTransformerModel`.
    """
    #: Weak supervised learning, that implies ground truth labels (e.g. number of clicks).
    TEACHER_FORCING = "teacher_forcing"

    #: Optimizes a given metric :math:`\mathcal{R}(\pi, y)` in the RL fashion.
    REINFORCEMENT_LEARNING = "reinforcement_learning"

    #: Pairwise attention model.
    PAIRWISE_ATTENTION = "pairwise_attention"
