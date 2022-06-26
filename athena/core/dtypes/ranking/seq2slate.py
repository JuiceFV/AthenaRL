import torch

from enum import Enum
from typing import NamedTuple, Optional


class Seq2SlateMode(Enum):
    r"""
    The mode in which :class:`athena.models.ranking.seq2slate.Seq2SlateTransformerModel` 
    performs.
    """
    #: Returns ranked items and their generative probabilities.
    RANK_MODE = "rank"
    #: Returns generative log probabilities of given target 
    #: sequences (used for REINFORCE training)
    PER_SEQ_LOG_PROB_MODE = "per_sequence_log_prob"
    #: Returns generative log probabilties of each item in 
    #: given sequences (used in TEACHER FORCING training)
    PER_ITEM_LOG_PROB_DIST_MODE = "per_item_log_prob_dist"
    #: Decoding occures only ones, thus not all permutations
    #: are considered
    DECODE_ONE_STEP_MODE = "decode_one_step"
    #: Produce only encoder scores.
    ENCODER_SCORE_MODE = "encoder_score_mode"


class Seq2SlateOutputArch(Enum):
    r"""
    The variation of Seq2Slate model.
    """
    #: Only output encoder scores
    ENCODER_SCORE = "encoder_score"

    #: A decoder outputs a sequence in an autoregressive way
    AUTOREGRESSIVE = "autoregressive"

    #: Using encoder scores, a decoder outputs a sequence using
    #: frechet sort (equivalent to iterative softmax)
    FRECHET_SORT = "frechet_sort"


class Seq2SlateTransformerOutput(NamedTuple):
    r"""
    Every possible output of :class:`athena.models.ranking.seq2slate.Seq2SlateTransformerModel`
    """
    #: Probability distribution over items for every permutation
    ordered_per_item_probas: Optional[torch.Tensor]
    #: Generative probabilities of each permutation, computed as
    #: :math:`P(s) = \prod{P(i)}`
    ordered_per_seq_probas: Optional[torch.Tensor]
    ordered_target_out_idcs: Optional[torch.Tensor]
    per_item_log_probas: Optional[torch.Tensor]
    per_seq_log_probas: Optional[torch.Tensor]
    encoder_scores: Optional[torch.Tensor]