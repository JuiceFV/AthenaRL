import torch

from enum import Enum
from typing import NamedTuple, Optional


class Seq2SlateMode(Enum):
    RANK_MODE = "rank"
    PER_SEQ_LOG_PROB_MODE = "per_sequence_log_prob"
    PER_ITEM_LOG_PROB_DIST_MODE = "per_item_log_prob_dist"
    DECODE_ONE_STEP_MODE = "decode_one_step"
    ENCODER_SCORE_MODE = "encoder_score_mode"


class Seq2SlateOutputArch(Enum):
    # Only output encoder scores
    ENCODER_SCORE = "encoder_score"

    # A decoder outputs a sequence in an autoregressive way
    AUTOREGRESSIVE = "autoregressive"

    # Using encoder scores, a decoder outputs a sequence using
    # frechet sort (equivalent to iterative softmax)
    FRECHET_SORT = "frechet_sort"


class Seq2SlateTransformerOutput(NamedTuple):
    ordered_per_item_probas: Optional[torch.Tensor]
    ordered_per_seq_probas: Optional[torch.Tensor]
    ordered_target_out_idcs: Optional[torch.Tensor]
    per_item_log_probas: Optional[torch.Tensor]
    per_seq_log_probas: Optional[torch.Tensor]
    encoder_scores: Optional[torch.Tensor]
