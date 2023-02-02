from typing import Optional, Tuple

import torch
import torch.nn as nn

from athena.core.dtypes.ranking.seq2slate import (Seq2SlateMode,
                                                  Seq2SlateOutputArch)
from athena.models.base import BaseModel
from athena.nn.utils.transformer import PADDING_SYMBOL
from athena.models.ranking.seq2slate import Seq2SlateTransformerNetwork
from athena.preprocessing.preprocessor import Preprocessor


class SlateRankingPreprocessor(BaseModel):
    def __init__(
        self,
        state_preprocessor: Preprocessor,
        candidate_preprocessor: Preprocessor,
        num_of_candidates: int
    ) -> None:
        super().__init__()
        self.state_preprocessor = state_preprocessor
        self.candidate_preprocessor = candidate_preprocessor
        self.num_of_candidates = num_of_candidates

    def input_prototype(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        candidate_input_prototype = self.candidate_preprocessor.input_prototype()
        return (
            self.state_preprocessor.input_prototype(),
            (
                candidate_input_prototype[0].repeat((1, self.num_of_candidates, 1)),
                candidate_input_prototype[1].repeat((1, self.num_of_candidates, 1)),
            )
        )

    def forward(
        self,
        state_with_presence: Tuple[torch.Tensor, torch.Tensor],
        candidate_with_presence: Tuple[torch.Tensor, torch.Tensor]
    ):
        batch_size, max_source_seq_len, candidate_feature_num = candidate_with_presence[0].shape

        preprocessed_state = self.state_preprocessor(state_with_presence[0], state_with_presence[1])
        preprocessed_candidates = self.candidate_preprocessor(
            candidate_with_presence[0].view(batch_size * max_source_seq_len, candidate_feature_num),
            candidate_with_presence[1].view(batch_size * max_source_seq_len, candidate_feature_num)
        ).view(batch_size, max_source_seq_len, -1)
        return preprocessed_state, preprocessed_candidates


class Seq2SlateWithPreprocessor(nn.Module):
    def __init__(
        self,
        model: Seq2SlateTransformerNetwork,
        state_preprocessor: Preprocessor,
        candidate_preprocessor: Preprocessor,
        greedy: bool
    ) -> None:
        super().__init__()
        self.model = model.seq2slate
        self.greedy = greedy
        preprocessor = SlateRankingPreprocessor(state_preprocessor, candidate_preprocessor, model.max_source_seq_len)
        self.input_prototype_data = preprocessor.input_prototype()
        if not self.can_be_traced():
            preprocessor = torch.jit.trace(preprocessor, preprocessor.input_prototype())
        self.preprocessor = preprocessor
        self.state_sorted_features = state_preprocessor.sorted_features
        self.candidate_sorted_features = candidate_preprocessor.sorted_features
        self.state_fid2index = state_preprocessor.fid2index
        self.candidate_fid2index = candidate_preprocessor.fid2index
        self.padding_symbol = PADDING_SYMBOL

    def input_prototype(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        return self.input_prototype_data

    def forward(
        self,
        state_with_presence: Tuple[torch.Tensor, torch.Tensor],
        candidate_with_presence: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        preprocessed_state, preprocessed_candidates = self.preprocessor(state_with_presence, candidate_with_presence)
        batch_size, max_source_seq_len = preprocessed_candidates.shape[:2]
        source_input_indcs = torch.arange(max_source_seq_len).repeat(batch_size, 1) + 2
        output = self.model(
            mode=Seq2SlateMode.RANK_MODE.value,
            state=preprocessed_state,
            source_seq=preprocessed_candidates,
            source_input_indcs=source_input_indcs,
            target_seq_len=max_source_seq_len,
            greedy=self.greedy
        )
        return (output.ordered_per_item_probas, output.ordered_per_seq_probas, output.ordered_target_out_indcs)

    def can_be_traced(self) -> bool:
        output_arch = self.model.output_arch
        return (
            output_arch == Seq2SlateOutputArch.ENCODER_SCORE
            or (output_arch == Seq2SlateOutputArch.FRECHET_SORT and self.greedy)
        )


class Seq2SlatePredictorWrapper(torch.jit.ScriptModule):
    def __init__(self, seq2slate_with_preprocessor: Seq2SlateWithPreprocessor) -> None:
        super().__init__()
        if seq2slate_with_preprocessor.can_be_traced():
            self.seq2slate_with_preprocessor = torch.jit.trace(
                seq2slate_with_preprocessor,
                seq2slate_with_preprocessor.input_prototype()
            )
        else:
            self.seq2slate_with_preprocessor = torch.jit.script(seq2slate_with_preprocessor)

    @torch.jit.script_method
    def forward(
        self,
        state_with_presence: Tuple[torch.Tensor, torch.Tensor],
        candidate_with_presence: Tuple[torch.Tensor, torch.Tensor]
    ):
        _, ordered_per_seq_probas, ordered_target_out_indcs = self.seq2slate_with_preprocessor(
            state_with_presence, candidate_with_presence
        )
        if ordered_per_seq_probas is None or ordered_target_out_indcs is None:
            raise RuntimeError("Got empty model output")
        ordered_target_out_indcs -= 2
        return ordered_per_seq_probas, ordered_target_out_indcs
