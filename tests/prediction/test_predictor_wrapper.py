import unittest
import torch
import random
import athena.core.dtypes as adt

from typing import Tuple
from athena.core.dtypes import Seq2SlateOutputArch
from athena.core.dtypes.ranking.seq2slate import Seq2SlateMode
from athena.models.ranking.seq2slate import Seq2SlateTransformerNetwork
from athena.prediction.ranking.slate_predictor import Seq2SlateWithPreprocessor, Seq2SlatePredictorWrapper
from athena.preprocessing.preprocessor import Preprocessor
from tests.prediction.utils import apply_variable_slate_size, fake_norm


def seq2slate_ip2rnaking_ip(
    state_input_prototype: Tuple[torch.Tensor, torch.Tensor],
    candidate_input_prototype: Tuple[torch.Tensor, torch.Tensor],
    state_preprocessor: Preprocessor,
    candidate_preprocessor: Preprocessor
) -> adt.PreprocessedRankingInput:
    batch_size, num_of_candidates, candidate_dim = candidate_input_prototype[0].shape
    preprocessed_state = state_preprocessor(state_input_prototype[0], state_input_prototype[1])
    preprocessed_candidates = candidate_preprocessor(
        candidate_input_prototype[0].view(batch_size * num_of_candidates, candidate_dim),
        candidate_input_prototype[1].view(batch_size * num_of_candidates, candidate_dim)
    ).view(batch_size, num_of_candidates, -1)
    source_input_indcs = torch.arange(num_of_candidates).repeat(batch_size, 1) + 2
    return adt.PreprocessedRankingInput.from_tensors(
        state=preprocessed_state,
        source_seq=preprocessed_candidates,
        source_input_indcs=source_input_indcs
    )


class TestPedictorWrapper(unittest.TestCase):
    def validate_seq2slate_output(
        self, expected_output: adt.RankingOutput, wrapper_output: Tuple[torch.Tensor, torch.Tensor]
    ):
        ordered_per_seq_probas = expected_output.ordered_per_seq_probas
        ordered_target_out_indcs = expected_output.ordered_target_out_indcs
        ordered_target_out_indcs -= 2

        self.assertTrue(ordered_per_seq_probas == wrapper_output[0])
        self.assertTrue(torch.all(torch.eq(ordered_target_out_indcs, wrapper_output[1])))

    def _test_seq2slate_wrapper(self, output_arch: Seq2SlateOutputArch):
        state_norm_params = {i: fake_norm() for i in range(1, 5)}
        candidate_norm_params = {i: fake_norm() for i in range(101, 106)}
        state_preprocessor = Preprocessor(state_norm_params)
        candidate_preprocessor = Preprocessor(candidate_norm_params)
        num_of_candidates = 10
        slate_size = 4

        seq2slate = Seq2SlateTransformerNetwork(
            state_dim=len(state_norm_params),
            candidate_dim=len(candidate_norm_params),
            nlayers=2,
            nheads=2,
            dim_model=10,
            dim_feedforward=10,
            max_source_seq_len=num_of_candidates,
            max_target_seq_len=slate_size,
            output_arch=output_arch,
            temperature=1
        )

        seq2slate_with_preprocessor = Seq2SlateWithPreprocessor(
            seq2slate, state_preprocessor, candidate_preprocessor, greedy=True
        )
        wrapper = Seq2SlatePredictorWrapper(seq2slate_with_preprocessor)

        state_ip, candidate_ip = seq2slate_with_preprocessor.input_prototype()
        wrapper_output = wrapper(state_ip, candidate_ip)

        ranking_input = seq2slate_ip2rnaking_ip(
            state_ip, candidate_ip, state_preprocessor, candidate_preprocessor
        )
        expected_output = seq2slate(
            ranking_input,
            mode=Seq2SlateMode.RANK_MODE,
            target_seq_len=num_of_candidates,
            greedy=True
        )
        self.validate_seq2slate_output(expected_output, wrapper_output)

        random_length = random.randint(num_of_candidates + 1, num_of_candidates * 2)
        state_ip, candidate_ip = apply_variable_slate_size(
            seq2slate_with_preprocessor.input_prototype(), random_length
        )
        wrapper_output = wrapper(state_ip, candidate_ip)
        ranking_input = seq2slate_ip2rnaking_ip(
            state_ip, candidate_ip, state_preprocessor, candidate_preprocessor
        )
        expected_output = seq2slate(
            ranking_input,
            mode=Seq2SlateMode.RANK_MODE,
            target_seq_len=random_length,
            greedy=True
        )
        self.validate_seq2slate_output(expected_output, wrapper_output)

    def test_seq2slate_transformer_frechet_sort_wrapper(self):
        self._test_seq2slate_wrapper(output_arch=Seq2SlateOutputArch.FRECHET_SORT)

    def test_seq2slate_transformer_autoregressive_wrapper(self):
        self._test_seq2slate_wrapper(output_arch=Seq2SlateOutputArch.AUTOREGRESSIVE)
