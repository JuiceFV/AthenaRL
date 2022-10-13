import unittest

import numpy.testing as npt
import torch
from athena.prediction.ranking.slate_predictor import Seq2SlateWithPreprocessor
from athena.models.ranking.seq2slate import Seq2SlateTransformerNetwork
from athena.preprocessing.preprocessor import Preprocessor
from athena.core.dtypes.ranking.seq2slate import Seq2SlateOutputArch
from tests.prediction.utils import fake_norm, apply_variable_slate_size


class TestModelWithPreprocessor(unittest.TestCase):
    def verify_results(
        self, expected_output: torch.Tensor, scripted_output: torch.Tensor
    ) -> None:
        for i, j in zip(expected_output, scripted_output):
            npt.assert_array_equal(i.detach(), j.detach())

    def _test_seq2slate_model_preprocessor(self, output_arch: Seq2SlateOutputArch):
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
            temperature=0.5
        )

        seq2slate_with_preprocessor = Seq2SlateWithPreprocessor(
            seq2slate, state_preprocessor, candidate_preprocessor, greedy=True
        )
        input_prototype = seq2slate_with_preprocessor.input_prototype()

        if seq2slate_with_preprocessor.can_be_traced():
            seq2slate_with_preprocessor_jit = torch.jit.trace(
                seq2slate_with_preprocessor,
                seq2slate_with_preprocessor.input_prototype()
            )
        else:
            seq2slate_with_preprocessor_jit = torch.jit.script(
                seq2slate_with_preprocessor,
            )

        expected_output = seq2slate_with_preprocessor(*input_prototype)
        jit_output = seq2slate_with_preprocessor_jit(*input_prototype)
        self.verify_results(expected_output, jit_output)

        input_prototype = apply_variable_slate_size(input_prototype, 20)
        expected_output = seq2slate_with_preprocessor(*input_prototype)
        jit_output = seq2slate_with_preprocessor_jit(*input_prototype)
        self.verify_results(expected_output, jit_output)

    def test_seq2slate_transformer_frechet_sort_model_with_preprocessor(self):
        self._test_seq2slate_model_preprocessor(Seq2SlateOutputArch.FRECHET_SORT)

    def test_seq2slate_transformer_autoregressive_model_with_preprocessor(self):
        self._test_seq2slate_model_preprocessor(Seq2SlateOutputArch.AUTOREGRESSIVE)
