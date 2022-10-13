import logging
import unittest
import pytorch_lightning as pl
import torch

from athena.core.dtypes import Seq2SlateOutputArch
from athena.prediction.ranking.slate_predictor import Seq2SlateWithPreprocessor
from athena.preprocessing.identify_types import Ftype
from athena.models.ranking.seq2slate import Seq2SlateTransformerNetwork, Seq2SlateTransformerModel
from athena.core.parameters import NormalizationData, NormalizationParams
from athena.preprocessing.preprocessor import Preprocessor
logger = logging.getLogger(__name__)


class TestSeq2SlateInference(unittest.TestCase):
    def setUp(self):
        pl.seed_everything(0)

    def test_seq2slate_scriptable(self):
        state_dim = 2
        candidate_dim = 3
        nlayers = 2
        nheads = 2
        dim_model = 128
        dim_feedforward = 128
        num_of_candidates = 8
        slate_size = 8
        output_arch = Seq2SlateOutputArch.AUTOREGRESSIVE
        temperature = 1.0
        greedy = True

        seq2slate = Seq2SlateTransformerModel(
            state_dim=state_dim,
            candidate_dim=candidate_dim,
            nlayers=nlayers,
            nheads=nheads,
            dim_model=dim_model,
            dim_feedforward=dim_feedforward,
            max_source_seq_len=num_of_candidates,
            max_target_seq_len=slate_size,
            output_arch=output_arch,
            temperature=temperature
        )
        _ = torch.jit.script(seq2slate)

        seq2slate_net = Seq2SlateTransformerNetwork(
            state_dim=state_dim,
            candidate_dim=candidate_dim,
            nlayers=nlayers,
            nheads=nheads,
            dim_model=dim_model,
            dim_feedforward=dim_feedforward,
            max_source_seq_len=num_of_candidates,
            max_target_seq_len=slate_size,
            output_arch=output_arch,
            temperature=temperature
        )

        state_normalization_data = NormalizationData(
            dense_normalization_params={
                0: NormalizationParams(ftype=Ftype.DO_NOT_PREPROCESS),
                1: NormalizationParams(ftype=Ftype.DO_NOT_PREPROCESS)
            }
        )

        candidate_normalization_data = NormalizationData(
            dense_normalization_params={
                0: NormalizationParams(ftype=Ftype.DO_NOT_PREPROCESS),
                1: NormalizationParams(ftype=Ftype.DO_NOT_PREPROCESS),
                2: NormalizationParams(ftype=Ftype.DO_NOT_PREPROCESS)
            }

        )

        state_preprocessor = Preprocessor(state_normalization_data.dense_normalization_params)
        candidate_preprocessor = Preprocessor(candidate_normalization_data.dense_normalization_params)

        seq2slate_with_preprocessor = Seq2SlateWithPreprocessor(
            seq2slate_net.eval(),
            state_preprocessor,
            candidate_preprocessor,
            greedy
        )
        torch.jit.script(seq2slate_with_preprocessor)
