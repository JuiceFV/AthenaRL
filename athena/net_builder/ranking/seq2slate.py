from athena.core.dataclasses import dataclass, field
from athena.core.dtypes import Seq2SlateOutputArch
from athena.core.parameters import (NormalizationData, TransformerParams,
                                    param_hash)
from athena.models import Seq2SlateTransformerNetwork
from athena.net_builder.slate_ranking_builder import SlateRankingNetBuilder
from athena.prediction.ranking.slate_predictor import (
    Seq2SlatePredictorWrapper, Seq2SlateWithPreprocessor)
from athena.preprocessing.preprocessor import Preprocessor


@dataclass
class Seq2SlateRanking(SlateRankingNetBuilder):
    __hash__ = param_hash

    output_arch: Seq2SlateOutputArch = Seq2SlateOutputArch.AUTOREGRESSIVE
    temperature: float = 1.0
    transformer: TransformerParams = field(
        default_factory=lambda: TransformerParams(
            nheads=8, dim_model=512, dim_feedforward=2048, nlayers=8
        )
    )

    def build_slate_ranking_network(
        self, state_dim: int, candidate_dim: int, num_of_candidates: int, slate_size: int
    ) -> Seq2SlateTransformerNetwork:
        return Seq2SlateTransformerNetwork(
            state_dim=state_dim,
            candidate_dim=candidate_dim,
            nlayers=self.transformer.nlayers,
            dim_model=self.transformer.dim_model,
            max_source_seq_len=num_of_candidates,
            max_target_seq_len=slate_size,
            output_arch=self.output_arch,
            temperature=self.temperature,
            nheads=self.transformer.nheads,
            dim_feedforward=self.transformer.dim_feedforward,
            state_embed_dim=self.transformer.state_embed_dim
        )

    def build_serving_module(
        self,
        network: Seq2SlateTransformerNetwork,
        state_normalization_data: NormalizationData,
        candidate_normalization_data: NormalizationData
    ) -> Seq2SlatePredictorWrapper:
        state_preprocessor = Preprocessor(state_normalization_data.dense_normalization_params)
        candidate_preprocessor = Preprocessor(candidate_normalization_data.dense_normalization_params)
        seq2slate_with_preprocessor = Seq2SlateWithPreprocessor(
            network.cpu_model().eval(), state_preprocessor, candidate_preprocessor, True
        )
        return Seq2SlatePredictorWrapper(seq2slate_with_preprocessor)
