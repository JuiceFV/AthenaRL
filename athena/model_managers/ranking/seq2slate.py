from typing import Dict
from athena.core.config import param_hash
from athena.core.dataclasses import dataclass, field
from athena.core.parameters import NormalizationData, NormalizationKey
from athena.model_managers.seq2slate_base import Seq2SlateBase
from athena.net_builder.ranking.seq2slate import Seq2SlateRanking
from athena.net_builder.roster import SlateRankingNetBuilderRoster
from athena.nn.rl.variance_reduction import BaselineNetwork
from athena.prediction.ranking.slate_predictor import Seq2SlatePredictorWrapper
from athena.preprocessing.normalization import get_normalization_data_dim
from athena.trainers.parameters import Seq2SlateTrainerParameters
from athena.trainers.ranking.seq2slate.seq2slate_base import Seq2SlateTrainer


@dataclass
class Seq2Slate(Seq2SlateBase):
    __hash__ = param_hash

    slate_size: int = -1
    num_of_candidates: int = -1
    net_builder: SlateRankingNetBuilderRoster = field(
        default_factory=lambda: SlateRankingNetBuilderRoster(
            Seq2SlateRanking=Seq2SlateRanking()
        )
    )
    trainer_params: Seq2SlateTrainerParameters = field(
        default_factory=Seq2SlateTrainerParameters
    )
    use_baseline: bool = True
    baseline_warmup_batches: int = 0

    def __post_init_post_parse__(self):
        super().__post_init_post_parse__()
        if self.slate_size <= 0:
            raise ValueError(f"Slate size is invalid ({self.slate_size})")
        if self.num_of_candidates <= 0:
            raise ValueError(f"Number of exploring candidates is invalid ({self.num_of_candidates})")

    def build_trainer(
        self,
        normalization_dict: Dict[str, NormalizationData],
    ) -> Seq2SlateTrainer:
        net_builder: Seq2SlateRanking = self.net_builder.value
        state_dim = get_normalization_data_dim(
            normalization_dict[NormalizationKey.STATE].dense_normalization_params
        )
        candidate_dim = get_normalization_data_dim(
            normalization_dict[NormalizationKey.CANDIDATE].dense_normalization_params
        )
        seq2slate_network = net_builder.build_slate_ranking_network(
            latent_state_dim=state_dim,
            candidate_dim=candidate_dim,
            num_of_candidates=self.num_of_candidates,
            slate_size=self.slate_size
        )

        # FIXME: Baseline could be any function which reduces
        # variance that implies faster loss convergance. It woths
        # to give a user ability to set this function manually.
        if self.use_baseline:
            baseline_network = BaselineNetwork(
                latent_state_dim=state_dim,
                dim_feedforward=net_builder.transformer.dim_feedforward,
                nlayers=net_builder.transformer.nlayers
            )
        else:
            self.baseline_warmup_batches = 0

        return Seq2SlateTrainer(
            reinforce_network=seq2slate_network,
            baseline_network=baseline_network,
            baseline_warmup_batches=self.baseline_warmup_batches,
            **self.trainer_params.asdict()
        )

    def build_serving_module(
        self,
        trainer_module: Seq2SlateTrainer,
        normalization_dict: Dict[str, NormalizationData]
    ) -> Seq2SlatePredictorWrapper:
        if not isinstance(trainer_module, Seq2SlateTrainer):
            raise TypeError(f"Wrong trainer module type is passed {type(trainer_module)}")
        net_builder: Seq2SlateRanking = self.net_builder.value
        return net_builder.build_serving_module(
            trainer_module.reinforce,
            normalization_dict[NormalizationKey.STATE].dense_normalization_params,
            normalization_dict[NormalizationKey.CANDIDATE].dense_normalization_params
        )
