from typing import Dict

from athena.core.config import param_hash
from athena.core.dataclasses import dataclass, field
from athena.core.dtypes.rl.options import RLOptions
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
    use_baseline_function: bool = True
    baseline_warmup_batches: int = 0

    def __post_init_post_parse__(self):
        super().__post_init_post_parse__()
        if self.slate_size <= 0:
            raise ValueError(f"Slate size is invalid ({self.slate_size})")
        if self.num_of_candidates <= 0:
            raise ValueError(f"Number of exploring candidates is invalid ({self.num_of_candidates})")
        if not self.use_baseline_function:
            self.warning("Basline function isn't in use, so baseline_warmup_batches is set to be 0.")
            self.baseline_warmup_batches = 0

    def build_trainer(
        self,
        use_gpu: bool,
        rl_options: RLOptions,
        normalization_dict: Dict[str, NormalizationData],
    ) -> Seq2SlateTrainer:
        net_builder: Seq2SlateRanking = self.net_builder.value
        state_dim = get_normalization_data_dim(
            normalization_dict[NormalizationKey.STATE].dense_normalization_params
        )
        candidate_dim = get_normalization_data_dim(
            normalization_dict[NormalizationKey.CANDIDATE].dense_normalization_params
        )

        if not (state_dim and candidate_dim):
            raise RuntimeError(
                f"Unable to infer metadata either from state ({state_dim}) or candidate ({candidate_dim})."
            )

        seq2slate_network = net_builder.build_slate_ranking_network(
            state_dim=state_dim,
            candidate_dim=candidate_dim,
            num_of_candidates=self.num_of_candidates,
            slate_size=self.slate_size
        )

        if self.use_baseline_function:
            baseline = BaselineNetwork(
                state_dim=state_dim,
                dim_feedforward=net_builder.transformer.dim_feedforward,
                nlayers=net_builder.transformer.nlayers
            )

        return Seq2SlateTrainer(
            reinforce_network=seq2slate_network,
            baseline_network=baseline,
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
            normalization_dict[NormalizationKey.STATE],
            normalization_dict[NormalizationKey.CANDIDATE]
        )
