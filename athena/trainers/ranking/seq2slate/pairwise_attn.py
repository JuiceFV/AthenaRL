from typing import Generator, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import dcg_score, ndcg_score

from athena.core.dataclasses import field
from athena.core.dtypes import PreprocessedRankingInput
from athena.core.dtypes.ranking.base import RankingOutput
from athena.core.dtypes.ranking.seq2slate import Seq2SlateMode
from athena.models import Seq2SlateTransformerNetwork
from athena.optim import OptimizerRoster
from athena.trainers import AthenaLightening
from athena.trainers.athena_lightening import STEP_OUTPUT


class Seq2SlatePairwiseAttnTrainer(AthenaLightening):
    def __init__(
        self,
        seq2slate_net: Seq2SlateTransformerNetwork,
        slate_size: int,
        cpe: bool,
        optimizer: OptimizerRoster = field(default_factory=OptimizerRoster.default)
    ) -> None:
        super().__init__()
        self.network = seq2slate_net
        self.slate_size = slate_size
        self.cpe = cpe
        self.optimizer = optimizer
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.kldiv_loss = nn.KLDivLoss(reduction="batchmean")

    def configure_optimizers(self):
        optimizers = []
        optimizers.append(
            self.optimizer.create_optimizer_scheduler(
                self.network.parameters()
            )
        )
        return optimizers

    def train_step_gen(
        self, training_batch: PreprocessedRankingInput, batch_idx: int
    ) -> Generator[STEP_OUTPUT, None, None]:
        if type(training_batch) is not PreprocessedRankingInput:
            raise TypeError(
                f"Incompatible batch type {type(training_batch)}; "
                f"Should be {PreprocessedRankingInput}"
            )

        encoder_scores: torch.Tensor = self.network(
            training_batch, mode=Seq2SlateMode.ENCODER_SCORE_MODE
        ).encoder_scores

        loss: torch.Tensor = self.kldiv_loss(
            self.log_softmax(encoder_scores), training_batch.position_reward
        )

        detached_loss = loss.detach().cpu()
        self.reporter.log(train_cross_entropy_loss=detached_loss)

        yield loss

    def validation_step(
        self, validation_batch: PreprocessedRankingInput, batch_idx: int
    ) -> Optional[STEP_OUTPUT]:
        batch_size = validation_batch.position_reward.shape[0]

        encoder_scores: torch.Tensor = self.network(
            validation_batch, mode=Seq2SlateMode.ENCODER_SCORE_MODE
        ).encoder_scores

        if not (encoder_scores.shape[1] == validation_batch.position_reward.shape[1] == self.slate_size):
            # TODO: Specify
            raise RuntimeError("Unexpected shapes")

        cross_entropy_loss: torch.Tensor = self.kldiv_loss(
            self.log_softmax(encoder_scores), validation_batch.position_reward
        )

        if not self.cpe:
            self.reporter.log(eval_cross_entropy_loss=cross_entropy_loss)
            return None

        ordered_output: RankingOutput = self.network(
            validation_batch, mode=Seq2SlateMode.RANK_MODE, greedy=True
        )

        ordered_indcs = (ordered_output.ordered_target_out_indcs - 2).cpu().numpy()
        logged_indcs = (validation_batch.target_output_indcs - 2).cpu().numpy()
        gain = np.arange(self.slate_size, 0, -1)

        # TODO: beautify
        batch_dcg = []
        batch_ndcg = []
        batch_base_dcg = []
        batch_base_ndcg = []
        for i in range(batch_size):
            if (not torch.any(validation_batch.position_reward[i].bool())) or (torch.all(validation_batch.position_reward[i].bool())):
                continue

            ordered_scores = np.zeros(self.slate_size)
            ordered_scores[ordered_indcs[i]] = gain
            truth_scores = np.zeros(self.slate_size)
            truth_scores[logged_indcs[i]] = validation_batch.position_reward[i].cpu().numpy()
            base_scores = np.zeros(self.slate_size)
            base_scores[logged_indcs[i]] = gain

            ordered_scores = np.expand_dims(ordered_scores, axis=0)
            truth_scores = np.expand_dims(truth_scores, axis=0)
            base_scores = np.expand_dims(base_scores, axis=0)

            batch_dcg.append(dcg_score(truth_scores, ordered_scores))
            batch_ndcg.append(ndcg_score(truth_scores, ordered_scores))
            batch_base_dcg.append(dcg_score(truth_scores, base_scores))
            batch_base_ndcg.append(ndcg_score(truth_scores, base_scores))

        self.reporter.log(
            eval_dcg=torch.mean(torch.tensor(batch_dcg)).reshape(1),
            eval_ndcg=torch.mean(torch.tensor(batch_ndcg)).reshape(1),
            eval_base_dcg=torch.mean(torch.tensor(batch_base_dcg)).reshape(1),
            eval_base_ndcg=torch.mean(torch.tensor(batch_base_ndcg)).reshape(1)
        )
