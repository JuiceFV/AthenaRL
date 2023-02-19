from typing import List, Optional, Tuple

import torch
import logging
import torch.nn as nn
import torch.nn.functional as F

import athena.core.dtypes as adt
from athena.core.dataclasses import field
from athena.core.dtypes.ranking.seq2slate import Seq2SlateMode
from athena.core.parameters import Seq2SlateParams
from athena.evaluation.rl.eval_on_batch import EvaluationOnBatch
from athena.metrics.roster import MetricRoster
from athena.models import Seq2SlateTransformerNetwork
from athena.nn.rl.variance_reduction import (BaselineNetwork, ips_blur,
                                             ips_ratio)
from athena.optim import OptimizerRoster
from athena.trainers import STEP_OUTPUT, AthenaLightening
from athena.core.tensorboard import SummaryWriterContext


logger = logging.getLogger(__name__)


class Seq2SlateTrainer(AthenaLightening):
    def __init__(
        self,
        reinforce_network: Seq2SlateTransformerNetwork,
        params: Seq2SlateParams = field(default_factory=Seq2SlateParams),
        metric: MetricRoster = field(default_factory=MetricRoster.default_ndcg),
        baseline_network: Optional[BaselineNetwork] = None,
        baseline_warmup_batches: int = 0,
        policy_optimizer: OptimizerRoster = field(default_factory=OptimizerRoster.default),
        baseline_optimizer: OptimizerRoster = field(default_factory=OptimizerRoster.default),
        policy_optimizer_interval: int = 1,
        cpe: bool = False,
        propensity_network: Optional[nn.Module] = None
    ) -> None:
        super().__init__()
        self.reinforce = reinforce_network
        self.params = params
        self.policy_optimizer_interval = policy_optimizer_interval

        self.baseline = baseline_network
        self.baseline_warmup_batches = baseline_warmup_batches

        self.reinforce_optimizer = policy_optimizer
        if self.baseline:
            self.baseline_optimizer = baseline_optimizer

        if self.params.on_policy:
            self.ranking_measure = metric

        self.automatic_optimization = False

        if cpe and propensity_network is None:
            raise RuntimeError("Counterfactual evaluation policy needs propensity logic to be passed.")
        self.cpe = cpe
        self.propensity_network = propensity_network
        SummaryWriterContext.add_graph(self.reinforce)

    def configure_optimizers(self):
        optimizers = []
        optimizers.append(self.reinforce_optimizer.create_optimizer_scheduler(self.reinforce.parameters()))
        if self.baseline:
            optimizers.append(self.baseline_optimizer.create_optimizer_scheduler(self.baseline.parameters()))
        return optimizers

    def importance_sampling(
        self, model_propensities: torch.Tensor, logged_propensities: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logged_propensities = logged_propensities.reshape(-1, 1)
        importance_weights = ips_ratio(model_propensities, logged_propensities)
        blured_importance_weights = ips_blur(importance_weights, self.params.ips_blur)
        return importance_weights, blured_importance_weights

    def training_step(self, batch: adt.PreprocessedRankingInput, batch_idx: int) -> STEP_OUTPUT:
        if type(batch) is not adt.PreprocessedRankingInput:
            raise TypeError(f"Batch has to be of type {adt.PreprocessedRankingInput}; got {type(batch)}")

        batch_size = batch.state.dense_features.shape[0]

        reward = batch.slate_reward
        if reward is None:
            raise ValueError(f"Expected some reward for the slate but got {None}.")

        optimizers = self.optimizers()
        if self.baseline:
            if len(optimizers) != 2:
                raise RuntimeError(f"Expected only 2 optimizers; got {len(optimizers)}")
            baseline_optimizer = optimizers[1]
        else:
            if len(optimizers) != 1:
                raise RuntimeError(f"Expected only 1 optimizer; but got {len(optimizers)} optimizers.")
        reinforce_optimizer = optimizers[0]

        b = torch.zeros_like(reward)
        b_loss = torch.zeros(1)
        if self.baseline:
            b = self.baseline(batch)
            b_loss = 1.0 / batch_size * torch.sum((b - reward)**2)
            baseline_optimizer.zero_grad()
            self.manual_backward(b_loss)
            baseline_optimizer.step()

        model_propensities = torch.exp(self.reinforce(batch, mode=adt.Seq2SlateMode.PER_SEQ_LOG_PROB_MODE).log_probas)

        b = b.detach()
        if b.shape != reward.shape != model_propensities.shape:
            raise RuntimeError(
                f"Shapes of policy gradient entities don't match. "
                f"{b.shape} {reward.shape} {model_propensities.shape}"
            )

        ips_weights, blured_ips_weights = self.importance_sampling(model_propensities, batch.target_output_probas)
        if ips_weights.shape != blured_ips_weights.shape != reward.shape:
            raise RuntimeError(
                f"Shapes of policy gradient entities don't match. "
                f"{ips_weights.shape} {blured_ips_weights.shape} {reward.shape}"
            )

        if (
            reward.requires_grad
            or batch.target_output_probas.requires_grad
            or not ips_weights.requires_grad
            or not blured_ips_weights.requires_grad
            or b.requires_grad
        ):
            raise RuntimeError("Gradient should be computed only for the model_propensitites.")

        batch_jacobian_loss = -blured_ips_weights * (reward - b)
        jacobian_loss = torch.mean(batch_jacobian_loss)

        if self.baseline is None or (self.all_batches_processed + 1) >= self.baseline_warmup_batches:
            self.manual_backward(jacobian_loss)
            if (self.all_batches_processed + 1) % self.policy_optimizer_interval == 0:
                reinforce_optimizer.step()
                reinforce_optimizer.zero_grad()
        else:
            logger.info("Baseline model is warming up, thus do not update reinforce model.")

        ips_loss = torch.mean(-ips_weights * reward).cpu().detach().numpy()
        blured_ips_loss = torch.mean(-blured_ips_weights * reward).cpu().detach().numpy()
        baseline_loss = b_loss.detach().cpu().numpy().item()
        advantages = (reward - b).detach().cpu().numpy()
        logged_slate_rank_probas = model_propensities.detach().cpu().numpy()

        # TODO: add print interval
        logger.info(
            f"{self.all_batches_processed + 1} batch: "
            f"ips_loss={ips_loss}, "
            f"blured_ips_loss={blured_ips_loss}, "
            f"baseline_loss={baseline_loss}, "
            f"max_ips_weight={torch.max(ips_weights)}, "
            f"mean_ips_weight={torch.mean(ips_weights)}, "
            f"is_gradient_updated={(self.all_batches_processed + 1) % self.policy_optimizer_interval == 0}"
        )

        self.reporter.log(
            train_ips_score=torch.tensor(ips_loss).reshape(1),
            train_blured_ips_score=torch.tensor(blured_ips_loss).reshape(1),
            train_baseline_loss=torch.tensor(baseline_loss).reshape(1),
            train_logged_slate_rank_probas=torch.FloatTensor(logged_slate_rank_probas),
            train_ips_ratio=ips_weights,
            train_blured_ips_ratio=blured_ips_weights,
            train_advantages=advantages
        )
        SummaryWriterContext.increase_global_step()

    def validation_step(self, batch: adt.PreprocessedRankingInput, batch_idx: int) -> Optional[STEP_OUTPUT]:
        reinforce = self.reinforce

        if reinforce.training:
            raise ValueError("The evaluation process is on, but training is still True.")

        logged_slate_rank_probas = torch.exp(
            reinforce(batch, mode=adt.Seq2SlateMode.PER_SEQ_LOG_PROB_MODE).log_probas.detach().flatten().cpu()
        )

        b = torch.zeros_like(batch.slate_reward)
        b_eval_loss = torch.tensor([0.0]).reshape(1)
        if self.baseline:
            b = self.baseline(batch).detach()
            b_eval_loss = F.mse_loss(b, batch.slate_reward).cpu().reshape(1)

        eval_advantages = (batch.slate_reward - b).flatten().cpu()

        ordered_slate_output: adt.RankingOutput = reinforce(batch, adt.Seq2SlateMode.RANK_MODE, greedy=True)
        ordered_slate_rank_probas = ordered_slate_output.ordered_per_seq_probas.cpu()

        self.reporter.log(
            baseline_eval_loss=b_eval_loss,
            eval_advantages=eval_advantages,
            logged_slate_rank_probas=logged_slate_rank_probas,
            ordered_slate_rank_probas=ordered_slate_rank_probas,
        )

        if not self.cpe:
            return None

        eob_greedy = EvaluationOnBatch.from_seq2slate(reinforce, self.propensity_network, batch, greedy_eval=True)
        eob_nongreedy = EvaluationOnBatch.from_seq2slate(reinforce, self.propensity_network, batch, greedy_eval=False)

        return eob_greedy, eob_nongreedy

    def on_train_batch_start(self, batch: adt.PreprocessedRankingInput, batch_idx: int) -> Optional[int]:
        if self.params.on_policy:
            with torch.no_grad():
                # TODO: Remove once padding issue is resolved
                num_of_candidates = min(self.reinforce.max_source_seq_len, batch.target_output_indcs.size(1))
                model_propensity, model_actions = _rank_on_policy(
                    self.reinforce, batch, num_of_candidates, False
                )
                gain = torch.arange(num_of_candidates, 0, -1) * torch.ones_like(model_actions)
                ordered_scores = gain.gather(1, model_actions)

                logged_indcs = batch.target_output_indcs - 2
                positional_reward = batch.position_reward
                true_scores = positional_reward.gather(1, logged_indcs)

                slate_reward = self.ranking_measure(true_scores, ordered_scores).unsqueeze(1)

            on_policy_batch = adt.PreprocessedRankingInput.from_input(
                state=batch.state.dense_features,
                candidates=batch.source_seq.dense_features,
                device=batch.state.dense_features.device,
                actions=model_actions,
                logged_propensities=model_propensity,
                slate_reward=-slate_reward
            )

            for attr in dir(on_policy_batch):
                if not attr.startswith("__") and not callable(getattr(on_policy_batch, attr)):
                    setattr(batch, attr, getattr(on_policy_batch, attr))
        super().on_train_batch_start(batch, batch_idx)

    def validation_epoch_end(self, outputs: Optional[List[Tuple[EvaluationOnBatch, EvaluationOnBatch]]]) -> None:
        if self.cpe:
            if outputs is None:
                raise RuntimeError("If counterfactual evaluation policy is on, batches' evaluators are expected.")

            eobs_greedy, eobs_nongreedy = None, None
            for eob_greedy, eob_nongreedy in outputs:
                if eobs_greedy is None and eobs_nongreedy is None:
                    eobs_greedy = eob_greedy
                    eobs_nongreedy = eob_nongreedy
                else:
                    eobs_greedy.append(eob_greedy)
                    eobs_nongreedy.append(eob_nongreedy)
            self.reporter.log(
                eobs_greedy=eobs_greedy,
                eobs_nongreedy=eob_nongreedy
            )


@torch.no_grad()
def _rank_on_policy(
    reinforce: Seq2SlateTransformerNetwork,
    batch: adt.PreprocessedRankingInput,
    num_of_candidates: int,
    greedy: bool
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    reinforce.eval()
    ordered_output: adt.RankingOutput = reinforce(
        batch, mode=Seq2SlateMode.RANK_MODE, target_seq_len=num_of_candidates, greedy=greedy
    )
    ordered_slate_proba = ordered_output.ordered_per_seq_probas
    ordered_items = ordered_output.ordered_target_out_indcs - 2
    reinforce.train()
    return ordered_slate_proba, ordered_items
