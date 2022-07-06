import torch
import torch.nn as nn

from typing import Dict, List, Optional, Tuple

import athena.core.dtypes as adt
import torch.nn.functional as F
from athena.core.dataclasses import field
from athena.trainers import AthenaLightening, STEP_OUTPUT
from athena.models import Seq2SlateTransformerNetwork
from athena.optim import OptimizerRoster
from athena.core.parameters import Seq2SlateParameters
from athena.nn.rl.variance_reduction import BaselineNetwork, ips_blur, ips_ratio


class Seq2SlateTrainer(AthenaLightening):
    def __init__(
        self,
        reinforce_network: Seq2SlateTransformerNetwork,
        params: Seq2SlateParameters = field(default_factory=Seq2SlateParameters),
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

        self.automatic_optimization = False

        if cpe and propensity_network is None:
            raise RuntimeError("Counterfactual evaluation policy needs propensity logic to be passed.")
        self.cpe = cpe
        self.propensity_network = propensity_network

    def configure_optimizers(self):
        optimizers = []
        optimizers.append(self.reinforce_optimizer.create_optimizer_scheduler(self.reinforce.parameters()))
        if self.baseline:
            optimizers.append(self.baseline_optimizer.create_optimizer_scheduler(self.baseline.parameters()))
        return optimizers

    def importance_sampling(self, model_propensities: torch.Tensor, logged_propensities: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logged_propensities = logged_propensities.reshape(-1, 1)
        importance_weights = ips_ratio(model_propensities, logged_propensities)
        blured_importance_weights = ips_blur(importance_weights, self.params.ips_blur)
        return importance_weights, blured_importance_weights

    def training_step(self, batch: adt.PreprocessedRankingInput, batch_idx: int, optimizer_idx: int = 0) -> STEP_OUTPUT:
        if type(batch) is not adt.PreprocessedRankingInput:
            raise TypeError(f"Batch has to be of type {adt.PreprocessedRankingInput}; got {type(batch)}")

        batch_size = batch.latent_state.repr.shape[0]

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
                f"Shapes of policy gradient entities doesn't match. "
                f"{b.shape} {reward.shape} {model_propensities.shape}"
            )

        ips_weights, blured_ips_weights = self.importance_sampling(model_propensities, batch.target_output_probas)
        if ips_weights.shape != blured_ips_weights.shape != reward.shape:
            raise RuntimeError(
                f"Shapes of policy gradient entities doesn't match. "
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
            self.info("Baseline model is warming up, thus do not update reinforce model.")

        ips_loss = torch.mean(-ips_weights * reward).cpu().detach().numpy()
        blured_ips_loss = torch.mean(-blured_ips_weights * reward).cpu().detach().numpy()
        baseline_loss = b_loss.detach().cpu().numpy().item()
        advantages = (reward - b).detach().cpu().numpy()
        logged_slate_rank_probas = model_propensities.detach().cpu().numpy()

        # TODO: add print interval
        self.info(
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
