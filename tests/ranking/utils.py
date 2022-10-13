import logging
import math
from itertools import permutations
from typing import List, Optional, Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

import athena.core.dtypes as adt
from athena import gather
from athena.core.dtypes.ranking.seq2slate import (Seq2SlateMode,
                                                  Seq2SlateOutputArch,
                                                  Seq2SlateVersion)
from athena.core.parameters import Seq2SlateParams
from athena.models.base import BaseModel
from athena.models.ranking.seq2slate import Seq2SlateTransformerNetwork
from athena.optim.optimizer_roster import OptimizerRoster
from athena.trainers.ranking.seq2slate.seq2slate_base import Seq2SlateTrainer

logger = logging.getLogger(__name__)

GLOBAL_CANDIDATES: Optional[torch.Tensor] = None
OFF_POLICY = "off_policy"
ON_POLICY = "on_policy"


def post_preprocess_batch(
    seq2slate_net: Seq2SlateTransformerNetwork,
    num_of_candidates: int,
    batch: adt.PreprocessedRankingInput,
    device: torch.device,
    epoch: int
) -> adt.PreprocessedRankingInput:
    model_propensity, model_actions, reward = rank_on_policy_and_eval(
        seq2slate_net, batch, num_of_candidates, greedy=False
    )
    batch = adt.PreprocessedRankingInput.from_input(
        state=batch.state.dense_features,
        candidates=batch.source_seq.dense_features,
        device=device,
        actions=model_actions,
        logged_propensities=model_propensity,
        slate_reward=-reward
    )
    logger.info(f"Epoch {epoch} mean on_policy reward: {torch.mean(reward)}")
    logger.info(f"Epoch {epoch} mean model_propensity: {torch.mean(model_propensity)}")
    return batch


class Seq2SlateOnPolicyTrainer(Seq2SlateTrainer):
    def on_train_batch_start(
        self,
        batch: adt.PreprocessedRankingInput,
        batch_idx: int,
    ) -> Optional[int]:
        new_batch = post_preprocess_batch(
            self.reinforce,
            self.reinforce.max_source_seq_len,
            batch,
            batch.state.dense_features.device,
            self.current_epoch
        )
        for attr in dir(new_batch):
            if not attr.startswith("__") and not callable(getattr(new_batch, attr)):
                setattr(batch, attr, getattr(new_batch, attr))
        super(Seq2SlateTrainer, self).on_train_batch_start(batch, batch_idx)


def create_seq2slate_net(
    num_of_candidates: int,
    candidate_dim: int,
    hidden_size: int,
    output_arch: Seq2SlateOutputArch,
    temperature: float,
    device: torch.device
) -> Seq2SlateTransformerNetwork:
    return Seq2SlateTransformerNetwork(
        state_dim=1,
        candidate_dim=candidate_dim,
        nlayers=2,
        nheads=2,
        dim_model=hidden_size,
        dim_feedforward=hidden_size,
        max_source_seq_len=num_of_candidates,
        max_target_seq_len=num_of_candidates,
        output_arch=output_arch,
        temperature=temperature,
        state_embed_dim=1
    ).to(device)


@torch.no_grad()
def create_batch(
    batch_size: int,
    num_of_candidates: int,
    candidate_dim: int,
    device: torch.device,
    learning_method: str,
    diverse_input: bool = False
) -> adt.PreprocessedRankingInput:
    state = torch.zeros(batch_size, 1)
    if diverse_input:
        candidates = torch.randint(5, (batch_size, num_of_candidates, candidate_dim)).float()
    else:
        global GLOBAL_CANDIDATES
        if GLOBAL_CANDIDATES is None or GLOBAL_CANDIDATES.shape != (batch_size, num_of_candidates, candidate_dim):
            candidates = torch.randint(5, (batch_size, num_of_candidates, candidate_dim)).float()
            candidates[1:] = candidates[0]
            GLOBAL_CANDIDATES = candidates
        else:
            candidates = GLOBAL_CANDIDATES

    batch_dict = {
        "state": state,
        "candidates": candidates,
        "device": device
    }
    if learning_method == OFF_POLICY:
        actions = torch.stack([torch.randperm(num_of_candidates) for _ in range(batch_size)])
        propensity = torch.full((batch_size, 1), 1.0 / math.factorial(num_of_candidates))
        ranked_cities = gather(candidates, actions)
        reward = compute_reward(ranked_cities)
        batch_dict["actions"] = actions
        batch_dict["logged_propensities"] = propensity
        batch_dict["slate_reward"] = -reward

    batch = adt.PreprocessedRankingInput.from_input(**batch_dict)
    logger.info("Generate a batch")
    return batch


def compute_reward(ordered_cities: torch.Tensor) -> torch.Tensor:
    assert len(ordered_cities.shape) == 3
    ranked_cities_offset = torch.roll(ordered_cities, shifts=1, dims=1)
    return torch.sqrt(((ranked_cities_offset - ordered_cities) ** 2).sum(-1)).sum(-1).unsqueeze(1)


def per_item_to_per_seq_log_probas(
    per_item_log_probas: torch.Tensor, target_output_indcs: torch.Tensor
) -> torch.Tensor:
    log_probas = torch.gather(per_item_log_probas, 2, target_output_indcs.unsqueeze(2)).squeeze(2)
    return log_probas.sum(dim=1, keepdim=True)


@torch.no_grad()
def rank_on_policy(
    model: BaseModel, batch: adt.PreprocessedRankingInput, target_seq_len: int, greedy: bool
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    model.eval()
    ordered_output: adt.RankingOutput = model(
        batch, mode=Seq2SlateMode.RANK_MODE, target_seq_len=target_seq_len, greedy=greedy)
    ordered_slate_proba = ordered_output.ordered_per_seq_probas
    ordered_items = ordered_output.ordered_target_out_indcs - 2
    model.train()
    return ordered_slate_proba, ordered_items


def run_seq2slate_tsp(
    batch_size: int,
    epochs: int,
    num_of_candidates: int,
    num_batches: int,
    hidden_size: int,
    diverse_input: bool,
    lr: float,
    expect_reward_threshold: float,
    learning_method: str,
    policy_optimizer_interval: int,
    device: torch.device
):
    pl.seed_everything(0)

    candidate_dim = 2
    eval_sample_size = 1

    train_batches, test_batch = create_train_and_test_batches(
        batch_size,
        num_of_candidates,
        candidate_dim,
        device,
        num_batches,
        learning_method,
        diverse_input
    )
    best_test_possible_reward = compute_best_reward(test_batch.source_seq.dense_features)

    seq2slate_net = create_seq2slate_net(
        num_of_candidates,
        candidate_dim,
        hidden_size,
        Seq2SlateOutputArch.AUTOREGRESSIVE,
        1.0,
        device
    )

    trainer = create_trainer(seq2slate_net, learning_method, lr, policy_optimizer_interval)

    def evaluate():
        best_test_reward = torch.full((batch_size,), 1e9).to(device)
        for _ in range(eval_sample_size):
            model_propensities, _, reward = rank_on_policy_and_eval(
                seq2slate_net.to(device), test_batch, num_of_candidates, greedy=True
            )
            best_test_reward = torch.where(reward < best_test_reward, reward, best_test_reward)
        logger.info(
            f"Test mean model_propensities {torch.mean(model_propensities)}, "
            f"Test mean reward: {torch.mean(best_test_reward)}, "
            f"best possible reward {best_test_possible_reward}"
        )
        if torch.any(torch.isnan(model_propensities)):
            raise Exception("Model propensities contain NaNs")
        ratio = torch.mean(best_test_reward) / best_test_possible_reward
        return ratio < expect_reward_threshold, ratio

    evaluate()

    training_data = DataLoader(train_batches, collate_fn=lambda x: x[0])
    pl_trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=None if device == torch.device("cpu") else 1,
        logger=False
    )
    pl_trainer.fit(trainer, training_data)

    result, ratio = evaluate()

    assert result, f"Test failed because it didn't reach expected test reward,  {ratio} > {expect_reward_threshold}"


def create_trainer(
    seq2slate_net: Seq2SlateTransformerNetwork,
    learning_method: str,
    lr: float,
    policy_optimizer_interval: int,
):
    if learning_method not in [ON_POLICY, OFF_POLICY]:
        raise ValueError("learning_method must be one of [ON_POLICY, OFF_POLICY]")

    if learning_method == ON_POLICY:
        on_policy = True
        trainer_cls = Seq2SlateOnPolicyTrainer
    else:
        on_policy = False
        trainer_cls = Seq2SlateTrainer

    seq2slate_params = Seq2SlateParams(on_policy=on_policy, version=Seq2SlateVersion.REINFORCEMENT_LEARNING)

    params_dict = {
        "reinforce_network": seq2slate_net,
        "params": seq2slate_params,
        "policy_optimizer": OptimizerRoster.default(lr=lr),
        "policy_optimizer_interval": policy_optimizer_interval
    }
    return trainer_cls(**params_dict)


@torch.no_grad()
def rank_on_policy_and_eval(
    seq2slate_net: Seq2SlateTransformerNetwork,
    batch: adt.PreprocessedRankingInput,
    target_seq_len: int,
    greedy: bool
):
    model_propensity, model_actions = rank_on_policy(seq2slate_net, batch, target_seq_len, greedy)
    ordered_cities = gather(batch.source_seq.dense_features, model_actions)
    reward = compute_reward(ordered_cities)
    return model_propensity, model_actions, reward


def create_train_and_test_batches(
    batch_size: int,
    num_of_candidates: int,
    candidate_dim: int,
    device: torch.device,
    num_train_batches: int,
    learning_method: str,
    diverse_input: bool
) -> Tuple[List[adt.PreprocessedRankingInput], adt.PreprocessedRankingInput]:
    train_batches = [
        create_batch(batch_size, num_of_candidates, candidate_dim, device, learning_method, diverse_input)
        for _ in range(num_train_batches)
    ]

    if diverse_input:
        test_batch = create_batch(batch_size, num_of_candidates, candidate_dim, device, learning_method, diverse_input)
    else:
        test_batch = train_batches[0]

    return train_batches, test_batch


def compute_best_reward(input_cities: torch.Tensor) -> torch.Tensor:
    batch_size, num_of_candidates, _ = input_cities.shape
    all_perms = torch.tensor(list(permutations(torch.arange(num_of_candidates), num_of_candidates)))
    res = [
        compute_reward(gather(input_cities, perm.repeat(batch_size, 1)))
        for perm in all_perms
    ]
    res = torch.cat(res, dim=1)
    best_possible_reward = torch.min(res, dim=1).values
    best_possible_reward_mean = torch.mean(best_possible_reward)
    return best_possible_reward_mean
