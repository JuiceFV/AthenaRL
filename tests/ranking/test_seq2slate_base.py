import itertools
import logging
import unittest
from copy import deepcopy

import athena.core.dtypes as adt
import numpy as np
import numpy.testing as npt
import pytorch_lightning as pl
import torch
from athena.core.dtypes import IPSBlurMethod, Seq2SlateOutputArch
from athena.core.dtypes.ranking.seq2slate import Seq2SlateMode
from athena.core.dtypes.rl.base import IPSBlur
from athena.core.parameters import Seq2SlateParams
from athena.models.ranking.seq2slate import Seq2SlateTransformerNetwork
from athena.nn.arch.samplers import FrechetSort
from athena.nn.rl.variance_reduction import ips_blur
from athena.optim.optimizer_roster import OptimizerRoster, opt_classes
from athena.trainers import Seq2SlateTrainer
from parameterized import parameterized
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

output_arch_list = [Seq2SlateOutputArch.FRECHET_SORT, Seq2SlateOutputArch.AUTOREGRESSIVE]
policy_optimizer_interval_list = [1, 5]
blur_method_list = [IPSBlurMethod.UNIVERSAL, IPSBlurMethod.AGGRESSIVE]
blur_max_list = [1.0, 10.0]
frechet_sort_shape_list = [0.1, 0.5, 1.0]


class Seq2SlateOnPolicyTrainer(Seq2SlateTrainer):
    def on_train_batch_start(self, batch: adt.PreprocessedRankingInput, batch_idx: int):
        return super(Seq2SlateTrainer, self).on_train_batch_start(batch, batch_idx)


def create_trainer(
    net: Seq2SlateTransformerNetwork,
    lr: float,
    params: Seq2SlateParams,
    policy_optimizer_interval: int
) -> Seq2SlateTrainer:
    trainer_cls = Seq2SlateOnPolicyTrainer if params.on_policy else Seq2SlateTrainer
    return trainer_cls(
        reinforce_network=net,
        params=params,
        policy_optimizer=OptimizerRoster(SGD=opt_classes["SGD"](lr=lr)),
        policy_optimizer_interval=policy_optimizer_interval,
    )


def create_seq2slate_transformer(
    state_dim: int,
    num_of_candidates: int,
    candidate_dim: int,
    hidden_size: int,
    output_arch: Seq2SlateOutputArch
) -> Seq2SlateTransformerNetwork:
    return Seq2SlateTransformerNetwork(
        state_dim=state_dim,
        candidate_dim=candidate_dim,
        nlayers=2,
        nheads=2,
        dim_model=hidden_size,
        dim_feedforward=hidden_size,
        max_source_seq_len=num_of_candidates,
        max_target_seq_len=num_of_candidates,
        output_arch=output_arch,
        temperature=0.5
    )


def create_on_policy_batch(
    net: Seq2SlateTransformerNetwork,
    batch_size: int,
    state_dim: int,
    num_of_candidates: int,
    candidate_dim: int,
    rank_seed: int,
    device: torch.device
) -> adt.PreprocessedRankingInput:
    state = torch.randn(batch_size, state_dim).to(device)
    candidates = torch.randn(batch_size, num_of_candidates, candidate_dim).to(device)
    reward = torch.rand(batch_size, 1).to(device)
    batch = adt.PreprocessedRankingInput.from_input(state=state, candidates=candidates, device=device)

    torch.manual_seed(rank_seed)
    ordered_output: adt.RankingOutput = net(
        batch, mode=Seq2SlateMode.RANK_MODE, target_seq_len=num_of_candidates, greedy=False
    )
    ordered_indcs = ordered_output.ordered_target_out_indcs - 2
    ordered_slate_proba = ordered_output.ordered_per_seq_probas
    on_policy_batch = adt.PreprocessedRankingInput.from_input(
        state=state,
        candidates=candidates,
        device=device,
        actions=ordered_indcs,
        logged_propensities=ordered_slate_proba.detach(),
        slate_reward=reward
    )
    return on_policy_batch


def create_off_policy_batch(
    batch_size: int,
    state_dim: int,
    num_of_candidates: int,
    candidate_dim: int,
    device: torch.device
):
    state = torch.randn(batch_size, state_dim)
    candidates = torch.randn(batch_size, num_of_candidates, candidate_dim)
    reward = torch.rand(batch_size, 1)
    actions = torch.stack([torch.randperm(num_of_candidates) for _ in range(batch_size)])
    logged_slate_proba = torch.rand(batch_size, 1) / 1e12
    off_policy_batch = adt.PreprocessedRankingInput.from_input(
        state=state,
        candidates=candidates,
        device=device,
        actions=actions,
        logged_propensities=logged_slate_proba,
        slate_reward=reward
    )
    return off_policy_batch


class TestSeq2SlateTrainer(unittest.TestCase):
    def setUp(self) -> None:
        pl.seed_everything(0)

    def assert_gradients_correct(
        self,
        manual_grad_net: Seq2SlateTransformerNetwork,
        fitted_net: Seq2SlateTransformerNetwork,
        policy_optimizer_interval: int,
        lr: float
    ):
        for (mg_name, mg_weights), (fitted_name, fitted_weights) in zip(
            manual_grad_net.named_parameters(), fitted_net.named_parameters()
        ):
            assert mg_name == fitted_name
            if mg_weights.grad is not None:
                assert torch.allclose(
                    mg_weights - policy_optimizer_interval * lr * mg_weights.grad,
                    fitted_weights,
                    rtol=1e-4,
                    atol=2e-6
                )

    def test_importance_sampling_blur(self):
        importance_sampling = torch.tensor([0.5, 0.3, 3.0, 10.0, 40.0])
        assert torch.all(ips_blur(importance_sampling, None) == importance_sampling)
        assert torch.all(
            ips_blur(importance_sampling, IPSBlur(IPSBlurMethod.AGGRESSIVE, 3.0))
            == torch.tensor([0.5, 0.3, 3.0, 0.0, 0.0])
        )
        assert torch.all(
            ips_blur(importance_sampling, IPSBlur(IPSBlurMethod.UNIVERSAL, 3.0))
            == torch.tensor([0.5, 0.3, 3.0, 3.0, 3.0])
        )

    @parameterized.expand(itertools.product(policy_optimizer_interval_list, output_arch_list))
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_seq2slate_base_trainer_on_policy_gpu(
        self, policy_optimizer_interval: int, output_arch: Seq2SlateOutputArch
    ):
        self._test_seq2slate_base_trainer_on_policy(
            policy_optimizer_interval, output_arch, device=torch.device("gpu")
        )

    @parameterized.expand(itertools.product(policy_optimizer_interval_list, output_arch_list))
    def test_seq2slate_base_trainer_on_policy_cpu(
        self, policy_optimizer_interval: int, output_arch: Seq2SlateOutputArch
    ):
        self._test_seq2slate_base_trainer_on_policy(
            policy_optimizer_interval, output_arch, device=torch.device("cpu")
        )

    @parameterized.expand(itertools.product(policy_optimizer_interval_list, output_arch_list))
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_seq2slate_base_trainer_off_policy_gpu(
        self, policy_optimizer_interval: int, output_arch: Seq2SlateOutputArch
    ):
        self._test_seq2slate_base_trainer_off_policy(
            policy_optimizer_interval, output_arch, device=torch.device("gpu")
        )

    @parameterized.expand(itertools.product(policy_optimizer_interval_list, output_arch_list))
    def test_seq2slate_base_trainer_off_policy_cpu(
        self, policy_optimizer_interval: int, output_arch: Seq2SlateOutputArch
    ):
        self._test_seq2slate_base_trainer_off_policy(
            policy_optimizer_interval, output_arch, device=torch.device("cpu")
        )

    def _test_seq2slate_base_trainer_on_policy(
        self, policy_optimizer_interval: int, output_arch: Seq2SlateOutputArch, device: torch.device
    ):
        batch_size = 32
        state_dim = 2
        num_of_candidates = 15
        candidate_dim = 4
        hidden_size = 16
        lr = 1.0
        on_policy = True
        rank_seed = 111
        seq2slate_params = Seq2SlateParams(on_policy=on_policy)
        seq2slate_net = create_seq2slate_transformer(
            state_dim, num_of_candidates, candidate_dim, hidden_size, output_arch
        ).to(device)
        seq2slate_net_copy1 = deepcopy(seq2slate_net).to(device)
        seq2slate_net_copy2 = deepcopy(seq2slate_net).to(device)
        trainer = create_trainer(seq2slate_net, lr, seq2slate_params, policy_optimizer_interval)
        batch = create_on_policy_batch(
            seq2slate_net,
            batch_size,
            state_dim,
            num_of_candidates,
            candidate_dim,
            rank_seed,
            device
        )
        training_data = DataLoader([batch], collate_fn=lambda x: x[0])
        pl_trainer = pl.Trainer(
            max_epochs=policy_optimizer_interval,
            gpus=None if device == torch.device("cpu") else 1,
            logger=False
        )
        pl_trainer.fit(trainer, training_data)
        seq2slate_net = trainer.reinforce.to(device)

        torch.manual_seed(rank_seed)
        ordered_output: adt.RankingOutput = seq2slate_net_copy1(
            batch, mode=Seq2SlateMode.RANK_MODE, target_seq_len=num_of_candidates, greedy=False
        )
        loss = -torch.mean(torch.log(ordered_output.ordered_per_seq_probas) * batch.slate_reward)
        loss.backward()
        self.assert_gradients_correct(seq2slate_net_copy1, seq2slate_net, policy_optimizer_interval, lr)

        torch.manual_seed(rank_seed)
        ordered_per_seq_probas: torch.Tensor = seq2slate_net_copy2(
            batch, mode=Seq2SlateMode.RANK_MODE, target_seq_len=num_of_candidates, greedy=False
        ).ordered_per_seq_probas
        loss = -torch.mean(ordered_per_seq_probas / ordered_per_seq_probas.detach() * batch.slate_reward)
        loss.backward()
        self.assert_gradients_correct(seq2slate_net_copy2, seq2slate_net, policy_optimizer_interval, lr)

    def _test_seq2slate_base_trainer_off_policy(
        self, policy_optimizer_interval: int, output_arch: Seq2SlateOutputArch, device: torch.device
    ):
        batch_size = 32
        state_dim = 2
        num_of_candidates = 15
        candidate_dim = 4
        hidden_size = 16
        lr = 1.0
        on_policy = False
        seq2slate_params = Seq2SlateParams(on_policy=on_policy)

        seq2slate_net = create_seq2slate_transformer(
            state_dim, num_of_candidates, candidate_dim, hidden_size, output_arch
        ).to(device)
        seq2slate_net_copy1 = deepcopy(seq2slate_net).to(device)
        seq2slate_net_copy2 = deepcopy(seq2slate_net).to(device)
        trainer = create_trainer(seq2slate_net, lr, seq2slate_params, policy_optimizer_interval)
        batch = create_off_policy_batch(batch_size,  state_dim, num_of_candidates, candidate_dim, device)

        training_data = DataLoader([batch], collate_fn=lambda x: x[0])
        pl_trainer = pl.Trainer(
            max_epochs=policy_optimizer_interval,
            gpus=None if device == torch.device("cpu") else 1,
            logger=False
        )
        pl_trainer.fit(trainer, training_data)
        seq2slate_net = trainer.reinforce.to(device)

        ordered_per_seq_log_probas = seq2slate_net_copy1(batch, mode=Seq2SlateMode.PER_SEQ_LOG_PROB_MODE).log_probas

        loss = -torch.mean(
            ordered_per_seq_log_probas * torch.exp(ordered_per_seq_log_probas).detach() /
            batch.target_output_probas * batch.slate_reward
        )
        loss.backward()
        self.assert_gradients_correct(seq2slate_net_copy1, seq2slate_net, policy_optimizer_interval, lr)

        ordered_per_seq_probas = torch.exp(
            seq2slate_net_copy2(batch, mode=Seq2SlateMode.PER_SEQ_LOG_PROB_MODE).log_probas
        )

        loss = -torch.mean(ordered_per_seq_probas / batch.target_output_probas * batch.slate_reward)
        loss.backward()
        self.assert_gradients_correct(seq2slate_net_copy2, seq2slate_net, policy_optimizer_interval, lr)

    @parameterized.expand(itertools.product(blur_method_list, output_arch_list))
    def test_seq2slate_base_trainer_off_policy_with_ips_blur(
        self, blur_method: IPSBlurMethod, output_arch: Seq2SlateOutputArch
    ):
        batch_size = 32
        state_dim = 2
        num_of_candidates = 15
        candidate_dim = 4
        hidden_size = 16
        lr = 1.0
        device = torch.device("cpu")
        policy_optimizer_interval = 1
        seq2slate_params = Seq2SlateParams(
            on_policy=False,
            ips_blur=IPSBlur(blur_method=blur_method, blur_max=3.0)
        )

        seq2slate_net = create_seq2slate_transformer(
            state_dim, num_of_candidates, candidate_dim, hidden_size, output_arch
        )
        seq2slate_net_copy = deepcopy(seq2slate_net)
        trainer = create_trainer(seq2slate_net, lr, seq2slate_params, policy_optimizer_interval)
        batch = create_off_policy_batch(batch_size, state_dim, num_of_candidates, candidate_dim, device)

        training_data = DataLoader([batch], collate_fn=lambda x: x[0])
        pl_trainer = pl.Trainer(max_epochs=policy_optimizer_interval, logger=False)
        pl_trainer.fit(trainer, training_data)

        ordered_per_seq_probas = torch.exp(
            seq2slate_net_copy(batch, mode=Seq2SlateMode.PER_SEQ_LOG_PROB_MODE).log_probas
        )
        logger.info(f"IPS ratio={ordered_per_seq_probas / batch.target_output_probas}")
        loss = -torch.mean(
            ips_blur(
                ordered_per_seq_probas / batch.target_output_probas, seq2slate_params.ips_blur
            ) * batch.slate_reward
        )
        loss.backward()
        self.assert_gradients_correct(seq2slate_net_copy, seq2slate_net, policy_optimizer_interval, lr)

    @parameterized.expand(
        itertools.product(output_arch_list, blur_method_list, blur_max_list, frechet_sort_shape_list)
    )
    def test_compute_importance_sampling(
        self, output_arch: Seq2SlateOutputArch, blur_method: IPSBlurMethod, blur_max: float, shape: float
    ):
        logger.info(f"Output arch: {output_arch}")
        logger.info(f"Blur method: {blur_method}")
        logger.info(f"Blur max: {blur_max}")
        logger.info(f"Frechet shape: {shape}")

        num_of_candidates = 5
        candidate_dim = 2
        state_dim = 1
        hidden_size = 32
        device = torch.device("cpu")
        lr = 0.001
        policy_optimizer_interval = 1

        candidates = torch.randint(5, (num_of_candidates, candidate_dim)).float()
        candidate_scores = torch.sum(candidates, dim=1)

        seq2slate_params = Seq2SlateParams(
            on_policy=False,
            ips_blur=IPSBlur(blur_method=blur_method, blur_max=blur_max)
        )
        seq2slate_net = create_seq2slate_transformer(
            state_dim, num_of_candidates, candidate_dim, hidden_size, output_arch
        )
        trainer = create_trainer(seq2slate_net, lr, seq2slate_params, policy_optimizer_interval)

        all_perms = torch.tensor(list(itertools.permutations(range(num_of_candidates), num_of_candidates)))
        sampler = FrechetSort(shape=shape, topk=num_of_candidates)
        sum_of_logged_propensity = 0
        sum_of_model_propensity = 0
        sum_of_ips_ratio = 0

        for i in range(len(all_perms)):
            simplex_vertex = all_perms[i]
            logged_propensity = torch.exp(sampler.log_proba(candidate_scores, simplex_vertex))
            batch = adt.PreprocessedRankingInput.from_input(
                state=torch.zeros(1, state_dim),
                candidates=candidates.unsqueeze(0),
                device=device,
                actions=simplex_vertex.unsqueeze(0),
                logged_propensities=logged_propensity.reshape(1, 1)
            )
            model_propensities = torch.exp(seq2slate_net(batch, mode=Seq2SlateMode.PER_SEQ_LOG_PROB_MODE).log_probas)
            ips_weights, blured_ips_weights = trainer.importance_sampling(model_propensities, logged_propensity)
            if ips_weights > blured_ips_weights:
                if blur_method == IPSBlurMethod.AGGRESSIVE:
                    npt.assert_allclose(blured_ips_weights.detach().numpy(), 0, rtol=1e-5)
                else:
                    npt.assert_allclose(blured_ips_weights.detach().numpy(), blur_max, rtol=1e-5)

            sum_of_model_propensity += model_propensities
            sum_of_logged_propensity += logged_propensity
            sum_of_ips_ratio += model_propensities / logged_propensity
            logger.info(
                f"shape={shape}, simplex_vertex={simplex_vertex}, logged_propensity={logged_propensity}, "
                f"model_propensity={model_propensities}"
            )

        logger.info(
            f"shape={shape}, sum_of_logged_propensity={sum_of_logged_propensity}, "
            f"sum_of_model_propensity={sum_of_model_propensity}, "
            f"mean sum_of_ips_ratio={sum_of_ips_ratio / len(all_perms)}"
        )
        npt.assert_allclose(sum_of_logged_propensity.detach().numpy(), 1, rtol=1e-5)
        npt.assert_allclose(sum_of_model_propensity.detach().numpy(), 1, rtol=1e-5)

    @parameterized.expand(itertools.product(output_arch_list, frechet_sort_shape_list))
    def test_ips_ratio_mean(self, output_arch: Seq2SlateOutputArch, shape: float):
        logger.info(f"Output arch: {output_arch}")
        logger.info(f"Frechet shape: {shape}")

        num_of_candidates = 5
        candidate_dim = 2
        state_dim = 1
        hidden_size = 8
        device = torch.device("cpu")
        batch_size = 1024
        num_butches = 400
        lr = 0.001
        policy_optimizer_interval = 1

        state = torch.zeros(batch_size, state_dim)

        candidates = torch.randint(5, (batch_size, num_of_candidates, candidate_dim)).float()
        candidates[1:] = candidates[0]
        candidate_scores = torch.sum(candidates, dim=-1)

        seq2slate_params = Seq2SlateParams(on_policy=False)
        seq2slate_net = create_seq2slate_transformer(
            state_dim, num_of_candidates, candidate_dim, hidden_size, output_arch
        )
        trainer = create_trainer(seq2slate_net, lr, seq2slate_params, policy_optimizer_interval)

        sampler = FrechetSort(shape=shape, topk=num_of_candidates)
        sum_of_ips_ratio = 0

        for i in range(num_butches):
            sampling = [sampler(candidate_scores[j: j + 1]) for j in range(batch_size)]
            simplex_vertex = torch.stack(list(map(lambda so: so[0].squeeze(0), sampling)))
            logged_propensity = torch.stack(list(map(lambda so: torch.exp(so[1]), sampling)))
            batch = adt.PreprocessedRankingInput.from_input(
                state=state,
                candidates=candidates,
                device=device,
                actions=simplex_vertex,
                logged_propensities=logged_propensity
            )
            model_propensities = torch.exp(seq2slate_net(batch, mode=Seq2SlateMode.PER_SEQ_LOG_PROB_MODE).log_probas)
            ips_weights, _ = trainer.importance_sampling(model_propensities, logged_propensity)
            sum_of_ips_ratio += torch.mean(ips_weights).detach().numpy()
            mean_of_ips_ratio = sum_of_ips_ratio / (i + 1)
            logger.info(f"{i}-th batch, mean IPS ratio={mean_of_ips_ratio}")

            if i > 100 and np.allclose(mean_of_ips_ratio, 1, atol=0.01):
                return
        raise Exception(f"Mean IPS ratio {mean_of_ips_ratio} doesn't converge to 1")
