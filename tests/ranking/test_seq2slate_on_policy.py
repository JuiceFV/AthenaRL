import itertools
import logging
import unittest
from collections import defaultdict

import athena.core.dtypes as adt
import numpy as np
import pytest
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from athena.core.dtypes import Seq2SlateOutputArch
from athena.core.dtypes.ranking.seq2slate import Seq2SlateMode
from athena.nn.utils.transformer import DECODER_START_SYMBOL
from athena.nn.functional import prod_probas
from athena.nn.utils.transformer import decoder_mask, subsequent_mask, mask_by_index
from parameterized import parameterized
from tests.ranking.utils import (ON_POLICY, create_batch, create_seq2slate_net,
                                 per_item_to_per_seq_log_probas,
                                 rank_on_policy, run_seq2slate_tsp)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


output_arch_list = [Seq2SlateOutputArch.FRECHET_SORT, Seq2SlateOutputArch.AUTOREGRESSIVE]
temperature_list = [1.0, 2.0]


class TestSeq2SlateOnPolicy(unittest.TestCase):
    def setUp(self):
        pl.seed_everything(0)

    def test_pytorch_decoder_mask(self):
        batch_size = 3
        source_seq_len = 4
        nheads = 2

        memory = torch.randn(batch_size, source_seq_len, nheads)
        target_input_indcs = torch.tensor([[1, 2, 3], [1, 4, 2], [1, 5, 4]]).long()
        target2target_mask, target2source_mask = decoder_mask(memory, target_input_indcs, nheads)

        expected_target2target_mask = (
            torch.tensor(
                [
                    [False, True, True],
                    [False, False, True],
                    [False, False, False]
                ]
            ).unsqueeze(0).repeat(batch_size * nheads, 1, 1)
        )
        expected_target2source_mask = torch.tensor(
            [
                [
                    [False, False, False, False],
                    [True, False, False, False],
                    [True, True, False, False]
                ],
                [
                    [False, False, False, False],
                    [False, False, True, False],
                    [True, False, True, False]
                ],
                [
                    [False, False, False, False],
                    [False, False, False, True],
                    [False, False, True, True]
                ]
            ]
        ).repeat_interleave(nheads, dim=0)
        assert torch.all(target2target_mask == expected_target2target_mask)
        assert torch.all(target2source_mask == expected_target2source_mask)

    def test_per_item_or_per_seq_log_probas(self):
        batch_size = 1
        seq_len = 3
        num_of_candidates = seq_len + 2

        target_output_indcs = torch.tensor([[0, 2, 1]]) + 2
        per_item_log_probas = torch.randn(batch_size, seq_len, num_of_candidates)
        per_item_log_probas[0, :, :2] = float("-inf")
        per_item_log_probas[0, 1, 2] = float("-inf")
        per_item_log_probas[0, 2, 2] = float("-inf")
        per_item_log_probas[0, 2, 4] = float("-inf")
        per_item_log_probas = F.log_softmax(per_item_log_probas, dim=2)
        per_item_probas = torch.exp(per_item_log_probas)

        expected_per_seq_probas = per_item_probas[0, 0, 2] * per_item_probas[0, 1, 4] * per_item_probas[0, 2, 3]
        computed_per_seq_probas = prod_probas(per_item_probas, target_output_indcs)

        np.testing.assert_allclose(expected_per_seq_probas, computed_per_seq_probas, atol=1e-6, rtol=1e-8)

    def test_subsequent_mask(self):
        expected_mask = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]])
        mask = subsequent_mask(3, torch.device("cpu"))
        assert torch.all(torch.eq(mask, expected_mask))

    def test_mask_logits_by_index(self):
        logits = torch.tensor(
            [
                [
                    [1.0, 2.0, 3.0, 4.0, 5.0],
                    [2.0, 3.0, 4.0, 5.0, 6.0],
                    [3.0, 4.0, 5.0, 6.0, 7.0]
                ],
                [
                    [5.0, 4.0, 3.0, 2.0, 1.0],
                    [6.0, 5.0, 4.0, 3.0, 2.0],
                    [7.0, 6.0, 5.0, 4.0, 3.0],
                ]
            ]
        )
        target_input_indcs = torch.tensor([[DECODER_START_SYMBOL, 2, 3], [DECODER_START_SYMBOL, 4, 3]])
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[:, :, : 2] = 1  # TODO: remove after fix mask_by_index
        masked_logits = logits.masked_fill(mask_by_index(mask, target_input_indcs), float("-inf"))
        expected_logits = torch.tensor(
            [
                [
                    [float("-inf"), float("-inf"), 3.0, 4.0, 5.0],
                    [float("-inf"), float("-inf"), float("-inf"), 5.0, 6.0],
                    [float("-inf"), float("-inf"), float("-inf"), float("-inf"), 7.0],
                ],
                [
                    [float("-inf"), float("-inf"), 3.0, 2.0, 1.0],
                    [float("-inf"), float("-inf"), 4.0, 3.0, float("-inf")],
                    [float("-inf"), float("-inf"), 5.0, float("-inf"), float("-inf")],
                ]
            ]
        )
        assert torch.all(torch.eq(masked_logits, expected_logits))

    @parameterized.expand(itertools.product(output_arch_list, temperature_list))
    @torch.no_grad()
    def test_seq2slate_transformer_propensity_computation(self, output_arch: Seq2SlateOutputArch, temperature: float):
        num_of_candidates = 4
        candidate_dim = 2
        hidden_size = 32
        all_perms = torch.tensor(list(itertools.permutations(torch.arange(num_of_candidates), num_of_candidates)))
        batch_size = len(all_perms)
        device = torch.device("cpu")

        seq2slate_net = create_seq2slate_net(
            num_of_candidates,
            candidate_dim,
            hidden_size,
            output_arch,
            temperature,
            device
        )
        batch = create_batch(
            batch_size,
            num_of_candidates,
            candidate_dim,
            device,
            ON_POLICY,
            diverse_input=False
        )
        batch = adt.PreprocessedRankingInput.from_input(
            state=batch.state.dense_features,
            candidates=batch.source_seq.dense_features,
            device=device,
            actions=all_perms
        )
        per_item_log_prob: torch.Tensor = seq2slate_net(
            batch, mode=Seq2SlateMode.PER_ITEM_LOG_PROB_DIST_MODE
        ).log_probas
        per_seq_log_prob: torch.Tensor = seq2slate_net(
            batch, mode=Seq2SlateMode.PER_SEQ_LOG_PROB_MODE
        ).log_probas
        per_seq_log_prob_computed = per_item_to_per_seq_log_probas(per_item_log_prob, all_perms + 2)

        np.testing.assert_allclose(per_seq_log_prob, per_seq_log_prob_computed, atol=1e-7)
        np.testing.assert_allclose(torch.sum(torch.exp(per_seq_log_prob)), 1.0, atol=1e-7)

    @parameterized.expand(itertools.product(output_arch_list, temperature_list))
    def test_seq2slate_transformer_onpolicy_basic_logic(self, output_arch: Seq2SlateOutputArch, temperature: float):
        device = torch.device("cpu")
        num_of_candidates = 4
        candidate_dim = 2
        batch_size = 4096
        hidden_size = 32
        seq2slate_net = create_seq2slate_net(
            num_of_candidates,
            candidate_dim,
            hidden_size,
            output_arch,
            temperature,
            device
        )
        batch = create_batch(
            batch_size,
            num_of_candidates,
            candidate_dim,
            device,
            ON_POLICY,
            diverse_input=False
        )

        actions2propensity = {}
        actions_count = defaultdict(int)
        total_count = 0
        for i in range(50):
            model_propensity, model_actions = rank_on_policy(seq2slate_net, batch, num_of_candidates, False)
            for propensity, actions in zip(model_propensity, model_actions):
                actions_key = ",".join(map(str, actions.numpy().tolist()))

                if actions2propensity.get(actions_key) is None:
                    actions2propensity[actions_key] = float(propensity)
                else:
                    np.testing.assert_allclose(
                        actions2propensity[actions_key],
                        float(propensity),
                        atol=1e-7,
                        rtol=1e-7
                    )

                actions_count[actions_key] += 1
                total_count += 1
            logger.info(f"Finish {i} round, {total_count} data counts")

        for actions_key, count in actions_count.items():
            empirical_propensity = count / total_count
            computed_propensity = actions2propensity[actions_key]
            logger.info(
                f"actions={actions_key}, empirical propensity={empirical_propensity} "
                f"computed propensity={computed_propensity}"
            )
            np.testing.assert_allclose(computed_propensity, empirical_propensity, atol=0.01, rtol=0.0)

    def test_seq2slate_transformer_on_policy_simple_tsp(self):
        device = torch.device("cpu")
        batch_size = 4096
        epochs = 1
        num_batches = 50
        expect_reward_threshold = 1.12
        hidden_size = 32
        num_of_candidates = 6
        diverse_input = False
        lr = 0.001
        learning_method = ON_POLICY
        policy_optimizer_interval = 1
        run_seq2slate_tsp(
            batch_size,
            epochs,
            num_of_candidates,
            num_batches,
            hidden_size,
            diverse_input,
            lr,
            expect_reward_threshold,
            learning_method,
            policy_optimizer_interval,
            device
        )

    @pytest.mark.seq2slate_long
    @unittest.skipIf(not torch.cuda.is_available(), "Too long test w/o CUDA")
    def test_seq2slate_transformer_on_policy_hard_tsp(self):
        device = torch.device("cuda")
        batch_size = 4096
        epochs = 3
        num_batches = 300
        expect_reward_threshold = 1.02
        hidden_size = 32
        num_of_candidates = 6
        diverse_input = True
        lr = 0.001
        learning_method = ON_POLICY
        policy_optimizer_interval = 1
        run_seq2slate_tsp(
            batch_size,
            epochs,
            num_of_candidates,
            num_batches,
            hidden_size,
            diverse_input,
            lr,
            expect_reward_threshold,
            learning_method,
            policy_optimizer_interval,
            device
        )
