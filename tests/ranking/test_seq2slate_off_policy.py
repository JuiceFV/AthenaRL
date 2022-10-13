import logging
import unittest

import pytest
import pytorch_lightning as pl
import torch
from tests.ranking.utils import OFF_POLICY, run_seq2slate_tsp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class TestSeq2SlateOffPolicy(unittest.TestCase):
    def setUp(self) -> None:
        pl.seed_everything(0)

    def test_seq2slate_transformer_on_policy_simple_tsp(self):
        device = torch.device("cpu")
        batch_size = 4096
        epochs = 1
        num_batches = 100
        expect_reward_threshold = 1.02
        hidden_size = 32
        num_of_candidates = 6
        diverse_input = False
        lr = 0.001
        learning_method = OFF_POLICY
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
        num_of_candidates = 4
        diverse_input = True
        lr = 0.001
        learning_method = OFF_POLICY
        policy_optimizer_interval = 20
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
