import unittest
import torch
from athena.optim.uninferrable_optimizers import Adam
from athena.optim.uninferrable_schedulers import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    MultiStepLR,
    OneCycleLR,
    StepLR
)

from athena.optim.utils import is_torch_lr_scheduler, is_torch_optimizer


class TestCreateOptimizer(unittest.TestCase):
    def setUp(self) -> None:
        self.model = torch.nn.Linear(3, 4)

    def _verify_optimizer(self, optimizer_scheduler):
        self.assertTrue(is_torch_optimizer(type(optimizer_scheduler["optimizer"])))
        self.assertTrue(is_torch_lr_scheduler(type(optimizer_scheduler["lr_scheduler"])))

    def test_create_optimizer_with_step_lr_scheduler(self):
        self._verify_optimizer(
            Adam(
                lr=0.001, lr_schedulers=[StepLR(gamma=0.1, step_size=0.01)]
            ).create_optimizer_scheduler(self.model.parameters())
        )

    def test_create_optimizer_with_multistep_lr_scheduler(self):
        self._verify_optimizer(
            Adam(
                lr=0.001,
                lr_schedulers=[MultiStepLR(gamma=0.2, milestones=[1000, 2000])],
            ).create_optimizer_scheduler(self.model.parameters())
        )

    def test_create_optimizer_with_exponential_lr_scheduler(self):
        self._verify_optimizer(
            Adam(
                lr=0.001, lr_schedulers=[ExponentialLR(gamma=0.9)]
            ).create_optimizer_scheduler(self.model.parameters())
        )

    def test_create_optimizer_with_cosine_annealing_lr_scheduler(self):
        self._verify_optimizer(
            Adam(
                lr=0.001, lr_schedulers=[CosineAnnealingLR(T_max=1)]
            ).create_optimizer_scheduler(self.model.parameters())
        )

    def test_create_optimizer_with_one_cycle_lr_scheduler(self):
        self._verify_optimizer(
            Adam(
                lr=0.001,
                lr_schedulers=[
                    OneCycleLR(max_lr=0.1, base_momentum=0.8, total_steps=1000)
                ],
            ).create_optimizer_scheduler(self.model.parameters())
        )

    def test_create_optimizer_with_cosine_annealing_warm_restarts_lr_scheduler(self):
        self._verify_optimizer(
            Adam(
                lr=0.001, lr_schedulers=[CosineAnnealingWarmRestarts(T_0=1)]
            ).create_optimizer_scheduler(self.model.parameters())
        )
