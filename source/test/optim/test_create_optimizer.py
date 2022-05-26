import unittest
import sys
sys.path.append('/home/akasyan/reranking/nastenka-solnishko/source/optim')
import torch
from source.optim.uninferrable_optimizers import Adam
from source.optim.uninferrable_schedulers import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    MultiStepLR,
    OneCycleLR,
    StepLR
)

from source.optim.utils import is_torch_lr_scheduler, is_torch_optimizer

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