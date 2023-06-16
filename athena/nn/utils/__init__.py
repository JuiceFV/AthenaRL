import copy
import torch.nn as nn
from athena.nn.utils import transformer  # noqa


def clones(module: nn.Module, ncopies: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(ncopies)])
