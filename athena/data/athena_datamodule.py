import abc
import pytorch_lightning as pl
from typing import Dict, List, Optional

from athena.core.parameters import NormalizationData


class AthenaDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def get_normalization_dict(self, keys: Optional[List[str]] = None) -> Dict[str, NormalizationData]:
        pass

    @property
    @abc.abstractmethod
    def train_data(self):
        pass

    @property
    @abc.abstractmethod
    def eval_data(self):
        pass

    @property
    @abc.abstractmethod
    def test_data(self):
        pass
