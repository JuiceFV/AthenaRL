import abc
import pytorch_lightning as pl
from typing import Dict, List, Optional

from nastenka_solnishko.core.dtypes import (
    TableSpec
)


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_transforms=None,
        val_transforms=None,
        test_transforms=None,
        dims=None
    ):
        super().__init__(
            train_transforms,
            val_transforms,
            test_transforms,
            dims
        )

    @abc.abstractclassmethod
    def get_normalization_dict(
        self,
        keys: Optional[List[str]] = None
    ) -> Dict[str, object]:  # TODO: return type should be Dict[str, Normalization]
        pass

    @abc.abstractproperty
    def train_data(self):
        pass

    @abc.abstractproperty
    def validation_data(self):
        pass

    @abc.abstractproperty
    def test_data(self):
        pass
