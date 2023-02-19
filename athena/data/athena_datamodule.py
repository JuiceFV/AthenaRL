import abc
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union

import pandas as pd
import pytorch_lightning as pl
import torch
from petastorm.pytorch import DataLoader, decimal_friendly_collate

from athena.core.dtypes.base import TensorDataClass
from athena.core.dtypes.dataset import Dataset, TableSpec
from athena.core.dtypes.rl.options import RLOptions
from athena.core.parameters import NormalizationData
from athena.data.data_extractor import DataExtractor
from athena.data.fap.fapper import FAPper
from athena.preprocessing.batch_preprocessor import BatchPreprocessor
from athena.preprocessing.transforms import Compose, Transformation

DATA_ITER_STEP = Generator[TensorDataClass, None, None]


def closing_iter(dataloader: DataLoader) -> DATA_ITER_STEP:
    yield from dataloader
    dataloader.__exit__(None, None, None)


def collate_and_preprocess(
    batch_preprocessor: BatchPreprocessor, use_gpu: bool
) -> Callable[[List[Dict]], torch.Tensor]:
    def collate_fn(batch_list: List[Dict]) -> torch.Tensor:
        batch = decimal_friendly_collate(batch_list)
        preprocessed_batch: torch.Tensor = batch_preprocessor(batch)
        if use_gpu:
            preprocessed_batch = preprocessed_batch.cuda()
        return preprocessed_batch
    return collate_fn


def arbitrary_transform(
    transformation: Optional[Union[Compose, Transformation]]
) -> Optional[Callable[[pd.DataFrame], pd.DataFrame]]:
    def transfrom_fn(table: pd.DataFrame) -> pd.DataFrame:
        return transformation(table)
    return transfrom_fn if transformation is not None else None


class AthenaDataModule(pl.LightningDataModule):
    def __init__(self, fapper: FAPper):
        super().__init__()
        self.fapper = fapper

    @abc.abstractmethod
    def get_normalization_dict(self, keys: Optional[List[str]] = None) -> Dict[str, NormalizationData]:
        pass

    @abc.abstractmethod
    def prepare_data(self) -> Optional[Dict[str, bytes]]:
        pass

    @abc.abstractmethod
    def setup(self) -> None:
        pass

    @abc.abstractmethod
    def run_feature_identification(self, input_table_spec: TableSpec) -> Dict[str, NormalizationData]:
        pass

    @property
    @abc.abstractmethod
    def should_generate_eval_data(self) -> bool:
        pass

    @abc.abstractmethod
    def query_data(
        self,
        input_table_spec: TableSpec,
        sample_range: Optional[Tuple[float, float]],
        rl_options: RLOptions,
        data_extractor: DataExtractor
    ) -> Dataset:
        pass

    @abc.abstractmethod
    def build_batch_preprocessor(self) -> BatchPreprocessor:
        pass

    @abc.abstractmethod
    def build_transformation(self) -> Union[Compose, Transformation]:
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
