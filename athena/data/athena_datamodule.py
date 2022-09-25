import abc
import pytorch_lightning as pl
from typing import Dict, List, Optional, Tuple, Union
from athena.core.dtypes.dataset import Dataset, TableSpec
from athena.core.dtypes.rl.options import RLOptions

from athena.core.parameters import NormalizationData
from athena.data.data_extractor import DataExtractor
from athena.data.fap.fapper import FAPper
from athena.preprocessing.batch_preprocessor import BatchPreprocessor
from athena.preprocessing.transforms import Transformation, Compose


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
    def should_generate_eval_dataset(self) -> bool:
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
