import abc
from typing import Dict, Union

import pandas as pd
import pyspark.sql as pss

from athena.core.logger import LoggerMixin
from athena.core.parameters import NormalizationParams
from athena.core.registry import RegistryMeta
from athena.core.singleton import Singleton

DataFrameUnion = Union[pss.DataFrame, pd.DataFrame]


class FAPper(LoggerMixin, metaclass=type("RegSingleton", (RegistryMeta, Singleton), {})):
    @abc.abstractmethod
    def fap(self, *args, **kwargs) -> str:
        pass

    @abc.abstractmethod
    def identify_normalization_params(self, *args, **kwargs) -> Dict[int, NormalizationParams]:
        pass

    @abc.abstractmethod
    def get_max_sequence_len(self, *args, **kwargs) -> int:
        pass

    @abc.abstractmethod
    def get_element_dim(self, *args, **kwargs) -> int:
        pass

    @staticmethod
    @abc.abstractmethod
    def _reward_discount(*args, **kwargs) -> DataFrameUnion:
        pass

    @staticmethod
    @abc.abstractmethod
    def _hash_mdp_id_and_subsample(*args, **kwargs) -> DataFrameUnion:
        pass

    @staticmethod
    @abc.abstractmethod
    def _misc_column_preprocessing(*args, **kwargs) -> DataFrameUnion:
        pass

    @staticmethod
    @abc.abstractmethod
    def _state_and_metrics_sparse2dense(*args, **kwargs) -> DataFrameUnion:
        pass

    @staticmethod
    @abc.abstractmethod
    def _discrete_actions_preprocessing(*args, **kwargs) -> DataFrameUnion:
        pass

    @staticmethod
    @abc.abstractmethod
    def _parametric_actions_preprocessing(*args, **kwargs) -> DataFrameUnion:
        pass

    @abc.abstractmethod
    def _process_extra_columns(self, df: DataFrameUnion, col_name: str) -> DataFrameUnion:
        pass

    @abc.abstractmethod
    def _select_relevant_columns(self, *args, **kwargs) -> DataFrameUnion:
        pass
