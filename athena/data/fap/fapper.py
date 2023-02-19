import abc
from typing import Dict, Union, Optional, List, Tuple

import pandas as pd
import pyspark.sql as pss

from athena.core.logger import LoggerMixin
from athena.core.parameters import NormalizationParams
from athena.core.registry import RegistryMeta
from athena.core.singleton import Singleton
from athena.core.dtypes.rl.options import RewardOptions
from athena.core.dtypes.preprocessing.options import PreprocessingOptions

DataFrameUnion = Union[pss.DataFrame, pd.DataFrame]


class FAPper(LoggerMixin, metaclass=type("RegSingleton", (RegistryMeta, Singleton), {})):
    @abc.abstractmethod
    def fap(
        self,
        table_name: str,
        discrete_actions: bool,
        actions_space: Optional[List[str]] = None,
        sample_range: Optional[Tuple[float, float]] = None,
        reward_options: Optional[RewardOptions] = None,
        extra_columns: List[str] = []
    ) -> str:
        pass

    @abc.abstractmethod
    def identify_normalization_params(
        self,
        table_name: str,
        col_name: str,
        preprocessing_options: PreprocessingOptions,
        seed: Optional[int] = None
    ) -> Dict[int, NormalizationParams]:
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
    def _process_extra_columns(self, *args, **kwargs) -> DataFrameUnion:
        pass

    @abc.abstractmethod
    def _select_relevant_columns(self, *args, **kwargs) -> DataFrameUnion:
        pass
