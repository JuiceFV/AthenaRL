import abc
from typing import Dict, Optional, Tuple
from athena.core.dataclasses import dataclass
from athena.core.dtypes.dataset import Dataset
from athena.core.dtypes.options import MLOptionsRoster, ReaderOptions, ResourceOptions
from athena.core.logger import LoggerMixin
from athena.core.dtypes import TableSpec
from athena.core.parameters import NormalizationData
from athena.data.data_extractor import DataExtractor
from athena.data.manual_datamodule import ManualDataModule
from athena.report.base import ReporterBase
from athena.trainers.athena_lightening import AthenaLightening


@dataclass
class ModelManager(LoggerMixin):
    def __post_init_post_parse__(self):
        pass

    def get_data_module(
        self,
        *,
        input_table_spec: Optional[TableSpec] = None,
        ml_options: Optional[MLOptionsRoster] = None,
        setup_data: Optional[Dict[str, bytes]] = None,
        saved_setup_data: Optional[Dict[str, bytes]] = None,
        reader_options: Optional[ReaderOptions] = None,
        resource_options: Optional[ResourceOptions] = None
    ) -> Optional[ManualDataModule]:
        return None

    @abc.abstractmethod
    def build_trainer(
        self,
        normalization_dict: Dict[str, NormalizationData],
        use_gpu: bool,
        ml_options: Optional[MLOptionsRoster] = None
    ) -> AthenaLightening:
        pass

    @abc.abstractmethod
    def get_reporter(self) -> ReporterBase:
        pass

    @abc.abstractmethod
    def run_feature_identification(self, input_table_spec: TableSpec) -> Dict[str, NormalizationData]:
        pass

    @abc.abstractmethod
    def query_data(
        self,
        input_table_spec: TableSpec,
        sample_range: Optional[Tuple[float, float]],
        ml_options: MLOptionsRoster,
        data_extractor: DataExtractor
    ) -> Dataset:
        pass

    @property
    @abc.abstractmethod
    def should_generate_eval_dataset(self) -> bool:
        pass
