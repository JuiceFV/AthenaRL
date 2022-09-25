import abc
import torch
import pytorch_lightning as pl
from typing import Dict, List, Optional, Tuple
from athena.core.dataclasses import dataclass
from athena.core.dtypes.dataset import Dataset
from athena.core.dtypes.options import RLOptions, ReaderOptions, ResourceOptions
from athena.core.logger import LoggerMixin
from athena.core.dtypes import TableSpec
from athena.core.dtypes.results import TrainingOutput
from athena.core.parameters import NormalizationData
from athena.data.athena_datamodule import AthenaDataModule
from athena.data.data_extractor import DataExtractor
from athena.data.fap.fapper import FAPper
from athena.report.base import ReporterBase
from athena.trainers.athena_lightening import AthenaLightening


@dataclass
class ModelManager(LoggerMixin):
    def __post_init_post_parse__(self):
        pass

    def get_data_module(
        self,
        fapper: FAPper,
        *,
        input_table_spec: Optional[TableSpec] = None,
        rl_options: Optional[RLOptions] = None,
        setup_data: Optional[Dict[str, bytes]] = None,
        saved_setup_data: Optional[Dict[str, bytes]] = None,
        reader_options: Optional[ReaderOptions] = None,
        resource_options: Optional[ResourceOptions] = None
    ) -> Optional[AthenaDataModule]:
        return None

    @abc.abstractmethod
    def build_trainer(
        self,
        normalization_dict: Dict[str, NormalizationData],
    ) -> AthenaLightening:
        pass

    @abc.abstractmethod
    def get_reporter(self) -> ReporterBase:
        pass

    @abc.abstractmethod
    def run_feature_identification(self, input_table_spec: TableSpec, fapper: FAPper) -> Dict[str, NormalizationData]:
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

    def train(
        self,
        trainer_module: AthenaLightening,
        train_dataset: Optional[Dataset],
        eval_dataset: Optional[Dataset],
        test_dataset: Optional[Dataset],
        data_module: Optional[AthenaDataModule],
        num_epochs: int,
        reader_options: ReaderOptions,
        resource_options: ResourceOptions,
        checkpoint_path: Optional[str] = None
    ) -> Tuple[TrainingOutput, pl.Trainer]:
        reporter = self.get_reporter()
        trainer_module.set_reporter(reporter)
        if data_module is None:
            raise ValueError("No datamodule provided")

    def build_serving_modules(
        self,
        trainer_module: AthenaLightening,
        normalization_dict: Dict[str, NormalizationData]
    ) -> Dict[str, torch.nn.Module]:
        return {"default_model": self.build_serving_module(trainer_module, normalization_dict)}

    def build_serving_module(
        self,
        trainer_module: AthenaLightening,
        normalization_dict: Dict[str, NormalizationData]
    ) -> torch.nn.Module:
        raise NotImplementedError

    def serving_module_names(self) -> List[str]:
        return ["default_model"]

    @property
    @abc.abstractmethod
    def should_generate_eval_dataset(self) -> bool:
        pass
