import abc
from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch

from athena.core.dataclasses import dataclass
from athena.core.dtypes import TableSpec
from athena.core.dtypes.dataset import Dataset
from athena.core.dtypes.options import (ReaderOptions, ResourceOptions,
                                        RLOptions)
from athena.core.dtypes.results import TrainingOutput, TrainingReportRoster
from athena.core.logger import LoggerMixin, ManifoldTensorboardLogger
from athena.core.parameters import NormalizationData
from athena.data.athena_datamodule import AthenaDataModule
from athena.data.data_extractor import DataExtractor
from athena.data.fap.fapper import FAPper
from athena.report.base import ReporterBase
from athena.trainers.athena_lightening import AthenaLightening
from athena.workflow.utils import get_rank, train_eval


@dataclass
class ModelManager(LoggerMixin):
    r"""
    ModelManager manages how to train models.

    To integrate training algorithms into the standard training workflow, you need:

    1. ``build_trainer()``: Builds the :class:`AthenaLightning`.
    2. ``get_data_module()``: Defines how to create data module for this algorithm.
    3. ``build_serving_modules()``: Creates the TorchScript modules for serving.
    4. ``get_reporter()``: Returns the reporter to collect training/evaluation metrics.
    """

    def __post_init_post_parse__(self):
        """
        We use pydantic to parse raw config into typed (dataclass) config.
        This method is called after everything is parsed, so you could
        validate constraints that may not be captured with the type alone.
        """
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
        """Return the data module. If this is not None, then ``run_feature_identification`` &
        ``query_data`` will not be run.

        Args:
            fapper (FAPper): The way how to fetch and process data.
            input_table_spec (Optional[TableSpec], optional): Input table specifications. Defaults to None.
            rl_options (Optional[RLOptions], optional): Options used in policy optimization. Defaults to None.
            setup_data (Optional[Dict[str, bytes]], optional): Serialized data and its metadata. Defaults to None.
            saved_setup_data (Optional[Dict[str, bytes]], optional): Manually preprocessed data. Defaults to None.
            reader_options (Optional[ReaderOptions], optional): Batch reader configuration. Defaults to None.
            resource_options (Optional[ResourceOptions], optional): Resource configuration. Defaults to None.

        Returns:
            Optional[AthenaDataModule]: Custom data module.
        """
        return None

    @abc.abstractmethod
    def build_trainer(
        self,
        use_gpu: bool,
        rl_options: RLOptions,
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
        train_data: Optional[Dataset],
        eval_data: Optional[Dataset],
        test_data: Optional[Dataset],
        data_module: Optional[AthenaDataModule],
        nepochs: int,
        reader_options: ReaderOptions,
        resource_options: ResourceOptions,
        checkpoint_path: Optional[str] = None
    ) -> Tuple[TrainingOutput, pl.Trainer]:
        reporter = self.get_reporter()
        trainer_module.set_reporter(reporter)
        if data_module is None:
            raise ValueError("No datamodule provided")
        trainer = train_eval(
            train_data=train_data,
            eval_data=eval_data,
            test_data=test_data,
            trainer_module=trainer_module,
            data_module=data_module,
            nepochs=nepochs,
            logger_name=str(type(self)),
            reader_options=reader_options,
            checkpoint_path=checkpoint_path,
            resource_options=resource_options
        )

        rank = get_rank()
        if rank == 0:
            trainer_logger: ManifoldTensorboardLogger = trainer.logger
            logger_data = trainer_logger.line_plot_aggregated
            trainer_logger.clear_local_data()
            if reporter is None:
                training_report = None
            else:
                training_report = TrainingReportRoster.make_roster_instance(
                    reporter.training_report()
                )
            return TrainingOutput(training_report=training_report, logger_data=logger_data), trainer
        return TrainingOutput(), trainer

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
    def should_generate_eval_data(self) -> bool:
        pass
