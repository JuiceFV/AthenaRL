from typing import Optional, Dict

from athena.core.dataclasses import dataclass, field
from athena.core.parameters import NormalizationData
from athena.data.athena_datamodule import AthenaDataModule
from athena.core.dtypes import TableSpec
from athena.core.dtypes.options import PreprocessingOptions
from athena.data.manual_datamodule import ManualDataModule
from athena.model_managers.manager import ModelManager


@dataclass
class Seq2SlateBase(ModelManager):

    def __post_init_post_parse__(self):
        super().__post_init_post_parse__()

    def get_data_module(self, input_table_spec: TableSpec) -> Optional[AthenaDataModule]:
        return super().get_data_module(input_table_spec)


class Seq2SlateDataModule(ManualDataModule):  # Inherits ManualDataModule
    @property
    def should_generate_eval_dataset(self) -> bool:
        return self.model_manager.eval_parameters.cpe_evaluation

    def run_feature_identification(self, input_table_spec: TableSpec) -> Dict[str, NormalizationData]:
        pass