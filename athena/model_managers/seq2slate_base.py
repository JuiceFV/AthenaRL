from typing import Optional

from athena.core.dataclasses import dataclass
from athena.data.base import DataModule
from athena.core.dtypes import TableSpec
from athena.model_managers.manager import ModelManager


@dataclass
class Seq2SlateBase(ModelManager):
    def __post_init_post_parse__(self):
        super().__post_init_post_parse__()

    def get_data_module(self, input_table_spec: TableSpec) -> Optional[DataModule]:
        return super().get_data_module(input_table_spec)
