from typing import Optional

from source.core.dataclasses import dataclass
from source.data.base import DataModule
from source.core.dtypes import TableSpec
from source.model_managers.manager import ModelManager


@dataclass
class Seq2SlateBase(ModelManager):
    def __post_init_post_parse__(self):
        super().__post_init_post_parse__()

    def get_data_module(self, input_table_spec: TableSpec) -> Optional[DataModule]:
        return super().get_data_module(input_table_spec)
