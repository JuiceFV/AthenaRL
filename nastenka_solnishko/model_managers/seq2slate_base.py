from typing import Optional

from nastenka_solnishko.core.dataclasses import dataclass
from nastenka_solnishko.data.base import DataModule
from nastenka_solnishko.core.dtypes import TableSpec
from nastenka_solnishko.model_managers.manager import ModelManager


@dataclass
class Seq2SlateBase(ModelManager):
    def __post_init_post_parse__(self):
        super().__post_init_post_parse__()

    def get_data_module(self, input_table_spec: TableSpec) -> Optional[DataModule]:
        return super().get_data_module(input_table_spec)
