from typing import Optional
from nastenka_solnishko.core.dataclasses import dataclass
from nastenka_solnishko.core.logger import LoggerMixin
from nastenka_solnishko.core.dtypes import TableSpec
from nastenka_solnishko.data.base import DataModule


@dataclass
class ModelManager(LoggerMixin):
    def __post_init_post_parse__(self):
        pass

    def get_data_module(
        self,
        input_table_spec: TableSpec
    ) -> Optional[DataModule]:
        return None
