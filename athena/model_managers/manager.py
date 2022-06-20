from typing import Optional
from athena.core.dataclasses import dataclass
from athena.core.logger import LoggerMixin
from athena.core.dtypes import TableSpec
from athena.data.base import DataModule


@dataclass
class ModelManager(LoggerMixin):
    def __post_init_post_parse__(self):
        pass

    def get_data_module(
        self,
        input_table_spec: TableSpec
    ) -> Optional[DataModule]:
        return None
