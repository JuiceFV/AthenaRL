import os
import pandas as pd
from athena.data.loaders.base import DataLoader
from athena.core.dtypes import Dataset, TableSpec


class LFSDataLoader(DataLoader):
    def query_data(
        self,
        input_table_spec: TableSpec
    ) -> Dataset:
        pass
