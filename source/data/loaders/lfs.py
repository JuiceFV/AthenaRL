import os
import pandas as pd
from source.data.loaders.base import DataLoader
from source.core.dtypes import Dataset, TableSpec


class LFSDataLoader(DataLoader):
    def query_data(
        self,
        input_table_spec: TableSpec
    ) -> Dataset:
        pass
