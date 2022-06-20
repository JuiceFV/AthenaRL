import os
import pandas as pd
from nastenka_solnishko.data.loaders.base import DataLoader
from nastenka_solnishko.core.dtypes import Dataset, TableSpec


class LFSDataLoader(DataLoader):
    def query_data(
        self,
        input_table_spec: TableSpec
    ) -> Dataset:
        pass
