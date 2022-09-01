from athena.data.fap.fapper import FAPper
from athena.core.dtypes.dataset import Dataset


class DataExtractor:
    def __init__(self, fapper: FAPper):
        self.fapper = fapper

    def query_data(self, *args, **kwargs) -> Dataset:
        return Dataset(self.fapper.fap(*args, **kwargs))
