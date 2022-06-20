from athena.core.dtypes import Dataset, TableSpec


class DataLoader:
    def query_data(
        self,
        input_table_spec: TableSpec
    ) -> Dataset:
        raise NotImplementedError()
