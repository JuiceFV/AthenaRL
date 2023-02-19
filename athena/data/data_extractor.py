from typing import Optional, List, Tuple
from athena.data.fap.fapper import FAPper
from athena.core.dtypes.dataset import Dataset
from athena.core.dtypes.rl.options import RewardOptions


class DataExtractor:
    def __init__(self, fapper: FAPper):
        self.fapper = fapper

    def query_data(
        self,
        table_name: str,
        discrete_actions: bool,
        actions_space: Optional[List[str]] = None,
        sample_range: Optional[Tuple[float, float]] = None,
        reward_options: Optional[RewardOptions] = None,
        extra_columns: List[str] = []
    ) -> Dataset:
        return Dataset(
            self.fapper.fap(
                table_name=table_name,
                discrete_actions=discrete_actions,
                actions_space=actions_space,
                sample_range=sample_range,
                reward_options=reward_options,
                extra_columns=extra_columns
            )
        )
