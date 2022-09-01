from typing import Optional, Dict
from athena.core.dataclasses import dataclass


@dataclass(frozen=True)
class RewardOptions:
    reward_col_name: str = "reward"
    metrics_col_name: str = "metrics"
    custom_reward: Optional[str] = None
    metric_reward_values: Optional[Dict[str, float]] = None
    gamma: Optional[float] = None

    def __post_init_post_parse__(self):
        if self.custom_reward is not None and self.gamma is not None:
            raise RuntimeError("The only option (either custom_reward or gamma) should be defined")


@dataclass
class RLOptions:
    reward_options: Optional[RewardOptions] = None
