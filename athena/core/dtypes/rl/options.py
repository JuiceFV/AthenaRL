from typing import Optional, Dict
from athena.core.dataclasses import dataclass


@dataclass(frozen=True)
class RewardOptions:
    custom_reward_expression: Optional[str] = None
    metric_reward_values: Optional[Dict[str, float]] = None


@dataclass
class RLOptions:
    reward_options: Optional[RewardOptions] = None
