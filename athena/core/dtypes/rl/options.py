from typing import Optional, Dict
from athena.core.dataclasses import dataclass


@dataclass(frozen=True)
class RewardOptions:
    r"""
    Configurable reward options.

    .. important::
        ``custom_reward`` and ``gamma`` are mutually exclusive.
    """
    #: Column name of the short-term reward.
    reward_col_name: str = "reward"
    #: Column name of the long-term reward.
    metrics_col_name: str = "metrics"
    #: Custom expression to calculate a reward. For example :math:`r_i^3 + 10`
    #: is implemented as following :code:`custom_reward="POWER(reward, 3) + 10"`.
    custom_reward: Optional[str] = None
    #: Custom metric to reward is used to optimize multiple metrics simultaneously.
    metric_reward_values: Optional[Dict[str, float]] = None
    #: Gamma value s.t. :math:`0 \leq \gamma \leq 1` is used in gamma discounting.
    gamma: Optional[float] = None

    def __post_init_post_parse__(self):
        if self.custom_reward is not None and self.gamma is not None:
            raise RuntimeError("The only option (either custom_reward or gamma) should be defined")


@dataclass
class RLOptions:
    r"""
    Reinforcement Learning Options.
    """
    #: Custom :class:`RewardOptions`.
    reward_options: Optional[RewardOptions] = None
