from typing import Optional

from athena.core.dataclasses import dataclass
from athena.core.dtypes.preprocessing import PreprocessingOptions
from athena.core.dtypes.rl import RLOptions


@dataclass
class ReaderOptions:
    r"""
    Petastorm batch reader options.
    """
    #: Batch size.
    minibatch_size: int = 1024
    #: A string denoting the reader pool type. Should be one of ``['thread', 'process', 'dummy']``
    petasorm_reader_pool_type: str = "thread"


@dataclass
class ResourceOptions:
    r"""
    Resource specification used for learining.
    """
    #: How many GPU use. If no CUDA available set it to 0.
    gpu: int = 0

    @property
    def use_gpu(self) -> bool:
        r"""
        Whether use GPU or not.

        Returns:
            bool: Whether use GPU or not.
        """
        return self.gpu > 0


@dataclass
class AthenaOptions:
    r"""
    All options are used in workflow.
    """
    #: Reinforcement Learning options.
    rl_options: Optional[RLOptions] = None
    #: Petastorm batch reader options.
    reader_options: Optional[ReaderOptions] = None
    #: Resources are used during DPP and learning process.
    resource_options: Optional[ResourceOptions] = None
    #: Options are used for data preprocessing.
    preprocessing_options: Optional[PreprocessingOptions] = None
