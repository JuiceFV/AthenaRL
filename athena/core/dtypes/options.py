from typing import Optional
from athena.core.dataclasses import dataclass
from athena.core.dtypes.rl import RLOptions
from athena.core.dtypes.preprocessing import PreprocessingOptions


@dataclass
class ReaderOptions:
    minibatch_size: int = 1024
    petasorm_reader_pool_type: str = "thread"


@dataclass
class ResourceOptions:
    gpu: int = 0

    @property
    def use_gpu(self) -> bool:
        return self.gpu > 0


@dataclass
class AthenaOptions:
    rl_options: Optional[RLOptions] = None
    reader_options: Optional[ReaderOptions] = None
    resource_options: Optional[ResourceOptions] = None
    preprocessing_options: Optional[PreprocessingOptions] = None
