from dataclasses import dataclass
from enum import Enum

from athena.core.base_dclass import BaseDataClass


class IPSBlurMethod(Enum):
    UNIVERSAL = "universal"
    CR = "cr"


@dataclass(frozen=True)
class IPSBlur(BaseDataClass):
    blur_method: IPSBlurMethod
    blur_max: float
