from dataclasses import dataclass
from enum import Enum

from athena.core.base_dclass import BaseDataClass
from athena.core.enum_meta import AthenaEnumMeta


class IPSBlurMethod(Enum, metaclass=AthenaEnumMeta):
    UNIVERSAL = "universal"
    AGGRESSIVE = "aggressive"


@dataclass(frozen=True)
class IPSBlur(BaseDataClass):
    blur_method: IPSBlurMethod
    blur_max: float
