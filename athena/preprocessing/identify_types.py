from typing import Any, Iterable, Union
import numpy as np

from athena.core.dtypes import Ftype
from athena.preprocessing import DEFAULT_MAX_UNIQUE_ENUM


def _is_probability(fvalues: Iterable[float]) -> bool:
    return np.all(0 <= fvalues) and np.all(fvalues <= 1)


def _is_binary(fvalues: Iterable[Union[int, bool]]) -> bool:
    return np.all(np.logical_or(fvalues == 0, fvalues == 1)) or np.min(fvalues) == np.max(fvalues)


def _is_continuous(fvalues: Any) -> bool:
    return True


def _is_enum(fvalues: Iterable[Union[int, float]],  enum_threshold: int) -> bool:
    are_all_ints = np.vectorize(lambda value: float(value).is_integer())
    return np.min(fvalues) >= 0 and len(np.unique(fvalues)) <= enum_threshold and np.all(are_all_ints(fvalues))


def identify_type(fvalues: Iterable[Union[int, float]], enum_threshold: int = DEFAULT_MAX_UNIQUE_ENUM) -> Ftype:
    if _is_binary(fvalues):
        return Ftype.BINARY
    elif _is_probability(fvalues):
        return Ftype.PROBABILITY
    elif _is_enum(fvalues, enum_threshold):
        return Ftype.ENUM
    elif _is_continuous(fvalues):
        return Ftype.CONTINUOUS
    else:
        raise TypeError("Unidentified feature type.")
