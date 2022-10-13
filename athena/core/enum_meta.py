from enum import Enum, EnumMeta
from typing import Union

__all__ = ["AthenaEnumMeta", "Enum"]


class AthenaEnumMeta(EnumMeta):
    def __contains__(cls, item: Union[Enum, str]) -> bool:
        return item in list(cls._value2member_map_.keys()) + list(cls._value2member_map_.values())

    def item_index(cls, item: Union[Enum, str]) -> int:
        str_item = item if isinstance(item, str) else item.value
        return list(cls._value2member_map_).index(str_item)
