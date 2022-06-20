import dataclasses
from typing import cast


class BaseDataClass:
    def _replace(self, **kwargs):
        return cast(type(self), dataclasses.replace(self, **kwargs))
