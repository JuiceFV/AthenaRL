"""Dataclass generally implemented for convenient way
of type-class definition (including type checking)
"""
import dataclasses
import logging
import os
from dataclasses import field  # noqa
from typing import TYPE_CHECKING, Any, Optional

import pydantic

USE_VANILLA_DATACLASS = bool(
    int(os.environ.get("USE_VANILLA_DATACLASS", False))
)
ARBITRARY_TYPES_ALLOWED = bool(
    int(os.environ.get("ARBITRARY_TYPES_ALLOWED", True))
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"USE_VANILLA_DATACLASS: {USE_VANILLA_DATACLASS}")
logger.info(f"ARBITRARY_TYPES_ALLOWED: {ARBITRARY_TYPES_ALLOWED}")

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    def dataclass(_cls: Optional[Any] = None, *, config=None, **kwargs):
        def wrap(cls):
            # We don't want to look at parent class
            if "__post_init__" in cls.__dict__:
                raise TypeError(
                    f"{cls} has __post_init__. "
                    "Please use __post_init_post_parse__ instead."
                )

            if USE_VANILLA_DATACLASS:
                try:
                    post_init_post_parse = cls.__dict__[
                        "__post_init_post_parse__"]
                    logger.info(
                        f"Setting {cls.__name__}.__post_init__ to its "
                        "__post_init_post_parse__"
                    )
                    cls.__post_init__ = post_init_post_parse
                except KeyError:
                    pass

                return dataclasses.dataclass(**kwargs)(cls)
            else:
                if ARBITRARY_TYPES_ALLOWED:

                    class Config:
                        arbitrary_types_allowed = ARBITRARY_TYPES_ALLOWED

                    if config in kwargs:
                        raise KeyError("Config duplication occures")
                    kwargs["config"] = Config

                return pydantic.dataclasses.dataclass(cls, **kwargs)

        if _cls is None:
            return wrap

        return wrap(_cls)
