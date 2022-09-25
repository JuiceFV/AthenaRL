from athena.data.fap.fapper import FAPper
from athena.data.fap.spark import SparkFapper  # noqa
from athena.core.registry import DiscriminatedUnion


@FAPper.register()
class FAPperRoster(DiscriminatedUnion):
    pass
