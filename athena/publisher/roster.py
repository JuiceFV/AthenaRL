from athena.core.registry import DiscriminatedUnion

from athena.publisher.lfs_publisher import LFSPublisher # noqa
from athena.publisher.noop_publisher import NoPublishing # noqa
from athena.publisher.publisher_base import ModelPublisher


@ModelPublisher.register()
class ModelPublisherRoster(DiscriminatedUnion):
    pass
