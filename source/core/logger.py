import logging


class LoggerMixin:
    """
    Mixin allowing to log actions inside of the class.
    """

    @classmethod
    def _get_logger(cls):
        return logging.getLogger('.'.join([cls.__module__, cls.__name__]))

    @classmethod
    def _is_verbose(cls):
        return cls._get_logger().getEffectiveLevel() <= logging.DEBUG

    @classmethod
    def exception(cls, *args, **kwargs):
        cls._get_logger().exception(*args, **kwargs)

    @classmethod
    def critical(cls, *args, **kwargs):
        cls._get_logger().critical(*args, **kwargs)

    @classmethod
    def error(cls, *args, **kwargs):
        cls._get_logger().error(*args, **kwargs)

    @classmethod
    def warning(cls, *args, **kwargs):
        cls._get_logger().warning(*args, **kwargs)

    @classmethod
    def info(cls, *args, **kwargs):
        cls._get_logger().info(*args, **kwargs)

    @classmethod
    def debug(cls, *args, **kwargs):
        cls._get_logger().debug(*args, **kwargs)
