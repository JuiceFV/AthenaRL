import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only


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


class ManifoldTensorboardLogger(TensorBoardLogger):
    def __init__(
        self,
        save_dir: str,
        name: Optional[str] = "default",
        version: Optional[Union[int, str]] = None,
        log_graph: bool = False,
        default_hp_metric: bool = True,
        prefix: str = "",
        **kwargs
    ) -> None:
        super().__init__(
            save_dir,
            name,
            version,
            log_graph,
            default_hp_metric,
            prefix,
            **kwargs
        )
        self.line_plot_aggregated: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
        self.line_plot_buffer: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}

    @rank_zero_only
    def log_metrics(
        self,
        metrics: Dict[str, Union[float, torch.Tensor, Dict[str, Union[float, torch.Tensor]]]],
        step: Optional[int] = None
    ) -> None:
        super().log_metrics(metrics, step)
        LocalCahceLogger.store_metrics(self, metrics, step)

    def clear_local_data(self) -> None:
        self.line_plot_aggregated = {}
        self.line_plot_buffer = {}


class LocalCahceLogger:
    @staticmethod
    def store_metrics(
        mtb_logger: ManifoldTensorboardLogger,
        metrics: Dict[str, Union[float, torch.Tensor, Dict[str, Union[float, torch.Tensor]]]],
        step: Optional[int] = None
    ) -> None:
        for plot_name, plot_value_or_dict in metrics.items():
            if isinstance(plot_value_or_dict, dict):
                if plot_name not in mtb_logger.line_plot_buffer:
                    mtb_logger.line_plot_buffer[plot_name] = {}
                for line_name, plot_value in plot_value_or_dict.items():
                    LocalCahceLogger._add_measure(mtb_logger, plot_name, line_name, plot_value, step)
            else:
                LocalCahceLogger._add_measure(mtb_logger, plot_name, "", plot_value_or_dict, step)

    @staticmethod
    def _add_measure(
        mtb_logger: ManifoldTensorboardLogger,
        plot_name: str,
        line_name: str,
        measure: Union[float, torch.Tensor],
        step: Optional[int]
    ) -> None:
        if isinstance(measure, torch.Tensor):
            measure = measure.item()

        if step is None:
            if plot_name in mtb_logger.line_plot_buffer and line_name in mtb_logger.line_plot_buffer[plot_name]:
                x = mtb_logger.line_plot_buffer[plot_name][line_name][-1][0] + 1.0
            else:
                x = 0.0
        else:
            x = float(step)

        LocalCahceLogger._create_plots_and_append(mtb_logger.line_plot_buffer, plot_name, line_name, x, measure)
        if len(mtb_logger.line_plot_buffer[plot_name][line_name] >= 50):
            mean = float(
                torch.mean(
                    torch.FloatTensor([float(p[1]) for p in mtb_logger.line_plot_buffer[plot_name][line_name]])
                ).item()
            )
            LocalCahceLogger._create_plots_and_append(mtb_logger.line_plot_aggregated, plot_name, line_name, x, mean)
            mtb_logger.line_plot_buffer[plot_name][line_name].clear()

    @staticmethod
    def _create_plots_and_append(
        plot_store: Dict[str, Dict[str, List[Tuple[float, float]]]],
        plot_name: str,
        line_name: str,
        x: int,
        y: float
    ) -> None:
        if plot_name in plot_store and line_name in plot_store[plot_name]:
            plot_store[plot_name][line_name].append((x, y))
        elif plot_name in plot_store:
            plot_store[plot_name][line_name] = [(x, y)]
        else:
            plot_store[plot_name] = {line_name: [(x, y)]}
